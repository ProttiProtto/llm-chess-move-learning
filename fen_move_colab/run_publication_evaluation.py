"""One-command vLLM quality and serving benchmark for publication results."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import aiohttp
import yaml
from transformers import AutoTokenizer

from .publication_eval import (
    audit_prompt_contract,
    build_all_moves_prompt,
    checkpoint_content_manifest,
    prepare_publication_split,
    prompt_contract,
    resolve_stop_config,
    score_records,
    write_json,
    write_jsonl,
)


CONTRACT_VERSION = "publication_vllm_eval_v3_eot_selection_provenance"


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_json(path: Path, default=None):
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 3):.2f} GiB"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _quantization_kind(model_path: Path) -> str:
    config = _load_json(model_path / "config.json")
    quant = config.get("quantization_config") or config.get("compression_config") or {}
    text = json.dumps(quant).lower() + " " + str(model_path).lower()
    if "nvfp4" in text or str(quant.get("quant_algo", "")).upper() == "NVFP4":
        return "nvfp4"
    if "fp8" in text or "float8" in text:
        return "fp8"
    return "bf16"


def _export_manifest(model_path: Path) -> Dict:
    for parent in [model_path, *model_path.parents]:
        candidate = parent / "export_manifest.json"
        if candidate.is_file():
            return _load_json(candidate)
    return {}


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _environment_manifest(model_path: Path, tokenizer) -> Dict:
    model_config = _load_json(model_path / "config.json")
    tokenizer_config = _load_json(model_path / "tokenizer_config.json")
    return {
        "python": sys.version,
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "vllm",
                "transformers",
                "tokenizers",
                "compressed-tensors",
                "llmcompressor",
                "nvidia-ml-py",
            )
        },
        "model_revision": (
            model_config.get("_commit_hash")
            or model_config.get("revision")
            or "local-checkpoint-content-hash-below"
        ),
        "model_source": model_config.get("_name_or_path"),
        "model_transformers_version": model_config.get("transformers_version"),
        "tokenizer_revision": (
            tokenizer_config.get("_commit_hash")
            or tokenizer_config.get("revision")
            or getattr(tokenizer, "_commit_hash", None)
            or "local-checkpoint-content-hash-below"
        ),
        "tokenizer_class": tokenizer.__class__.__name__,
    }


class NvmlMonitor:
    def __init__(self):
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None
        self._pynvml = None
        self._handle = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.baseline = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            self.peak = self.baseline
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            self._pynvml = None
            self._handle = None

    def _run(self) -> None:
        while not self._stop.is_set():
            used = int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            self.peak = max(self.peak, used)
            time.sleep(0.05)

    def stop(self) -> Dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return {
            "gpu_used_baseline_bytes": self.baseline or None,
            "gpu_used_peak_bytes": self.peak or None,
            "gpu_runtime_delta_peak_bytes": (
                max(0, self.peak - self.baseline) if self.peak and self.baseline else None
            ),
        }


def _generation_config(config: Mapping, stop_config: Mapping, fixed: bool = False) -> Dict:
    generation = config["generation"]
    payload = {
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
        "top_k": int(generation["top_k"]),
        "min_p": float(generation.get("min_p", 0.0)),
        "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
        "seed": int(generation["seed"]),
        "max_tokens": int(
            config["performance"]["fixed_output_tokens"]
            if fixed
            else generation["max_output_tokens"]
        ),
        "stream": True,
        "stream_options": {"include_usage": True},
        "include_stop_str_in_output": False,
        "skip_special_tokens": False,
    }
    if fixed:
        payload["ignore_eos"] = True
    else:
        payload["stop"] = list(stop_config["stop_strings"])
        payload["stop_token_ids"] = list(stop_config["stop_token_ids"])
    return payload


def _offline_generation_config(config: Mapping, stop_config: Mapping) -> Dict:
    generation = _generation_config(config, stop_config)
    return {
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "min_p": generation["min_p"],
        "repetition_penalty": generation["repetition_penalty"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stop": list(stop_config["stop_strings"]),
    }


def _run_offline_quality_worker(
    runtime_path: Path,
    prompts: Sequence[str],
    model_dir: Path,
    config: Mapping,
    stop_config: Mapping,
) -> Dict:
    spec = {
        "model_path": str(runtime_path),
        "prompts": list(prompts),
        "generation": _offline_generation_config(config, stop_config),
        "server": dict(config["server"]),
        "warmup_positions": int(config["run"].get("quality_warmup_positions", 8)),
    }
    spec_path = model_dir / "offline_worker_spec.json"
    output_path = model_dir / "offline_worker_result.json"
    log_path = model_dir / "offline_worker.log"
    write_json(spec_path, spec)
    command = [
        sys.executable,
        "-m",
        "fen_move_colab.offline_vllm_worker",
        "--spec",
        str(spec_path),
        "--output",
        str(output_path),
    ]
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    print(f"[parent] Worker spec: {spec_path}", flush=True)
    print(f"[parent] Worker log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    worker = _load_json(output_path)
    if return_code or worker.get("status") != "completed":
        raise RuntimeError(
            f"Offline vLLM worker failed with exit code {return_code}: "
            f"{worker.get('error', 'no worker report')} (see {log_path})"
        )
    return worker


async def _request_completion(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    prompt,
    generation: Mapping,
    semaphore: asyncio.Semaphore,
    prompt_tokens: int,
) -> Dict:
    payload = {"model": model_name, "prompt": prompt, **generation}
    started = time.perf_counter()
    first_token_at = None
    completion_parts: List[str] = []
    finish_reason = None
    usage = None
    async with semaphore:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"vLLM returned HTTP {response.status}: {body[:1000]}")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if text and first_token_at is None:
                    first_token_at = time.perf_counter()
                completion_parts.append(text)
                finish_reason = choices[0].get("finish_reason") or finish_reason
    finished = time.perf_counter()
    return {
        "completion": "".join(completion_parts),
        "ttft_seconds": (
            first_token_at - started if first_token_at is not None else finished - started
        ),
        "e2e_latency_seconds": finished - started,
        "finish_reason": finish_reason,
        "usage": usage,
        "prompt_tokens": prompt_tokens,
    }


async def _run_client_batch(
    base_url: str,
    model_name: str,
    prompts: Sequence,
    prompt_token_counts: Sequence[int],
    generation: Mapping,
    concurrency: int,
) -> Dict:
    timeout = aiohttp.ClientTimeout(total=None, connect=60)
    connector = aiohttp.TCPConnector(limit=0)
    semaphore = asyncio.Semaphore(int(concurrency))
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                _request_completion(
                    session,
                    f"{base_url}/v1/completions",
                    model_name,
                    prompt,
                    generation,
                    semaphore,
                    prompt_tokens,
                )
            )
            for prompt, prompt_tokens in zip(prompts, prompt_token_counts)
        ]
        outputs = await asyncio.gather(*tasks)
    duration = time.perf_counter() - started
    return {"duration_seconds": duration, "outputs": outputs}


def _wait_for_server(process: subprocess.Popen, port: int, timeout_seconds: int) -> float:
    started = time.perf_counter()
    health_url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() - started < timeout_seconds:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited during startup with code {process.returncode}.")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return time.perf_counter() - started
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"vLLM server was not ready after {timeout_seconds} seconds.")


def _server_command(
    model_path: Path,
    port: int,
    quantization: str,
    server: Mapping,
    served_model_name: str,
) -> List[str]:
    executable = shutil.which("vllm")
    if not executable:
        raise FileNotFoundError("The vllm CLI is not installed in this environment.")
    command = [
        executable,
        "serve",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(server["max_model_len"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(server["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--generation-config",
        "vllm",
        "--scheduling-policy",
        str(server.get("scheduling_policy", "fcfs")),
        "--language-model-only",
        "--disable-log-stats",
    ]
    if server.get("enable_prefix_caching", True):
        command.append("--enable-prefix-caching")
    if server.get("enable_chunked_prefill", True):
        command.append("--enable-chunked-prefill")
    if server.get("enforce_eager", False):
        command.append("--enforce-eager")
    if quantization == "fp8":
        command.extend(["--quantization", "compressed-tensors"])
    elif quantization == "nvfp4":
        command.extend(["--quantization", "modelopt"])
        kernel_config = {
            "linear_backend": server["nvfp4_linear_backend"],
            "moe_backend": server["nvfp4_moe_backend"],
        }
        command.extend(["--kernel-config", json.dumps(kernel_config, separators=(",", ":"))])
    return command


def _parse_server_memory(log_text: str) -> Dict:
    patterns = {
        "model_loading_gib": r"Model loading took ([0-9.]+) GiB",
        "kv_cache_available_gib": r"Available KV cache memory: ([0-9.]+) GiB",
        "cuda_graph_memory_gib": r"Graph capturing finished.*took ([0-9.]+) GiB",
        "kv_cache_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
        "max_concurrency_at_model_len": r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x",
    }
    output = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, log_text)
        if not matches:
            output[key] = None
        elif key == "kv_cache_tokens":
            output[key] = int(matches[-1].replace(",", ""))
        else:
            output[key] = float(matches[-1])
    return output


def _performance_summary(batch: Mapping, tokenizer) -> Dict:
    outputs = list(batch["outputs"])
    duration = float(batch["duration_seconds"])
    ttfts = [float(row["ttft_seconds"]) for row in outputs]
    e2es = [float(row["e2e_latency_seconds"]) for row in outputs]
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in outputs)
    output_tokens = sum(
        int((row.get("usage") or {}).get("completion_tokens") or 0)
        or len(tokenizer.encode(row["completion"], add_special_tokens=False))
        for row in outputs
    )
    return {
        "requests": len(outputs),
        "duration_seconds": duration,
        "requests_per_second": len(outputs) / duration,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / duration,
        "total_tokens_per_second": (prompt_tokens + output_tokens) / duration,
        "ttft_p50_seconds": _percentile(ttfts, 0.50),
        "ttft_p95_seconds": _percentile(ttfts, 0.95),
        "e2e_latency_p50_seconds": _percentile(e2es, 0.50),
        "e2e_latency_p95_seconds": _percentile(e2es, 0.95),
    }


def _fixed_prompts(tokenizer, natural_prompts: Sequence[str], length: int) -> List[List[int]]:
    fixed = []
    fallback_id = int(tokenizer.eos_token_id or tokenizer.pad_token_id or 0)
    for prompt in natural_prompts:
        token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        if not token_ids:
            token_ids = [fallback_id]
        while len(token_ids) < length:
            token_ids.extend(token_ids[: length - len(token_ids)])
        fixed.append([int(value) for value in token_ids[:length]])
    return fixed


def _scenario_request_count(performance: Mapping, concurrency: int) -> int:
    configured = performance.get("requests_per_concurrency", {})
    return int(configured.get(str(concurrency), max(32, concurrency * 2)))


def evaluate_model(
    model_spec: Mapping,
    records: Sequence[Mapping],
    run_dir: Path,
    config: Mapping,
    split_manifest: Mapping,
) -> Dict:
    name = str(model_spec["name"])
    source_path = Path(model_spec["path"]).expanduser().resolve()
    print(f"[{name}] Starting publication quality evaluation.", flush=True)
    print(f"[{name}] Source checkpoint: {source_path}", flush=True)
    if not (source_path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid checkpoint for {name}: {source_path}")
    print(f"[{name}] Hashing source checkpoint files...", flush=True)
    manifest_started = time.perf_counter()
    source_checkpoint_manifest = checkpoint_content_manifest(source_path)
    print(
        f"[{name}] Source hash completed in {time.perf_counter() - manifest_started:.2f}s; "
        f"size={_format_bytes(source_checkpoint_manifest['checkpoint_size_bytes'])}; "
        f"sha256={source_checkpoint_manifest['checkpoint_content_sha256'][:16]}...",
        flush=True,
    )
    model_dir = run_dir / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = model_dir / "report.json"
    contract_hash = hashlib.sha256(
        json.dumps(
            {
                "contract": CONTRACT_VERSION,
                "model": model_spec,
                "quality_backend": config["run"].get(
                    "quality_backend", "offline_vllm_llm_generate"
                ),
                "generation": config["generation"],
                "server": config["server"],
                "performance": config["performance"],
                "checkpoint_content_sha256": source_checkpoint_manifest[
                    "checkpoint_content_sha256"
                ],
                "final_test_identity_sha256": split_manifest[
                    "final_test_identity_sha256"
                ],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    previous = _load_json(report_path)
    if config["run"].get("resume", True) and previous.get("status") == "completed":
        if previous.get("evaluation_contract_sha256") == contract_hash:
            print(
                f"[{name}] Reusing contract-matched completed report. "
                "Set run.resume=false to force fresh inference.",
                flush=True,
            )
            previous["resumed_from_cache"] = True
            write_json(report_path, previous)
            return previous

    report = {
        "name": name,
        "status": "running",
        "contract": CONTRACT_VERSION,
        "evaluation_contract_sha256": contract_hash,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_model_path": str(source_path),
        "quality_backend": config["run"].get(
            "quality_backend", "offline_vllm_llm_generate"
        ),
        "resumed_from_cache": False,
    }
    write_json(report_path, report)

    runtime_path = source_path
    copy_seconds = 0.0
    local_cache = Path(config["run"].get("local_model_cache", "/content/publication_model"))
    if config["run"].get("copy_model_to_local_ssd", True):
        print(
            f"[{name}] Copying {_format_bytes(source_checkpoint_manifest['checkpoint_size_bytes'])} "
            f"from Drive to local SSD: {local_cache}",
            flush=True,
        )
        shutil.rmtree(local_cache, ignore_errors=True)
        started = time.perf_counter()
        shutil.copytree(source_path, local_cache)
        copy_seconds = time.perf_counter() - started
        runtime_path = local_cache
        print(f"[{name}] Local copy completed in {copy_seconds:.2f}s.", flush=True)

    print(f"[{name}] Verifying local checkpoint copy hash...", flush=True)
    checkpoint_manifest = checkpoint_content_manifest(runtime_path)
    if (
        checkpoint_manifest["checkpoint_content_sha256"]
        != source_checkpoint_manifest["checkpoint_content_sha256"]
    ):
        raise RuntimeError(f"Local checkpoint copy failed content verification for {name}.")
    write_json(model_dir / "checkpoint_manifest.json", checkpoint_manifest)
    print(f"[{name}] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(runtime_path, trust_remote_code=True)
    stop_config = resolve_stop_config(tokenizer)
    print(
        f"[{name}] Stop strategy: strings={stop_config['stop_strings']}, "
        f"verified_token_ids={stop_config['stop_token_ids']}, "
        f"encoded_ids={stop_config['end_of_turn_encoded_ids']}",
        flush=True,
    )
    if stop_config.get("warning"):
        print(f"[{name}] Stop warning: {stop_config['warning']}", flush=True)
    print(f"[{name}] Building and tokenizing {len(records):,} prompts...", flush=True)
    prompts = [build_all_moves_prompt(row["fen"]) for row in records]
    prompt_token_counts = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    quantization = _quantization_kind(runtime_path)
    export_manifest = _export_manifest(source_path)
    print(
        f"[{name}] Prompt tokens: min={min(prompt_token_counts)}, "
        f"max={max(prompt_token_counts)}, "
        f"avg={sum(prompt_token_counts) / len(prompt_token_counts):.1f}; "
        f"quantization={quantization}",
        flush=True,
    )

    report.update(
        {
            "runtime_model_path": str(runtime_path),
            "quantization_kind": quantization,
            "checkpoint": checkpoint_manifest,
            "source_checkpoint": source_checkpoint_manifest,
            "checkpoint_directory_size_bytes": _directory_size(source_path),
            "export_manifest": export_manifest,
            "environment": _environment_manifest(runtime_path, tokenizer),
            "prompt_contract": prompt_contract(),
            "stop_config": stop_config,
            "generation_config": _generation_config(config, stop_config),
            "server_config": dict(config["server"]),
            "model_copy_seconds": copy_seconds,
        }
    )
    write_json(report_path, report)

    quality_backend = report["quality_backend"]
    if quality_backend == "offline_vllm_llm_generate":
        try:
            print(f"[{name}] Launching isolated offline vLLM worker...", flush=True)
            worker = _run_offline_quality_worker(
                runtime_path, prompts, model_dir, config, stop_config
            )
            outputs = worker["results"]
            print(f"[{name}] Scoring {len(outputs):,} completions with python-chess...", flush=True)
            quality_metrics, details = score_records(
                records, [row["completion"] for row in outputs]
            )
            for detail, output in zip(details, outputs):
                detail.update(
                    {
                        "finish_reason": output.get("finish_reason"),
                        "prompt_tokens": output.get("prompt_tokens"),
                        "completion_tokens": output.get("completion_tokens"),
                    }
                )
            predictions_path = model_dir / "predictions.jsonl"
            write_jsonl(predictions_path, details)
            overall = quality_metrics["overall"]
            print(
                f"[{name}] Scoring complete: F1={overall['macro_f1']:.4f}, "
                f"exact={overall['set_exact_rate']:.4f}, "
                f"precision={overall['macro_precision']:.4f}, "
                f"recall={overall['macro_recall']:.4f}.",
                flush=True,
            )
            report.update(
                {
                    "status": "completed",
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "quality": quality_metrics,
                    "quality_performance": {
                        key: value for key, value in worker.items()
                        if key not in {"results", "memory", "environment", "llm_kwargs"}
                    },
                    "worker_environment": worker.get("environment"),
                    "vllm_llm_kwargs": worker.get("llm_kwargs"),
                    "memory": {
                        "model_safetensor_weight_bytes": sum(
                            entry["size_bytes"]
                            for entry in checkpoint_manifest["files"]
                            if entry["path"].endswith(".safetensors")
                        ),
                        "checkpoint_artifact_bytes": checkpoint_manifest[
                            "checkpoint_size_bytes"
                        ],
                        **(worker.get("memory") or {}),
                        "note": (
                            "Offline-worker NVML peak includes weights, activations, CUDA "
                            "graphs, and vLLM KV cache; it is not model-weight memory."
                        ),
                    },
                    "performance": [],
                    "performance_status": "not_run_separate_optional_stage",
                    "predictions_path": str(predictions_path.resolve()),
                }
            )
            (model_dir / "offline_worker_spec.json").unlink(missing_ok=True)
            (model_dir / "offline_worker_result.json").unlink(missing_ok=True)
            write_json(report_path, report)
            print(f"[{name}] COMPLETED. Report: {report_path}", flush=True)
            return report
        except Exception as exc:
            report.update(
                {
                    "status": "failed",
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failure_stage": "offline_vllm_quality",
                }
            )
            write_json(report_path, report)
            print(
                f"[{name}] FAILED during offline vLLM quality: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                flush=True,
            )
            raise
        finally:
            if runtime_path == local_cache:
                shutil.rmtree(local_cache, ignore_errors=True)

    if quality_backend != "online_vllm_server":
        raise ValueError(f"Unsupported quality_backend: {quality_backend}")

    port = _free_port()
    served_model_name = "publication-model"
    server_log_path = model_dir / "server.log"
    command = _server_command(
        runtime_path, port, quantization, config["server"], served_model_name
    )
    report["server_command"] = command
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    monitor = NvmlMonitor()
    monitor.start()
    write_json(report_path, report)
    process = None
    log_handle = server_log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        startup_seconds = _wait_for_server(
            process, port, int(config["server"].get("startup_timeout_seconds", 1200))
        )
        base_url = f"http://127.0.0.1:{port}"

        quality_generation = _generation_config(config, stop_config)
        quality_batch = asyncio.run(
            _run_client_batch(
                base_url,
                served_model_name,
                prompts,
                prompt_token_counts,
                quality_generation,
                int(config["run"]["quality_concurrency"]),
            )
        )
        completions = [row["completion"] for row in quality_batch["outputs"]]
        quality_metrics, details = score_records(records, completions)
        for detail, timing in zip(details, quality_batch["outputs"]):
            detail.update(
                {
                    "ttft_seconds": timing["ttft_seconds"],
                    "e2e_latency_seconds": timing["e2e_latency_seconds"],
                    "finish_reason": timing["finish_reason"],
                }
            )
        write_jsonl(model_dir / "predictions.jsonl", details)

        performance_results = []
        performance = config["performance"]
        natural_pool = prompts
        natural_token_pool = prompt_token_counts
        fixed_pool = _fixed_prompts(
            tokenizer, prompts, int(performance["fixed_input_tokens"])
        )
        fixed_token_pool = [int(performance["fixed_input_tokens"])] * len(fixed_pool)
        for mode in performance["modes"]:
            fixed = mode == "fixed_tokens"
            generation = _generation_config(config, stop_config, fixed=fixed)
            source_prompts = fixed_pool if fixed else natural_pool
            source_lengths = fixed_token_pool if fixed else natural_token_pool
            for concurrency in performance["concurrencies"]:
                request_count = _scenario_request_count(performance, int(concurrency))
                selected_prompts = [
                    source_prompts[index % len(source_prompts)] for index in range(request_count)
                ]
                selected_lengths = [
                    source_lengths[index % len(source_lengths)] for index in range(request_count)
                ]
                for repetition in range(1, int(performance["repetitions"]) + 1):
                    batch = asyncio.run(
                        _run_client_batch(
                            base_url,
                            served_model_name,
                            selected_prompts,
                            selected_lengths,
                            generation,
                            int(concurrency),
                        )
                    )
                    summary = _performance_summary(batch, tokenizer)
                    summary.update(
                        {
                            "mode": mode,
                            "concurrency": int(concurrency),
                            "repetition": repetition,
                            "max_num_seqs": int(config["server"]["max_num_seqs"]),
                            "max_num_batched_tokens": int(
                                config["server"]["max_num_batched_tokens"]
                            ),
                            "generation_config": generation,
                        }
                    )
                    performance_results.append(summary)
                    write_json(model_dir / "performance.json", performance_results)

        report.update(
            {
                "status": "completed",
                "server_startup_seconds": startup_seconds,
                "quality": quality_metrics,
                "quality_performance": _performance_summary(quality_batch, tokenizer),
                "performance": performance_results,
                "predictions_path": str((model_dir / "predictions.jsonl").resolve()),
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log_handle.close()
        memory = monitor.stop()
        log_text = server_log_path.read_text(encoding="utf-8", errors="replace")
        report["memory"] = {
            "model_safetensor_weight_bytes": sum(
                entry["size_bytes"]
                for entry in checkpoint_manifest["files"]
                if entry["path"].endswith(".safetensors")
            ),
            "checkpoint_artifact_bytes": checkpoint_manifest["checkpoint_size_bytes"],
            **memory,
            **_parse_server_memory(log_text),
            "note": (
                "NVML peak includes model, activations, CUDA graphs, and reserved KV cache; "
                "it is not model-weight memory."
            ),
        }
        write_json(report_path, report)
        if runtime_path == local_cache:
            shutil.rmtree(local_cache, ignore_errors=True)
    return report


def _write_comparison_tables(run_dir: Path, results: Sequence[Mapping]) -> None:
    quality_rows = []
    performance_rows = []
    for result in results:
        quality = result.get("quality") or {}
        overall = quality.get("overall") or {}
        check = quality.get("in_check") or {}
        non_check = quality.get("not_in_check") or {}
        quality_rows.append(
            {
                "name": result.get("name"),
                "status": result.get("status"),
                "quantization": result.get("quantization_kind"),
                "move_set_f1": overall.get("macro_f1"),
                "exact_position_accuracy": overall.get("set_exact_rate"),
                "precision": overall.get("macro_precision"),
                "recall": overall.get("macro_recall"),
                "in_check_f1": check.get("macro_f1"),
                "in_check_exact_accuracy": check.get("set_exact_rate"),
                "not_in_check_f1": non_check.get("macro_f1"),
                "not_in_check_exact_accuracy": non_check.get("set_exact_rate"),
                "illegal_moves_from_check_rate": quality.get(
                    "illegal_moves_from_check_rate"
                ),
            }
        )
        for row in result.get("performance") or []:
            performance_rows.append({"name": result.get("name"), **row})

    for path, rows in (
        (run_dir / "quality_comparison.csv", quality_rows),
        (run_dir / "performance_comparison.csv", performance_rows),
    ):
        if not rows:
            continue
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(config: Mapping) -> Dict:
    output_root = Path(config["run"]["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print("===== PUBLICATION EVALUATION INITIALIZATION =====", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(
        f"Quality backend: {config['run'].get('quality_backend', 'offline_vllm_llm_generate')}; "
        f"models: {len(config['models'])}; resume: {config['run'].get('resume', True)}; "
        f"fail_fast: {config['run'].get('fail_fast', True)}",
        flush=True,
    )
    config_payload = json.loads(json.dumps(config))
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_dir = output_root / f"publication_eval_{config_hash[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "evaluation_config.json", config_payload)
    print(f"Evaluation contract hash: {config_hash}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)

    print("Preparing leakage-audited publication split...", flush=True)
    split = prepare_publication_split(
        config["data"]["validation_path"],
        config["data"]["training_paths"],
        run_dir / "testset",
        int(config["data"]["checkpoint_selection_positions"]),
    )
    records = split["records"]
    print(
        f"Final test positions: {len(records):,}; checkpoint-selection positions excluded: "
        f"{len(split['selection_records']):,}; test identity: "
        f"{split['manifest']['final_test_identity_sha256']}",
        flush=True,
    )
    print("Auditing evaluation prompt against the training prompt contract...", flush=True)
    prompt_audit = audit_prompt_contract(records, config["data"]["all_moves_training_path"])
    write_json(run_dir / "prompt_audit.json", prompt_audit)
    print(
        f"Prompt audit: {prompt_audit.get('evaluation_kind')}; "
        f"contract={prompt_contract()['prompt_contract_sha256']}",
        flush=True,
    )

    results = []
    for index, model in enumerate(config["models"], start=1):
        print(
            f"\n===== PUBLICATION EVAL {index}/{len(config['models'])}: "
            f"{model['name']} =====",
            flush=True,
        )
        try:
            results.append(
                evaluate_model(model, records, run_dir, config, split["manifest"])
            )
        except Exception as exc:
            print(f"FAILED {model['name']}: {type(exc).__name__}: {exc}", flush=True)
            failed_report_path = run_dir / "models" / model["name"] / "report.json"
            failed = _load_json(failed_report_path)
            failed.update(
                {
                    "name": model["name"],
                    "status": "failed",
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": failed.get("error_type") or type(exc).__name__,
                    "error": failed.get("error") or str(exc),
                    "traceback": failed.get("traceback") or traceback.format_exc(),
                }
            )
            write_json(failed_report_path, failed)
            results.append(failed)
            if config["run"].get("fail_fast", True):
                raise

    summary = {
        "contract": CONTRACT_VERSION,
        "config_sha256": config_hash,
        "config_path": str((run_dir / "evaluation_config.json").resolve()),
        "split_manifest": split["manifest"],
        "prompt_audit": prompt_audit,
        "results": results,
    }
    write_json(run_dir / "comparison.json", summary)
    _write_comparison_tables(run_dir, results)
    completed = sum(row.get("status") == "completed" for row in results)
    failed = sum(row.get("status") == "failed" for row in results)
    print(
        f"===== EVALUATION FINISHED: completed={completed}, failed={failed}, "
        f"total={len(results)} =====",
        flush=True,
    )
    print(f"Comparison report: {run_dir / 'comparison.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary = run(config)
    print(
        json.dumps(
            {
                "models": len(summary["results"]),
                "completed": sum(row.get("status") == "completed" for row in summary["results"]),
                "config_sha256": summary["config_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
