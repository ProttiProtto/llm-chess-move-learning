"""Run reproducible online vLLM concurrency benchmarks without quality inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import yaml

from .publication_eval import (
    END_OF_TURN,
    build_all_moves_prompt,
    checkpoint_content_manifest,
    write_json,
    write_jsonl,
)


CONTRACT_VERSION = "fen_move_vllm_parallelism_v2"


def _load_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload[-1] if payload else {}
    return payload


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


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
    if server.get("enable_prefix_caching", False):
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


def _tail(path: Path, lines: int = 60) -> str:
    if not path.is_file():
        return ""
    return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])


def _wait_for_server(
    process: subprocess.Popen,
    port: int,
    timeout_seconds: int,
    log_path: Path,
) -> float:
    started = time.perf_counter()
    next_update = started
    health_url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() - started < timeout_seconds:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited during startup with code {process.returncode}.\n"
                f"Last server log lines:\n{_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    elapsed = time.perf_counter() - started
                    print(f"Server is healthy after {elapsed:.1f}s.", flush=True)
                    return elapsed
        except Exception:
            pass
        now = time.perf_counter()
        if now >= next_update:
            print(
                f"Waiting for vLLM server: {now - started:.0f}s elapsed; "
                f"server_alive={process.poll() is None}",
                flush=True,
            )
            next_update = now + 15
        time.sleep(2)
    raise TimeoutError(
        f"vLLM server was not ready after {timeout_seconds}s.\n"
        f"Last server log lines:\n{_tail(log_path)}"
    )


def _stop_process_tree(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=15)


def _request_count(benchmark: Mapping, concurrency: int) -> int:
    configured = benchmark.get("requests_per_concurrency", {})
    count = int(configured.get(str(concurrency), max(128, concurrency * 4)))
    if count < concurrency:
        raise ValueError(
            f"Request count {count} must be at least concurrency {concurrency}."
        )
    return count


def _scenario_plan(benchmark: Mapping) -> List[Dict]:
    scenarios = []
    for mode in benchmark["modes"]:
        if mode not in {"natural_fen", "fixed_tokens"}:
            raise ValueError(f"Unsupported benchmark mode: {mode}")
        for concurrency in benchmark["concurrencies"]:
            concurrency = int(concurrency)
            for repetition in range(1, int(benchmark["repetitions"]) + 1):
                scenarios.append(
                    {
                        "mode": mode,
                        "concurrency": concurrency,
                        "repetition": repetition,
                        "requests": _request_count(benchmark, concurrency),
                    }
                )
    return scenarios


def _bench_help() -> str:
    executable = shutil.which("vllm")
    if not executable:
        raise FileNotFoundError("vllm is not installed.")
    required = [
        "--max-concurrency",
        "--save-detailed",
        "--extra-body",
        "--percentile-metrics",
        "--metric-percentiles",
        "--skip-chat-template",
        "--custom-output-len",
        "--random-input-len",
        "--random-output-len",
        "--ignore-eos",
    ]
    attempts = []
    for help_argument in ("--help=all", "--help"):
        result = subprocess.run(
            [executable, "bench", "serve", help_argument],
            text=True,
            capture_output=True,
        )
        help_text = "\n".join(part for part in (result.stdout, result.stderr) if part)
        attempts.append(f"$ vllm bench serve {help_argument}\n{help_text}")
        if result.returncode == 0 and not [flag for flag in required if flag not in help_text]:
            return help_text

    combined_help = "\n\n".join(attempts)
    missing = [flag for flag in required if flag not in combined_help]
    if missing:
        raise RuntimeError(
            "vllm bench serve help did not expose required options. "
            "The CLI may be incompatible or may only be showing abbreviated help. "
            f"Missing: {missing}\n{combined_help[-6000:]}"
        )
    return combined_help


def _benchmark_command(
    model_path: Path,
    served_model_name: str,
    port: int,
    scenario: Mapping,
    config: Mapping,
    natural_dataset: Path,
    result_dir: Path,
    help_text: str,
) -> List[str]:
    benchmark = config["benchmark"]
    generation = config["generation"]
    executable = shutil.which("vllm")
    command = [
        executable,
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--endpoint",
        "/v1/completions",
        "--model",
        served_model_name,
        "--tokenizer",
        str(model_path),
        "--num-prompts",
        str(scenario["requests"]),
        "--max-concurrency",
        str(scenario["concurrency"]),
        "--request-rate",
        str(benchmark.get("request_rate", "inf")),
        "--num-warmups",
        str(benchmark.get("warmup_requests", 16)),
        "--seed",
        str(generation["seed"]),
        "--temperature",
        str(generation["temperature"]),
        "--top-p",
        str(generation["top_p"]),
        "--top-k",
        str(generation["top_k"]),
        "--min-p",
        str(generation.get("min_p", 0.0)),
        "--repetition-penalty",
        str(generation.get("repetition_penalty", 1.0)),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95",
        "--save-result",
        "--save-detailed",
        "--disable-shuffle",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        "result.json",
    ]
    if scenario["mode"] == "natural_fen":
        skip_flag = (
            "--custom-skip-chat-template"
            if "--custom-skip-chat-template" in help_text
            else "--skip-chat-template"
        )
        extra_body = {
            "stop": [END_OF_TURN],
            "include_stop_str_in_output": False,
            "skip_special_tokens": False,
        }
        command.extend(
            [
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(natural_dataset),
                "--custom-output-len",
                str(generation["natural_max_output_tokens"]),
                skip_flag,
                "--extra-body",
                json.dumps(extra_body, separators=(",", ":")),
            ]
        )
    else:
        command.extend(
            [
                "--dataset-name",
                "random",
                "--random-input-len",
                str(benchmark["fixed_input_tokens"]),
                "--random-output-len",
                str(benchmark["fixed_output_tokens"]),
                "--random-range-ratio",
                "0",
                "--ignore-eos",
            ]
        )
    return command


def _stream_command(command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> int:
    print("Running benchmark command:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(env),
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="", flush=True)
        return process.wait()


def _scalar_result(payload: Mapping) -> Dict:
    return {
        key: value
        for key, value in payload.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def _validate_benchmark_result(payload: Mapping, expected_requests: int) -> None:
    completed = payload.get("completed")
    if completed is None:
        raise ValueError("vLLM benchmark result does not contain a completed request count.")
    if int(completed) != int(expected_requests):
        raise ValueError(
            f"vLLM completed {completed} of {expected_requests} measured requests."
        )
    failed = payload.get("failed")
    if failed is not None and int(failed) != 0:
        raise ValueError(f"vLLM reported {failed} failed measured requests.")


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _quality_evidence(comparison: Mapping) -> Dict[str, Dict]:
    evidence = {}
    for result in comparison.get("results", []):
        evidence[str(result.get("name"))] = {
            "quality_status": result.get("status"),
            "evaluation_contract_sha256": result.get("evaluation_contract_sha256"),
            "checkpoint_content_sha256": (
                (result.get("source_checkpoint") or {}).get("checkpoint_content_sha256")
                or (result.get("checkpoint") or {}).get("checkpoint_content_sha256")
            ),
        }
    return evidence


def _validate_quality_provenance(
    model_name: str, quality_evidence: Mapping, serving_manifest: Mapping
) -> Dict:
    """Require quality evidence for the exact checkpoint being benchmarked."""
    if not quality_evidence:
        raise ValueError(f"No quality evidence is recorded for {model_name}.")
    if quality_evidence.get("quality_status") != "completed":
        raise ValueError(
            f"Quality evaluation for {model_name} is not completed: "
            f"{quality_evidence.get('quality_status')!r}."
        )
    expected_hash = str(quality_evidence.get("checkpoint_content_sha256") or "")
    actual_hash = str(serving_manifest.get("checkpoint_content_sha256") or "")
    if not expected_hash:
        raise ValueError(
            f"Quality evaluation for {model_name} has no checkpoint content hash."
        )
    if not actual_hash:
        raise ValueError(f"Serving checkpoint for {model_name} has no content hash.")
    if expected_hash != actual_hash:
        raise ValueError(
            f"Checkpoint hash mismatch for {model_name}: quality={expected_hash}, "
            f"serving={actual_hash}."
        )
    return {
        "quality_status": quality_evidence["quality_status"],
        "quality_checkpoint_content_sha256": expected_hash,
        "serving_checkpoint_content_sha256": actual_hash,
        "checkpoint_hash_match": True,
    }


def _signature_config(config: Mapping) -> Dict:
    """Remove transient runtime ports before hashing a resumable run contract."""
    payload = json.loads(json.dumps(config))

    def strip_transient(value):
        if isinstance(value, dict):
            return {
                key: strip_transient(item)
                for key, item in value.items()
                if key != "port"
            }
        if isinstance(value, list):
            return [strip_transient(item) for item in value]
        return value

    return strip_transient(payload)


def _prepare_natural_dataset(testset_path: Path, output_path: Path) -> Dict:
    records = _read_jsonl(testset_path)
    prompts = []
    for index, record in enumerate(records):
        fen = record.get("fen")
        if not fen:
            raise ValueError(f"Missing FEN in test row {index}")
        prompts.append({"prompt": build_all_moves_prompt(str(fen))})
    write_jsonl(output_path, prompts)
    return {
        "source_path": str(testset_path),
        "source_sha256": _sha256_file(testset_path),
        "prompt_dataset_path": str(output_path),
        "prompt_dataset_sha256": _sha256_file(output_path),
        "positions": len(records),
    }


def _environment() -> Dict:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
    )
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
                "nvidia-ml-py",
            )
        },
        "gpu": gpu.stdout.strip(),
    }


def _model_scenarios(
    model_spec: Mapping,
    model_dir: Path,
    runtime_path: Path,
    port: int,
    config: Mapping,
    natural_dataset: Path,
    help_text: str,
    env: Mapping[str, str],
) -> List[Dict]:
    rows = []
    for scenario in _scenario_plan(config["benchmark"]):
        slug = (
            f"{scenario['mode']}_c{scenario['concurrency']}_"
            f"r{scenario['repetition']}"
        )
        scenario_dir = model_dir / "scenarios" / slug
        scenario_dir.mkdir(parents=True, exist_ok=True)
        result_path = scenario_dir / "result.json"
        manifest_path = scenario_dir / "scenario.json"
        command = _benchmark_command(
            runtime_path,
            "parallelism-model",
            port,
            scenario,
            config,
            natural_dataset,
            scenario_dir,
            help_text,
        )
        contract = {
            "contract": CONTRACT_VERSION,
            "model": dict(model_spec),
            "scenario": scenario,
            "generation": config["generation"],
            "server": config["server"],
            "command": command,
        }
        contract_sha = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest()
        previous = _load_json(manifest_path)
        reuse = False
        if (
            config["run"].get("resume", True)
            and previous.get("status") == "completed"
            and previous.get("contract_sha256") == contract_sha
            and result_path.is_file()
        ):
            payload = _load_json(result_path)
            try:
                _validate_benchmark_result(payload, int(scenario["requests"]))
                print(f"Reusing completed scenario {model_spec['name']} / {slug}.", flush=True)
                reuse = True
            except ValueError:
                reuse = False
        if not reuse:
            result_path.unlink(missing_ok=True)
            write_json(
                manifest_path,
                {
                    **contract,
                    "contract_sha256": contract_sha,
                    "status": "running",
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            started = time.perf_counter()
            return_code = _stream_command(command, scenario_dir / "benchmark.log", env)
            elapsed = time.perf_counter() - started
            if return_code or not result_path.is_file():
                failure = {
                    **contract,
                    "contract_sha256": contract_sha,
                    "status": "failed",
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed,
                    "return_code": return_code,
                    "error": _tail(scenario_dir / "benchmark.log", 100),
                }
                write_json(manifest_path, failure)
                raise RuntimeError(
                    f"Benchmark scenario failed: {model_spec['name']} / {slug}; "
                    f"exit={return_code}. See {scenario_dir / 'benchmark.log'}"
                )
            payload = _load_json(result_path)
            try:
                _validate_benchmark_result(payload, int(scenario["requests"]))
            except ValueError as exc:
                failure = {
                    **contract,
                    "contract_sha256": contract_sha,
                    "status": "failed",
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed,
                    "return_code": return_code,
                    "error": str(exc),
                }
                write_json(manifest_path, failure)
                raise
            write_json(
                manifest_path,
                {
                    **contract,
                    "contract_sha256": contract_sha,
                    "status": "completed",
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed,
                    "return_code": return_code,
                },
            )
        else:
            payload = _load_json(result_path)
        rows.append(
            {
                "model": model_spec["name"],
                "mode": scenario["mode"],
                "concurrency": scenario["concurrency"],
                "repetition": scenario["repetition"],
                "requested_prompts": scenario["requests"],
                "contract_sha256": contract_sha,
                **_scalar_result(payload),
                "result_path": str(result_path),
            }
        )
    return rows


def evaluate_model(
    model_spec: Mapping,
    run_dir: Path,
    config: Mapping,
    natural_dataset: Path,
    help_text: str,
    quality_evidence: Mapping[str, Mapping],
) -> Dict:
    name = str(model_spec["name"])
    source_path = Path(model_spec["path"]).expanduser().resolve()
    if not (source_path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid model checkpoint for {name}: {source_path}")
    model_dir = run_dir / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = model_dir / "report.json"
    serving_manifest = checkpoint_content_manifest(source_path)
    provenance = _validate_quality_provenance(
        name, quality_evidence.get(name), serving_manifest
    )
    report = {
        "name": name,
        "status": "running",
        "contract": CONTRACT_VERSION,
        "source_model_path": str(source_path),
        "quality_evidence": quality_evidence.get(name),
        "serving_checkpoint": serving_manifest,
        "checkpoint_provenance": provenance,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(report_path, report)

    local_cache = Path(config["run"].get("local_model_cache", "/content/parallelism_model"))
    runtime_path = source_path
    copy_seconds = 0.0
    if config["run"].get("copy_model_to_local_ssd", True):
        shutil.rmtree(local_cache, ignore_errors=True)
        print(f"[{name}] Copying checkpoint from Drive to {local_cache}...", flush=True)
        started = time.perf_counter()
        shutil.copytree(source_path, local_cache)
        copy_seconds = time.perf_counter() - started
        runtime_path = local_cache
        print(f"[{name}] Copy completed in {copy_seconds:.1f}s.", flush=True)

    quantization = _quantization_kind(runtime_path)
    port = _free_port()
    server_log = model_dir / "server.log"
    server_command = _server_command(
        runtime_path,
        port,
        quantization,
        config["server"],
        "parallelism-model",
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    process = None
    log_handle = None
    try:
        print(f"[{name}] Starting {quantization} vLLM server on port {port}.", flush=True)
        print("Server command:", " ".join(server_command), flush=True)
        log_handle = server_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            server_command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        startup_seconds = _wait_for_server(
            process,
            port,
            int(config["server"].get("startup_timeout_seconds", 1200)),
            server_log,
        )
        rows = _model_scenarios(
            model_spec,
            model_dir,
            runtime_path,
            port,
            config,
            natural_dataset,
            help_text,
            env,
        )
        report.update(
            {
                "status": "completed",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "quantization_kind": quantization,
                "model_copy_seconds": copy_seconds,
                "server_startup_seconds": startup_seconds,
                "server_command": server_command,
                "scenarios": rows,
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "server_log_tail": _tail(server_log, 100),
            }
        )
        raise
    finally:
        _stop_process_tree(process)
        if log_handle is not None:
            log_handle.close()
        write_json(report_path, report)
        if runtime_path == local_cache:
            shutil.rmtree(local_cache, ignore_errors=True)
        time.sleep(3)
    return report


def run(config: Mapping) -> Dict:
    quality_path_value = config["data"].get("quality_comparison_path")
    testset_path_value = config["data"].get("testset_path")
    if not quality_path_value or not testset_path_value:
        raise ValueError(
            "quality_comparison_path and testset_path must point to the completed "
            "publication evaluation. The Colab notebook resolves them automatically."
        )
    quality_path = Path(quality_path_value).expanduser().resolve()
    testset_path = Path(testset_path_value).expanduser().resolve()
    quality = _load_json(quality_path)
    if not quality.get("results"):
        raise ValueError(f"Invalid publication comparison: {quality_path}")
    if not testset_path.is_file():
        raise FileNotFoundError(testset_path)

    evidence = _quality_evidence(quality)
    missing_quality = [model["name"] for model in config["models"] if model["name"] not in evidence]
    if missing_quality:
        raise ValueError(f"Models absent from publication quality evidence: {missing_quality}")

    config_payload = json.loads(json.dumps(config))
    testset_sha = _sha256_file(testset_path)
    signature_payload = {
        "contract": CONTRACT_VERSION,
        "config": _signature_config(config_payload),
        "testset_sha256": testset_sha,
        "quality_config_sha256": quality.get("config_sha256"),
        "quality_checkpoint_hashes": evidence,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_root = Path(config["run"]["output_root"]).expanduser().resolve()
    run_dir = output_root / f"parallelism_{signature[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "benchmark_config.json", config_payload)
    write_json(run_dir / "contract.json", signature_payload)
    print("===== VLLM PARALLELISM BENCHMARK =====", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Models: {len(config['models'])}; scenarios/model: {len(_scenario_plan(config['benchmark']))}", flush=True)
    print(f"Quality evidence: {quality_path}", flush=True)
    print(f"Held-out test set: {testset_path}; sha256={testset_sha}", flush=True)

    natural_dataset = Path("/content/fen_move_natural_prompts.jsonl")
    dataset_manifest = _prepare_natural_dataset(testset_path, natural_dataset)
    write_json(run_dir / "dataset_manifest.json", dataset_manifest)
    help_text = _bench_help()
    environment = _environment()
    write_json(run_dir / "environment.json", environment)

    results = []
    for index, model in enumerate(config["models"], start=1):
        print(f"\n===== MODEL {index}/{len(config['models'])}: {model['name']} =====", flush=True)
        try:
            results.append(
                evaluate_model(
                    model,
                    run_dir,
                    config,
                    natural_dataset,
                    help_text,
                    evidence,
                )
            )
        except Exception as exc:
            print(f"FAILED {model['name']}: {type(exc).__name__}: {exc}", flush=True)
            failed = _load_json(run_dir / "models" / model["name"] / "report.json")
            results.append(failed or {"name": model["name"], "status": "failed", "error": str(exc)})
            if config["run"].get("fail_fast", False):
                raise

        scenario_rows = [
            row
            for result in results
            for row in (result.get("scenarios") or [])
        ]
        write_jsonl(run_dir / "performance_comparison.jsonl", scenario_rows)
        _write_csv(run_dir / "performance_comparison.csv", scenario_rows)
        write_json(
            run_dir / "comparison.json",
            {
                "contract": CONTRACT_VERSION,
                "run_signature_sha256": signature,
                "environment": environment,
                "dataset": dataset_manifest,
                "results": results,
            },
        )

    completed = sum(result.get("status") == "completed" for result in results)
    print(
        f"===== BENCHMARK FINISHED: completed={completed}, "
        f"failed={len(results) - completed} =====",
        flush=True,
    )
    print(f"Results: {run_dir / 'performance_comparison.csv'}", flush=True)
    return {"run_dir": str(run_dir), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(config)


if __name__ == "__main__":
    main()
