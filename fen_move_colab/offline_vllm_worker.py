"""Isolated offline vLLM quality worker used by publication evaluation."""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Mapping


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _quantization_kind(model_path: Path) -> tuple[str, Dict]:
    config = _load_json(model_path / "config.json")
    quant = config.get("quantization_config") or config.get("compression_config") or {}
    text = json.dumps(quant).lower() + " " + str(model_path).lower()
    algorithm = str(quant.get("quant_algo", "")).upper()
    method = str(quant.get("quant_method", "")).lower()
    if algorithm == "NVFP4" or "nvfp4" in text or "modelopt" in method:
        return "nvfp4", quant
    if "fp8" in text or "float8" in text:
        return "fp8", quant
    return "bf16", quant


class NvmlPeakMonitor:
    def __init__(self) -> None:
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
            self._thread.join(timeout=1.0)
        return {
            "gpu_used_baseline_bytes": self.baseline or None,
            "gpu_used_peak_bytes": self.peak or None,
            "gpu_runtime_delta_peak_bytes": (
                max(0, self.peak - self.baseline) if self.peak and self.baseline else None
            ),
        }


def _sampling_params(generation: Mapping):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
        min_p=float(generation.get("min_p", 0.0)),
        repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
        seed=int(generation["seed"]),
        max_tokens=int(generation["max_tokens"]),
        stop=list(generation["stop"]),
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )


def run(spec: Mapping) -> Dict:
    print("[worker] Importing PyTorch, Transformers, and vLLM...", flush=True)
    import torch
    import transformers
    import vllm
    from vllm import LLM

    model_path = Path(spec["model_path"])
    quantization, quantization_metadata = _quantization_kind(model_path)
    print(f"[worker] Model: {model_path}", flush=True)
    print(f"[worker] Quantization detected: {quantization}", flush=True)
    print(
        f"[worker] GPU: {torch.cuda.get_device_name(0)}; "
        f"compute capability: {torch.cuda.get_device_capability()}; "
        f"CUDA: {torch.version.cuda}",
        flush=True,
    )
    server = spec["server"]
    llm_kwargs = {
        "model": str(model_path),
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "generation_config": "vllm",
        "gpu_memory_utilization": float(server["gpu_memory_utilization"]),
        "max_model_len": int(server["max_model_len"]),
        "max_num_seqs": int(server["max_num_seqs"]),
        "enable_prefix_caching": bool(server.get("enable_prefix_caching", False)),
        "enforce_eager": bool(server.get("enforce_eager", False)),
        "language_model_only": True,
        "model_impl": "vllm",
    }
    if quantization == "nvfp4":
        llm_kwargs["quantization"] = "modelopt"
        llm_kwargs["kernel_config"] = {
            "linear_backend": server["nvfp4_linear_backend"],
            "moe_backend": server["nvfp4_moe_backend"],
        }

    monitor = NvmlPeakMonitor()
    monitor.start()
    print(
        "[worker] Loading model with "
        f"max_model_len={llm_kwargs['max_model_len']}, "
        f"max_num_seqs={llm_kwargs['max_num_seqs']}, "
        f"gpu_memory_utilization={llm_kwargs['gpu_memory_utilization']}...",
        flush=True,
    )
    load_started = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - load_started
    print(f"[worker] Model loaded in {load_seconds:.2f}s.", flush=True)
    sampling = _sampling_params(spec["generation"])
    prompts = list(spec["prompts"])

    warmup_count = min(int(spec.get("warmup_positions", 0)), len(prompts))
    warmup_seconds = 0.0
    if warmup_count:
        print(f"[worker] Warming up on {warmup_count} positions...", flush=True)
        warmup_started = time.perf_counter()
        llm.generate(prompts[:warmup_count], sampling, use_tqdm=False)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started
        print(f"[worker] Warmup completed in {warmup_seconds:.2f}s.", flush=True)

    print(
        f"[worker] Generating {len(prompts):,} completions with "
        f"max_tokens={spec['generation']['max_tokens']} and "
        f"stop={spec['generation']['stop']}...",
        flush=True,
    )
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    memory = monitor.stop()
    print(
        f"[worker] Generation completed in {inference_seconds:.2f}s "
        f"({len(prompts) / inference_seconds:.2f} requests/s).",
        flush=True,
    )

    rows = []
    prompt_tokens = 0
    output_tokens = 0
    for request in outputs:
        candidate = request.outputs[0]
        request_prompt_tokens = len(request.prompt_token_ids or [])
        request_output_tokens = len(candidate.token_ids or [])
        prompt_tokens += request_prompt_tokens
        output_tokens += request_output_tokens
        rows.append(
            {
                "completion": candidate.text,
                "finish_reason": candidate.finish_reason,
                "prompt_tokens": request_prompt_tokens,
                "completion_tokens": request_output_tokens,
            }
        )

    finish_reason_counts: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("finish_reason"))
        finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1
    print(
        f"[worker] Output tokens: {output_tokens:,}; finish reasons: "
        f"{finish_reason_counts}; peak VRAM: {memory.get('gpu_used_peak_bytes')}",
        flush=True,
    )

    return {
        "status": "completed",
        "backend": "offline_vllm_llm_generate",
        "quantization_kind": quantization,
        "quantization_metadata": quantization_metadata,
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "inference_seconds": inference_seconds,
        "requests": len(rows),
        "requests_per_second": len(rows) / inference_seconds,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / inference_seconds,
        "results": rows,
        "finish_reason_counts": finish_reason_counts,
        "memory": memory,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "llm_kwargs": llm_kwargs,
        "sampling": dict(spec["generation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run(_load_json(args.spec))
    except Exception as exc:
        print(
            f"[worker] FAILED: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            flush=True,
        )
        report = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    _write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
