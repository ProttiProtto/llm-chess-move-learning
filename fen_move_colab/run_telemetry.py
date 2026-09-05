import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback

from .common import ensure_dir


TRACKED_PACKAGES = ("transformers", "peft", "trl", "torchao", "datasets", "accelerate")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def package_versions() -> Dict[str, Optional[str]]:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def runtime_environment() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "packages": package_versions(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        gpus = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
        result["gpus"] = gpus
    return result


def reset_cuda_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def cuda_memory_metrics(prefix: str = "gpu") -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    per_device = []
    for index in range(torch.cuda.device_count()):
        per_device.append(
            {
                "index": index,
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
    return {
        f"{prefix}_memory_per_device": per_device,
        f"{prefix}_peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in per_device),
        f"{prefix}_peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in per_device),
    }


def process_peak_rss_bytes() -> Optional[int]:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:
        return None


def model_parameter_metrics(model) -> Dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "model_parameters_total": int(total),
        "model_parameters_trainable": int(trainable),
        "model_parameters_trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }


def directory_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                total += os.path.getsize(file_path)
    return total


def prefixed_trainer_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {f"trainer_{key}": _json_safe(value) for key, value in (metrics or {}).items()}


class TrainingTelemetryCallback(TrainerCallback):
    def __init__(self, history_path: str, stage_start_perf: float):
        self.history_path = history_path
        self.stage_start_perf = stage_start_perf
        ensure_dir(os.path.dirname(history_path))
        with open(history_path, "w", encoding="utf-8"):
            pass

    def record(
        self,
        event: str,
        metrics: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        epoch: Optional[float] = None,
    ) -> None:
        payload = {
            "event": event,
            "timestamp_utc": utc_now_iso(),
            "stage_elapsed_seconds": time.perf_counter() - self.stage_start_perf,
            "global_step": step,
            "epoch": epoch,
            **(metrics or {}),
            **cuda_memory_metrics("current_gpu"),
        }
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(payload), separators=(",", ":")) + "\n")
            f.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        self.record("train_begin", step=int(state.global_step), epoch=state.epoch)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.record("trainer_log", logs or {}, int(state.global_step), state.epoch)

    def on_train_end(self, args, state, control, **kwargs):
        self.record("train_end", step=int(state.global_step), epoch=state.epoch)
