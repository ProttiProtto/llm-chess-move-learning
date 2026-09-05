"""Merge a PEFT adapter and export a reloadable TorchAO FP8 inference model."""

import argparse
import gc
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from .common import load_config, load_json, resolve_drive_path, save_json


def _latest_checkpoint(config: Dict, stage: str) -> str:
    state_path = resolve_drive_path(config, "state", f"{stage}_state.json")
    state = load_json(state_path, default={})
    checkpoint = state.get("latest_checkpoint")
    if not checkpoint:
        raise FileNotFoundError(
            f"No {stage.upper()} checkpoint is recorded in {state_path}. "
            "Train a chunk first or pass --checkpoint explicitly."
        )
    return str(checkpoint)


def _resolve_adapter_checkpoint(config: Dict, checkpoint: Optional[str], stage: str) -> str:
    resolved = checkpoint or _latest_checkpoint(config, stage)
    adapter_config = os.path.join(resolved, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        raise ValueError(
            f"FP8 export expects a PEFT adapter checkpoint, but {adapter_config} was not found. "
            "Enable LoRA for training or point --checkpoint at an adapter directory."
        )
    return resolved


def _base_model_from_adapter(adapter_checkpoint: str, fallback: str) -> str:
    with open(os.path.join(adapter_checkpoint, "adapter_config.json"), "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    return str(adapter_config.get("base_model_name_or_path") or fallback)


def _training_metadata(adapter_checkpoint: str, stage: str, config: Dict) -> Dict:
    """Resolve authoritative training precision from the saved run metrics."""
    checkpoint_path = Path(adapter_checkpoint).resolve()
    sample_checkpoint = next(
        (
            candidate
            for candidate in (checkpoint_path, *checkpoint_path.parents)
            if candidate.name.startswith("step_") and candidate.name[5:].isdigit()
        ),
        None,
    )
    metrics_path = None
    metrics = {}
    if sample_checkpoint is not None and len(sample_checkpoint.parents) >= 3:
        run_root = sample_checkpoint.parents[2]
        candidate_path = run_root / "metrics" / stage / f"{sample_checkpoint.name}.json"
        if candidate_path.is_file():
            metrics_path = candidate_path
            metrics = load_json(str(candidate_path), default={})

    configured_precision = str(config.get("precision", {}).get("mode", "bf16")).lower()
    training_precision = str(metrics.get("precision_mode") or configured_precision).lower()
    precision_source = "saved_run_metrics" if metrics_path is not None else "current_export_config_fallback"
    return {
        "training_precision_mode": training_precision,
        "training_precision_source": precision_source,
        "training_metrics_path": str(metrics_path) if metrics_path is not None else None,
        "training_used_fp8_compute": metrics.get("fp8_training_compute_enabled"),
        "training_fp8_linear_modules": metrics.get("fp8_training_linear_modules"),
        "training_fp8_master_weight_dtype": metrics.get("fp8_training_master_weight_dtype"),
    }


def _granularity(name: str):
    from torchao.quantization import PerRow, PerTensor

    normalized = str(name).strip().lower()
    if normalized == "tensorwise":
        return PerTensor()
    if normalized == "rowwise":
        return PerRow()
    raise ValueError("fp8.export.granularity must be 'tensorwise' or 'rowwise'.")


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _fp8_weight_metrics(model) -> Dict:
    from torchao.quantization import Float8Tensor

    quantized_tensors = 0
    quantized_values = 0
    quantized_data_bytes = 0
    scale_bytes = 0
    seen_weights = set()
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Float8Tensor) or id(weight) in seen_weights:
            continue
        seen_weights.add(id(weight))
        quantized_tensors += 1
        quantized_values += int(weight.qdata.numel())
        quantized_data_bytes += int(weight.qdata.numel() * weight.qdata.element_size())
        scale_bytes += int(weight.scale.numel() * weight.scale.element_size())
    return {
        "fp8_weight_tensors": quantized_tensors,
        "fp8_weight_values": quantized_values,
        "fp8_quantized_data_bytes": quantized_data_bytes,
        "fp8_scale_bytes": scale_bytes,
    }


def _free_cuda_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _merge_adapter_to_bf16(
    base_model: str,
    adapter_checkpoint: str,
    output_dir: Path,
    max_shard_size: str,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_checkpoint, is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(output_dir)
    return merged, tokenizer


def _quantize_merged_model(
    merged_model_dir: Path,
    output_dir: Path,
    granularity: str,
    modules_to_not_convert,
    max_shard_size: str,
):
    import torch
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    ao_config = Float8DynamicActivationFloat8WeightConfig(
        granularity=_granularity(granularity),
        set_inductor_config=True,
    )
    quantization_config = TorchAoConfig(
        ao_config,
        modules_to_not_convert=list(modules_to_not_convert),
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


def _verify_reload(output_dir: Path, prompt: str, max_new_tokens: int) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    del model
    _free_cuda_memory()
    return text


def export_fp8_inference(
    config: Dict,
    checkpoint: Optional[str] = None,
    stage: str = "sft",
    output_dir: Optional[str] = None,
) -> Tuple[str, Dict]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FP8 activation-and-weight export requires an FP8-capable CUDA GPU.")

    from .fp8_lora import ensure_fp8_runtime

    ensure_fp8_runtime(config)
    export_cfg = config.get("fp8", {}).get("export", {})
    adapter_checkpoint = _resolve_adapter_checkpoint(config, checkpoint, stage)
    fallback_model = str(config.get("sft", {}).get("model_id", ""))
    base_model = _base_model_from_adapter(adapter_checkpoint, fallback_model)
    checkpoint_name = Path(adapter_checkpoint).name
    export_root = Path(
        output_dir
        or resolve_drive_path(config, "exports", "fp8", f"{stage}_{checkpoint_name}")
    )
    fp8_dir = export_root / "fp8_dynamic_activation_weight"
    merged_dir = export_root / "merged_bf16"
    keep_merged = bool(export_cfg.get("save_merged_bf16", True))
    verify_reload = bool(export_cfg.get("verify_reload", True))
    max_shard_size = str(export_cfg.get("max_shard_size", "5GB"))
    granularity = str(export_cfg.get("granularity", "tensorwise"))
    modules_to_not_convert = export_cfg.get("modules_to_not_convert") or ["lm_head"]
    overwrite = bool(export_cfg.get("overwrite", False))

    if export_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Export already exists: {export_root}. Set fp8.export.overwrite=true or choose another output."
            )
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    temporary_dir = None
    if keep_merged:
        merge_target = merged_dir
    else:
        temporary_dir = tempfile.TemporaryDirectory(prefix="fen_move_merged_bf16_")
        merge_target = Path(temporary_dir.name)

    merged, _ = _merge_adapter_to_bf16(base_model, adapter_checkpoint, merge_target, max_shard_size)
    del merged
    _free_cuda_memory()

    fp8_model, _ = _quantize_merged_model(
        merge_target,
        fp8_dir,
        granularity,
        modules_to_not_convert,
        max_shard_size,
    )
    quantization_metrics = _fp8_weight_metrics(fp8_model)
    if quantization_metrics["fp8_weight_tensors"] <= 0:
        raise RuntimeError("FP8 export completed without quantizing any weights.")
    del fp8_model
    _free_cuda_memory()

    verification_text = None
    if verify_reload:
        verification_text = _verify_reload(
            fp8_dir,
            str(export_cfg.get("verification_prompt", "FEN: 8/8/8/8/8/8/4K3/7k w - - 0 1\nLegal UCI move:")),
            int(export_cfg.get("verification_max_new_tokens", 8)),
        )

    if temporary_dir is not None:
        temporary_dir.cleanup()

    training_metadata = _training_metadata(adapter_checkpoint, stage, config)
    manifest = {
        "format": "torchao_float8_dynamic_activation_float8_weight",
        # Keep training and export precision separate. A BF16-trained adapter
        # that is quantized here must never be labeled as FP8-trained.
        "training_precision": training_metadata["training_precision_mode"],
        **training_metadata,
        "export_precision": "fp8_dynamic_activation_float8_weight",
        "post_training_quantization": True,
        "adapter_checkpoint": adapter_checkpoint,
        "base_model": base_model,
        "stage": stage,
        "granularity": granularity,
        "modules_to_not_convert": list(modules_to_not_convert),
        "merged_bf16_saved": keep_merged,
        "merged_bf16_path": str(merged_dir) if keep_merged else None,
        "fp8_model_path": str(fp8_dir),
        "fp8_checkpoint_size_bytes": _directory_size(fp8_dir),
        "merged_bf16_checkpoint_size_bytes": _directory_size(merged_dir) if keep_merged else None,
        "reload_verified": verify_reload,
        "reload_verification_scope": "serialization_reload_and_generation_smoke_test_not_accuracy",
        "verification_output": verification_text,
        "export_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        **quantization_metrics,
    }
    save_json(str(export_root / "export_manifest.json"), manifest)
    print(f"Saved merged FP8 inference model: {fp8_dir}")
    if keep_merged:
        print(f"Saved canonical merged BF16 model: {merged_dir}")
    print(json.dumps(manifest, indent=2))
    return str(fp8_dir), manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stage", choices=["sft", "grpo"], default="sft")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    export_fp8_inference(
        load_config(args.config),
        checkpoint=args.checkpoint,
        stage=args.stage,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
