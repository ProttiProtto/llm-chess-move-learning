import json
import os
import warnings
from importlib import metadata
from typing import Dict, List, Optional, Tuple

import torch

from .performance import gradient_checkpointing_enabled


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


SUPPORTED_PRECISION_MODES = {"bf16", "fp8", "nvfp4_qat"}


def precision_mode(config: Dict) -> str:
    """Resolve the explicit precision mode while accepting legacy FP8 configs."""
    configured = config.get("precision", {}).get("mode")
    if configured is None:
        configured = "fp8" if config.get("fp8", {}).get("enabled", False) else "bf16"
    mode = str(configured).strip().lower()
    if mode not in SUPPORTED_PRECISION_MODES:
        choices = ", ".join(sorted(SUPPORTED_PRECISION_MODES))
        raise ValueError(f"precision.mode must be one of: {choices}. Found {configured!r}.")
    return mode


def fp8_mode_enabled(config: Dict) -> bool:
    return precision_mode(config) == "fp8"


def nvfp4_qat_mode_enabled(config: Dict) -> bool:
    return precision_mode(config) == "nvfp4_qat"


def low_precision_alignment_enabled(config: Dict) -> bool:
    return precision_mode(config) in {"fp8", "nvfp4_qat"}


def precision_runtime_metrics(model, config: Dict) -> Dict:
    """Describe what is actually quantized so reports do not overstate the mode."""
    mode = precision_mode(config)
    metrics = {
        "fp8_training_compute_enabled": mode == "fp8",
        "fp8_training_linear_modules": 0,
        "fp8_training_master_weight_dtype": None,
        "fp8_training_persistent_quantized_base": False,
    }
    if mode != "fp8":
        return metrics

    try:
        from torchao.float8.float8_linear import Float8Linear
    except Exception:
        return metrics

    fp8_linears = [module for module in model.modules() if isinstance(module, Float8Linear)]
    metrics["fp8_training_linear_modules"] = len(fp8_linears)
    if fp8_linears:
        metrics["fp8_training_master_weight_dtype"] = str(fp8_linears[0].weight.dtype)
    return metrics


def get_model_load_kwargs(config: Dict) -> Dict:
    use_cuda = torch.cuda.is_available()
    return {
        "torch_dtype": torch.bfloat16 if use_cuda else torch.float32,
        "device_map": "auto" if use_cuda else None,
    }


def _get_fp8_config(config: Dict) -> Dict:
    return config.get("fp8", {})


def _get_nvfp4_config(config: Dict) -> Dict:
    return config.get("nvfp4", {})


def _get_lora_config(config: Dict) -> Dict:
    # LoRA originally lived under fp8; preserve those configs while allowing a
    # precision-independent lora section in new notebooks.
    merged = dict(_get_fp8_config(config))
    merged.update(config.get("lora", {}))
    return merged


def _version_tuple(raw: str) -> Tuple[int, ...]:
    parts = []
    for token in raw.replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts or [0])


def ensure_fp8_runtime(config: Dict) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 mode requires a CUDA-enabled runtime.")

    min_major = int(_get_fp8_config(config).get("min_compute_capability_major", 8))
    min_minor = int(_get_fp8_config(config).get("min_compute_capability_minor", 9))
    capability = torch.cuda.get_device_capability()
    if capability < (min_major, min_minor):
        raise RuntimeError(
            "FP8 mode requires a GPU with compute capability "
            f">= {min_major}.{min_minor}. Found {capability[0]}.{capability[1]}."
        )

    try:
        import torchao.float8  # noqa: F401
    except Exception as exc:
        raise RuntimeError("FP8 mode requires torchao. Run fen_move_colab/install_colab.sh first.") from exc

    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        torchao_version = "0"

    if _version_tuple(torchao_version) < (0, 15, 0):
        raise RuntimeError(f"FP8 mode requires torchao>=0.15.0. Found torchao=={torchao_version}.")


def ensure_nvfp4_qat_runtime(config: Dict) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 QAT requires a CUDA-enabled Blackwell runtime.")

    min_major = int(_get_nvfp4_config(config).get("min_compute_capability_major", 10))
    min_minor = int(_get_nvfp4_config(config).get("min_compute_capability_minor", 0))
    capability = torch.cuda.get_device_capability()
    if capability < (min_major, min_minor):
        raise RuntimeError(
            "NVFP4 QAT requires Blackwell compute capability "
            f">= {min_major}.{min_minor}. Found {capability[0]}.{capability[1]}. "
            "Use precision.mode='bf16' or 'fp8' on this GPU."
        )

    cuda_version = torch.version.cuda or "0"
    if _version_tuple(cuda_version) < (12, 8):
        raise RuntimeError(f"NVFP4 QAT requires CUDA>=12.8. Found CUDA {cuda_version}.")

    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("NVFP4 QAT requires torchao. Run fen_move_colab/install_colab.sh first.") from exc

    if _version_tuple(torchao_version) < (0, 17, 0):
        raise RuntimeError(f"NVFP4 QAT requires torchao>=0.17.0. Found torchao=={torchao_version}.")


def resolve_adapter_source(model_source: str, fallback_base_model: str) -> Tuple[str, Optional[str]]:
    if model_source and os.path.isdir(model_source):
        adapter_config_path = os.path.join(model_source, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path") or fallback_base_model
            return str(base_model_name), model_source
    return model_source, None


def _lora_target_modules(config: Dict) -> List[str]:
    values = _get_lora_config(config).get("target_modules")
    if values is None:
        values = _get_lora_config(config).get("lora_target_modules")
    values = values or DEFAULT_LORA_TARGET_MODULES
    return [str(v) for v in values]


def _resolve_lora_target_modules(model, config: Dict) -> List[str]:
    requested = _lora_target_modules(config)
    requested_set = set(requested)
    resolved = []
    wrapped = []

    for fqn, module in model.named_modules():
        if not fqn or fqn.rsplit(".", 1)[-1] not in requested_set:
            continue
        if isinstance(module, torch.nn.Linear):
            resolved.append(fqn)
            continue

        # Gemma 4 exposes q_proj/k_proj/etc. as Gemma4ClippableLinear
        # wrappers. PEFT cannot replace the wrapper, but its child is a
        # standard Linear and remains on the exact same forward path.
        inner_linear = getattr(module, "linear", None)
        if isinstance(inner_linear, torch.nn.Linear):
            inner_fqn = f"{fqn}.linear"
            resolved.append(inner_fqn)
            wrapped.append(inner_fqn)

    if not wrapped:
        return requested
    if not resolved:
        raise ValueError(f"Could not resolve supported LoRA modules from requested targets: {requested}")

    unique_targets = sorted(set(resolved))
    print(
        "Resolved wrapped model projections to inner Linear modules for LoRA: "
        f"{len(wrapped)} wrapped targets, {len(unique_targets)} total targets."
    )
    return unique_targets


def _make_module_filter(model):
    allowed_module_ids = set()
    for fqn, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "lm_head" in fqn or ".lora_" in fqn:
            continue
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            continue
        allowed_module_ids.add(id(module))

    def module_filter_fn(module: torch.nn.Module, fqn: str = "") -> bool:
        return id(module) in allowed_module_ids

    return module_filter_fn


def _enable_training_memory_savers(model, config: Dict) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = False
    if gradient_checkpointing_enabled(config):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()


def _prepare_lora_model(model, config: Dict, adapter_path: Optional[str]):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    lora_cfg = _get_lora_config(config)
    if adapter_path:
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    if not bool(lora_cfg.get("enabled", lora_cfg.get("use_lora", True))):
        return model

    return get_peft_model(
        model,
        LoraConfig(
            r=int(lora_cfg.get("r", lora_cfg.get("lora_r", 16))),
            lora_alpha=int(lora_cfg.get("alpha", lora_cfg.get("lora_alpha", 32))),
            lora_dropout=float(lora_cfg.get("dropout", lora_cfg.get("lora_dropout", 0.05))),
            bias=str(lora_cfg.get("bias", lora_cfg.get("lora_bias", "none"))),
            task_type=TaskType.CAUSAL_LM,
            target_modules=_resolve_lora_target_modules(model, config),
            inference_mode=False,
        ),
    )


def _prepare_nvfp4_qat(model, config: Dict):
    ensure_nvfp4_qat_runtime(config)
    try:
        from torchao.prototype.mx_formats import NVFP4InferenceConfig
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
    except Exception as exc:
        raise RuntimeError(
            "This Torch/TorchAO build does not expose the prototype NVFP4 QAT API. "
            "Use a Blackwell runtime with CUDA>=12.8 and compatible current PyTorch/TorchAO packages, "
            "or set precision.mode='bf16'."
        ) from exc

    try:
        base_config = NVFP4InferenceConfig()
        quantize_(
            model,
            QATConfig(base_config, step="prepare"),
            filter_fn=_make_module_filter(model),
        )
    except Exception as exc:
        try:
            torchao_version = metadata.version("torchao")
        except metadata.PackageNotFoundError:
            torchao_version = "not installed"
        raise RuntimeError(
            "TorchAO NVFP4 QAT preparation failed. The API is still prototype and version-sensitive. "
            f"Runtime: torch={torch.__version__}, torchao={torchao_version}, CUDA={torch.version.cuda}. "
            "Use precision.mode='bf16' to continue without QAT."
        ) from exc

    warnings.warn(
        "NVFP4 QAT uses fake quantization during LoRA training. It targets NVFP4 deployment numerics, "
        "but does not guarantee lower training VRAM or faster training. Saved checkpoints remain PEFT "
        "adapters unless a separate merge-and-convert export is performed.",
        RuntimeWarning,
        stacklevel=2,
    )
    return model


def maybe_prepare_fp8_lora_model(model, config: Dict, adapter_path: Optional[str] = None):
    _enable_training_memory_savers(model, config)
    model = _prepare_lora_model(model, config, adapter_path)
    mode = precision_mode(config)

    if mode == "bf16":
        return model
    if mode == "nvfp4_qat":
        return _prepare_nvfp4_qat(model, config)

    ensure_fp8_runtime(config)
    from torchao.float8 import convert_to_float8_training

    try:
        from torchao.float8 import Float8LinearConfig
    except Exception:
        Float8LinearConfig = None

    _enable_training_memory_savers(model, config)
    if hasattr(model, "config"):
        model.config.use_cache = False

    convert_kwargs = {"module_filter_fn": _make_module_filter(model)}
    recipe = str(_get_fp8_config(config).get("recipe", "rowwise"))
    if Float8LinearConfig is not None and hasattr(Float8LinearConfig, "from_recipe_name"):
        convert_kwargs["config"] = Float8LinearConfig.from_recipe_name(recipe)

    convert_to_float8_training(model, **convert_kwargs)
    return model
