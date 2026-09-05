import inspect
from typing import Dict, List, Tuple, Type

import torch


SUPPORTED_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
SUPPORTED_COMPILE_STRATEGIES = {"auto", "full", "regional"}


def performance_config(config: Dict) -> Dict:
    return config.get("performance", {})


def _compile_strategy(config: Dict) -> str:
    strategy = str(performance_config(config).get("torch_compile_strategy", "auto")).lower()
    if strategy not in SUPPORTED_COMPILE_STRATEGIES:
        choices = ", ".join(sorted(SUPPORTED_COMPILE_STRATEGIES))
        raise ValueError(f"performance.torch_compile_strategy must be one of: {choices}.")
    return strategy


def _model_type_names(model) -> set[str]:
    names = {type(model).__name__.lower()}
    model_config = getattr(model, "config", None)
    if model_config is not None:
        model_type = getattr(model_config, "model_type", None)
        if model_type:
            names.add(str(model_type).lower())
        text_config = getattr(model_config, "text_config", None)
        text_model_type = getattr(text_config, "model_type", None)
        if text_model_type:
            names.add(str(text_model_type).lower())
    return names


def resolved_compile_strategy(config: Dict, model=None) -> str:
    if not bool(performance_config(config).get("torch_compile", True)):
        return "disabled"

    requested = _compile_strategy(config)
    if requested != "auto":
        return requested

    if model is not None and any("gemma4" in name or "gemma_4" in name for name in _model_type_names(model)):
        return "regional"
    return "full"


def _repeated_decoder_lists(model) -> List[Tuple[str, torch.nn.ModuleList]]:
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) < 2:
            continue
        child_names = {type(child).__name__.lower() for child in module}
        if len(child_names) == 1 and any("decoderlayer" in child_name for child_name in child_names):
            candidates.append((name, module))
    return candidates


def prepare_model_for_compilation(model, config: Dict):
    """Apply regional compilation without wrapping modules or changing state-dict keys."""
    strategy = resolved_compile_strategy(config, model)
    setattr(model, "_fen_compile_strategy", strategy)
    setattr(model, "_fen_regional_compiled_modules", 0)
    if strategy != "regional":
        return model

    perf_cfg = performance_config(config)
    compile_kwargs = {
        "backend": str(perf_cfg.get("torch_compile_backend", "inductor")),
        "mode": str(perf_cfg.get("torch_compile_mode", "default")),
        "dynamic": bool(perf_cfg.get("torch_compile_dynamic", True)),
        "fullgraph": bool(perf_cfg.get("torch_compile_fullgraph", False)),
    }
    compiled_count = 0
    compiled_groups = []
    for group_name, layers in _repeated_decoder_lists(model):
        for layer in layers:
            if getattr(layer, "_fen_regional_compile_enabled", False):
                continue
            original_forward = layer.forward
            layer.forward = torch.compile(original_forward, **compile_kwargs)
            layer._fen_uncompiled_forward = original_forward
            layer._fen_regional_compile_enabled = True
            compiled_count += 1
        compiled_groups.append(group_name)

    if compiled_count == 0:
        raise RuntimeError(
            "Regional torch.compile could not find a repeated decoder-layer ModuleList. "
            "Set performance.torch_compile_strategy to 'full' for this model."
        )

    setattr(model, "_fen_regional_compiled_modules", compiled_count)
    print(
        f"Enabled regional torch.compile for {compiled_count} decoder layers "
        f"across: {', '.join(compiled_groups)}."
    )
    return model


def gradient_checkpointing_enabled(config: Dict) -> bool:
    perf_cfg = performance_config(config)
    if "gradient_checkpointing" in perf_cfg:
        return bool(perf_cfg["gradient_checkpointing"])
    return bool(config.get("fp8", {}).get("gradient_checkpointing", False))


def configure_torch_runtime(config: Dict) -> None:
    perf_cfg = performance_config(config)
    matmul_precision = str(perf_cfg.get("float32_matmul_precision", "high"))
    torch.set_float32_matmul_precision(matmul_precision)

    if not torch.cuda.is_available():
        return

    allow_tf32 = bool(perf_cfg.get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32


def training_argument_performance_kwargs(config: Dict, argument_cls: Type, model=None) -> Dict:
    perf_cfg = performance_config(config)
    compile_enabled = bool(perf_cfg.get("torch_compile", True))
    compile_mode = str(perf_cfg.get("torch_compile_mode", "default"))
    if compile_mode not in SUPPORTED_COMPILE_MODES:
        choices = ", ".join(sorted(SUPPORTED_COMPILE_MODES))
        raise ValueError(f"performance.torch_compile_mode must be one of: {choices}.")

    strategy = resolved_compile_strategy(config, model)
    full_compile_enabled = compile_enabled and strategy == "full"
    num_workers = max(0, int(perf_cfg.get("dataloader_num_workers", 0)))
    desired = {
        "torch_compile": full_compile_enabled,
        "optim": str(perf_cfg.get("optimizer", "adamw_torch_fused")),
        "tf32": bool(perf_cfg.get("tf32", True)) if torch.cuda.is_available() else False,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": bool(perf_cfg.get("dataloader_pin_memory", True)),
        "dataloader_persistent_workers": bool(
            num_workers > 0 and perf_cfg.get("dataloader_persistent_workers", True)
        ),
    }
    # Transformers enables compilation whenever backend or mode is non-null,
    # even if torch_compile=False, so only pass these for full compilation.
    if full_compile_enabled:
        desired["torch_compile_backend"] = str(perf_cfg.get("torch_compile_backend", "inductor"))
        desired["torch_compile_mode"] = compile_mode
    if gradient_checkpointing_enabled(config):
        desired["gradient_checkpointing_kwargs"] = {
            "use_reentrant": bool(perf_cfg.get("gradient_checkpointing_use_reentrant", False))
        }
    if num_workers > 0:
        desired["dataloader_prefetch_factor"] = max(
            1, int(perf_cfg.get("dataloader_prefetch_factor", 2))
        )

    init_params = inspect.signature(argument_cls.__init__).parameters
    supported = {key: value for key, value in desired.items() if key in init_params}
    ignored = sorted(set(desired) - set(supported))
    if ignored:
        print(
            f"Ignoring performance arguments unsupported by {argument_cls.__name__}: "
            + ", ".join(ignored)
        )
    return supported


def performance_metrics(config: Dict, model=None) -> Dict:
    perf_cfg = performance_config(config)
    strategy = getattr(model, "_fen_compile_strategy", None) or resolved_compile_strategy(config, model)
    return {
        "torch_compile_enabled": bool(perf_cfg.get("torch_compile", True)),
        "torch_compile_strategy_requested": _compile_strategy(config),
        "torch_compile_strategy_resolved": strategy,
        "torch_compile_full_model_enabled": strategy == "full",
        "torch_compile_regional_modules": int(getattr(model, "_fen_regional_compiled_modules", 0)),
        "torch_compile_backend": str(perf_cfg.get("torch_compile_backend", "inductor")),
        "torch_compile_mode": str(perf_cfg.get("torch_compile_mode", "default")),
        "torch_compile_dynamic": bool(perf_cfg.get("torch_compile_dynamic", True)),
        "optimizer": str(perf_cfg.get("optimizer", "adamw_torch_fused")),
        "tf32_enabled": bool(perf_cfg.get("tf32", True)),
        "gradient_checkpointing_enabled": gradient_checkpointing_enabled(config),
        "dataloader_num_workers": max(0, int(perf_cfg.get("dataloader_num_workers", 0))),
    }
