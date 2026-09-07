"""Re-evaluate LoRA checkpoints while loading each BF16 base model only once."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import load_config, save_json, set_seed
from .evaluate_sft_checkpoints import discover_adapter_checkpoints
from .fp8_lora import get_model_load_kwargs, resolve_adapter_source
from .publication_eval import END_OF_TURN
from .train_sft_chunk import AllLegalMovesValidationCallback, _load_sft_validation_examples


CONTRACT_VERSION = "checkpoint_selection_eot_v2_shared_base"
SCORE_KEY = "fen_sft_val_all_moves_f1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_training_config(run_root: Path) -> Path:
    candidates = list((run_root / "metrics" / "sft").glob("step_*_config.json"))
    if not candidates:
        raise FileNotFoundError(f"No step_*_config.json found below {run_root}")

    def key(path: Path):
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=key)


def _selection_identity(examples: Sequence[Mapping]) -> str:
    payload = "\n".join(
        f"{row.get('puzzle_id', '')}\t{row.get('fen', '')}" for row in examples
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_contract(
    config_path: Path,
    checkpoints: Sequence[Path],
    examples: Sequence[Mapping],
    samples: int,
    batch_size: int,
    max_new_tokens: int,
) -> Dict:
    return {
        "contract": CONTRACT_VERSION,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checkpoints": [str(path) for path in checkpoints],
        "selection_identity_sha256": _selection_identity(examples),
        "samples": samples,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "parser": "strict_whitespace_uci_before_first_end_of_turn",
        "end_of_turn_marker": END_OF_TURN,
    }


def _contract_hash(contract: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prepare_run(
    drive_bundle: Path,
    run_name: str,
    samples: int,
    batch_size: int,
    max_new_tokens: int,
) -> Dict:
    run_root = drive_bundle / "runs" / run_name
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run does not exist: {run_root}")
    config_path = _latest_training_config(run_root)
    config = load_config(str(config_path))
    config = copy.deepcopy(config)
    config.setdefault("performance", {})["torch_compile"] = False
    config["performance"]["gradient_checkpointing"] = False
    config.setdefault("precision", {})["mode"] = "bf16"
    if str(config["sft"].get("output_mode")) != "all_legal_moves":
        raise ValueError(f"{run_name} is not an all_legal_moves run.")

    examples = _load_sft_validation_examples(config, samples)
    checkpoints = discover_adapter_checkpoints(
        str(run_root / "checkpoints" / "sft")
    )
    base_model, _ = resolve_adapter_source(
        str(checkpoints[0]), config["sft"]["model_id"]
    )
    contract = _checkpoint_contract(
        config_path,
        checkpoints,
        examples,
        samples,
        batch_size,
        max_new_tokens,
    )
    contract_sha = _contract_hash(contract)
    metrics_dir = run_root / "metrics" / "sft"
    old_report_path = metrics_dir / "checkpoint_evaluation.json"
    backup_path = metrics_dir / "checkpoint_evaluation_before_eot_fix.json"
    corrected_path = metrics_dir / "checkpoint_evaluation_eot_v2.json"
    if old_report_path.is_file() and not backup_path.exists():
        shutil.copy2(old_report_path, backup_path)
        print(f"[{run_name}] Backed up old selection report: {backup_path}", flush=True)

    previous = {}
    if corrected_path.is_file():
        previous = json.loads(corrected_path.read_text(encoding="utf-8"))
        if previous.get("selection_contract_sha256") != contract_sha:
            previous = {}
    prior_results = {
        str(row.get("checkpoint")): row for row in previous.get("results", [])
    }
    payload = {
        **contract,
        "selection_contract_sha256": contract_sha,
        "run_name": run_name,
        "run_root": str(run_root),
        "base_model": base_model,
        "precision_mode": "bf16",
        "status": "running",
        "started_utc": previous.get("started_utc") or _utc_now(),
        "selection_metric": SCORE_KEY,
        "results": [
            prior_results[str(path)]
            for path in checkpoints
            if str(path) in prior_results
        ],
    }
    save_json(str(corrected_path), payload)
    return {
        "run_name": run_name,
        "run_root": run_root,
        "config": config,
        "examples": examples,
        "checkpoints": checkpoints,
        "base_model": base_model,
        "contract_sha": contract_sha,
        "payload": payload,
        "corrected_path": corrected_path,
        "standard_path": old_report_path,
        "backup_path": backup_path,
    }


def _finalize_run(context: Dict) -> Dict:
    payload = context["payload"]
    ordered = {row["checkpoint"]: row for row in payload["results"]}
    payload["results"] = [
        ordered[str(path)] for path in context["checkpoints"] if str(path) in ordered
    ]
    if len(payload["results"]) != len(context["checkpoints"]):
        raise RuntimeError(f"{context['run_name']} did not evaluate every checkpoint.")
    payload["best"] = max(payload["results"], key=lambda row: row[SCORE_KEY])
    payload["status"] = "completed"
    payload["finished_utc"] = _utc_now()
    save_json(str(context["corrected_path"]), payload)
    save_json(str(context["standard_path"]), payload)

    old_best = None
    if context["backup_path"].is_file():
        old = json.loads(context["backup_path"].read_text(encoding="utf-8"))
        old_best = old.get("best") or {}
    new_best = payload["best"]
    changed = (old_best or {}).get("checkpoint") != new_best.get("checkpoint")
    return {
        "run_name": context["run_name"],
        "base_model": context["base_model"],
        "old_best_checkpoint": (old_best or {}).get("checkpoint"),
        "old_best_f1": (old_best or {}).get(SCORE_KEY),
        "new_best_checkpoint": new_best.get("checkpoint"),
        "new_best_f1": new_best.get(SCORE_KEY),
        "best_checkpoint_changed": changed,
        "checkpoints_evaluated": len(payload["results"]),
        "corrected_report": str(context["corrected_path"]),
    }


def compare_checkpoint_selections(
    drive_bundle_dir: str,
    run_names: Sequence[str],
    samples: int = 128,
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> Dict:
    """Evaluate all adapter candidates with one base-model load per model family."""
    drive_bundle = Path(drive_bundle_dir).expanduser().resolve()
    contexts = [
        _prepare_run(
            drive_bundle,
            run_name,
            samples,
            batch_size,
            max_new_tokens,
        )
        for run_name in run_names
    ]
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for context in contexts:
        groups[str(context["base_model"])].append(context)

    print(
        f"Prepared {len(contexts)} runs in {len(groups)} base-model groups. "
        "Merged and quantized exports will not be loaded.",
        flush=True,
    )
    for base_index, (base_model_id, group) in enumerate(groups.items(), start=1):
        pending = [
            (context, checkpoint)
            for context in group
            for checkpoint in context["checkpoints"]
            if str(checkpoint)
            not in {str(row["checkpoint"]) for row in context["payload"]["results"]}
        ]
        if not pending:
            print(f"[{base_model_id}] All checkpoint scores are already cached.", flush=True)
            continue

        print(
            f"\n===== BASE {base_index}/{len(groups)}: {base_model_id} =====\n"
            f"Loading one BF16 base for {len(pending)} pending adapter checkpoints...",
            flush=True,
        )
        first_config = group[0]["config"]
        set_seed(int(first_config["run"].get("seed", 42)))
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, **get_model_load_kwargs(first_config)
        )
        if hasattr(base, "config"):
            base.config.use_cache = True
        base.eval()
        model = None

        for adapter_index, (context, checkpoint) in enumerate(pending, start=1):
            adapter_name = f"selection_{adapter_index}"
            print(
                f"[{base_model_id}] {adapter_index}/{len(pending)} "
                f"{context['run_name']} :: {checkpoint.name}",
                flush=True,
            )
            if model is None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    base,
                    str(checkpoint),
                    adapter_name=adapter_name,
                    is_trainable=False,
                )
            else:
                model.load_adapter(
                    str(checkpoint),
                    adapter_name=adapter_name,
                    is_trainable=False,
                )
            model.set_adapter(adapter_name)
            model.eval()
            evaluator = AllLegalMovesValidationCallback(
                tokenizer=tokenizer,
                examples=context["examples"],
                eval_steps=1,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            metrics = evaluator._evaluate(model)
            result = {
                "checkpoint": str(checkpoint),
                "checkpoint_name": checkpoint.name,
                **metrics,
            }
            context["payload"]["results"].append(result)
            save_json(str(context["corrected_path"]), context["payload"])
            print(
                f"  F1={metrics[SCORE_KEY]:.6f}; "
                f"exact={metrics['fen_sft_val_all_moves_set_exact_rate']:.6f}",
                flush=True,
            )
            del evaluator
            model.delete_adapter(adapter_name)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model, base, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparisons = [_finalize_run(context) for context in contexts]
    summary = {
        "contract": CONTRACT_VERSION,
        "completed_utc": _utc_now(),
        "base_models_loaded": len(groups),
        "runs_compared": len(comparisons),
        "changed_runs": [
            row["run_name"] for row in comparisons if row["best_checkpoint_changed"]
        ],
        "results": comparisons,
    }
    summary_path = drive_bundle / "runs" / "checkpoint_selection_eot_v2_summary.json"
    save_json(str(summary_path), summary)
    print("\n===== CHECKPOINT SELECTION COMPARISON =====", flush=True)
    for row in comparisons:
        print(
            f"{row['run_name']}: changed={row['best_checkpoint_changed']} "
            f"old_f1={row['old_best_f1']} new_f1={row['new_best_f1']}",
            flush=True,
        )
        print(f"  old: {row['old_best_checkpoint']}", flush=True)
        print(f"  new: {row['new_best_checkpoint']}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return summary
