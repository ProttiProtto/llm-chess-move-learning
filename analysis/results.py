from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd


MODEL_LABELS = {
    "270m": "Gemma 3 270M",
    "e2b": "Gemma 4 E2B",
    "e4b": "Gemma 4 E4B",
}
QUANTIZATIONS = ("bf16", "fp8", "nvfp4")
CONCURRENCIES = (1, 8, 32, 128, 256)


def parse_variant_name(name: str) -> tuple[str, str, str]:
    normalized = name.lower().replace("_best", "").replace("_", "-")
    match = re.fullmatch(r"(270m|e2b|e4b)-(base|r16|r32)-(bf16|fp8|nvfp4)", normalized)
    if not match:
        raise ValueError(f"Unrecognized model variant name: {name}")
    return match.group(1), match.group(2), match.group(3)


def load_quality_comparison(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for result in payload.get("results", []):
        size, variant, quantization = parse_variant_name(result["name"])
        quality = result.get("quality", {}).get("overall", {})
        if not quality:
            raise ValueError(f"Missing quality.overall for {result['name']}")
        checkpoint = result.get("checkpoint", {})
        rows.append(
            {
                "model": result["name"],
                "size": size,
                "model_label": MODEL_LABELS[size],
                "variant": variant,
                "lora_rank": int(variant[1:]) if variant.startswith("r") else 0,
                "quantization": quantization,
                "samples": int(quality["samples"]),
                "macro_precision": float(quality["macro_precision"]),
                "macro_recall": float(quality["macro_recall"]),
                "macro_f1": float(quality["macro_f1"]),
                "set_exact_rate": float(quality["set_exact_rate"]),
                "illegal_move_rate": float(quality["illegal_move_rate"]),
                "evaluation_contract_sha256": result.get("evaluation_contract_sha256"),
                "checkpoint_content_sha256": checkpoint.get("checkpoint_content_sha256"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No quality rows found in {path}")
    return frame.sort_values(["size", "variant", "quantization"]).reset_index(drop=True)


def load_performance_comparison(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    required = {
        "model",
        "mode",
        "concurrency",
        "repetition",
        "output_throughput",
        "request_throughput",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p50_e2el_ms",
        "p95_e2el_ms",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Performance CSV is missing columns: {missing}")

    parsed = raw["model"].map(parse_variant_name)
    raw = raw.copy()
    raw[["size", "variant", "quantization"]] = pd.DataFrame(parsed.tolist(), index=raw.index)
    grouped = (
        raw.groupby(["model", "size", "variant", "quantization", "mode", "concurrency"], as_index=False)
        .agg(
            output_tps=("output_throughput", "median"),
            output_tps_min=("output_throughput", "min"),
            output_tps_max=("output_throughput", "max"),
            request_rps=("request_throughput", "median"),
            p50_ttft_ms=("p50_ttft_ms", "median"),
            p95_ttft_ms=("p95_ttft_ms", "median"),
            p50_e2e_ms=("p50_e2el_ms", "median"),
            p95_e2e_ms=("p95_e2el_ms", "median"),
            p95_e2e_min=("p95_e2el_ms", "min"),
            p95_e2e_max=("p95_e2el_ms", "max"),
        )
        .sort_values(["mode", "size", "quantization", "concurrency"])
    )
    return raw, grouped


def _training_model_size(model_id: str) -> str:
    value = model_id.lower()
    if "gemma-3-270m" in value:
        return "270m"
    if "gemma-4-e2b" in value:
        return "e2b"
    if "gemma-4-e4b" in value:
        return "e4b"
    raise ValueError(f"Unrecognized training model ID: {model_id}")


def _step_number(name: str) -> int:
    match = re.search(r"step_(\d+)", name)
    return int(match.group(1)) if match else -1


def _zip_training_records(path: Path) -> Iterable[tuple[dict, dict, dict, list[dict]]]:
    with zipfile.ZipFile(path) as archive:
        configs = sorted(
            (name for name in archive.namelist() if re.search(r"metrics/sft/step_\d+_config\.json$", name)),
            key=_step_number,
        )
        if not configs:
            raise ValueError(f"No metrics/sft/step_*_config.json found in {path}")
        config_name = configs[-1]
        base = PurePosixPath(config_name).parent
        step = _step_number(config_name)
        summary_name = str(base / f"step_{step}.json")
        history_name = str(base / f"step_{step}_history.jsonl")
        evaluation_name = str(base / "checkpoint_evaluation.json")
        config = json.loads(archive.read(config_name))
        summary = json.loads(archive.read(summary_name))
        evaluation = json.loads(archive.read(evaluation_name))
        history = [json.loads(line) for line in archive.read(history_name).decode("utf-8").splitlines() if line.strip()]
        yield config, summary, evaluation, history


def _directory_training_records(path: Path) -> Iterable[tuple[dict, dict, dict, list[dict]]]:
    configs = sorted(path.rglob("step_*_config.json"), key=lambda item: (_step_number(item.name), str(item)))
    if not configs:
        raise ValueError(f"No step_*_config.json found under {path}")
    by_metrics_dir: dict[Path, Path] = {}
    for config_path in configs:
        by_metrics_dir[config_path.parent] = config_path
    for config_path in by_metrics_dir.values():
        step = _step_number(config_path.name)
        metrics_dir = config_path.parent
        summary_path = metrics_dir / f"step_{step}.json"
        history_path = metrics_dir / f"step_{step}_history.jsonl"
        evaluation_path = metrics_dir / "checkpoint_evaluation.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        yield config, summary, evaluation, history


def load_training_metrics(paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    histories = []
    for path in paths:
        records = _zip_training_records(path) if path.suffix.lower() == ".zip" else _directory_training_records(path)
        for config, summary, evaluation, history in records:
            model_id = config["sft"]["model_id"]
            size = _training_model_size(model_id)
            rank = int(config["lora"]["r"])
            best = evaluation.get("best", {})
            run_name = config.get("run", {}).get("project_name", f"{size}-r{rank}")
            summaries.append(
                {
                    "run": run_name,
                    "size": size,
                    "model": MODEL_LABELS[size],
                    "base_model": model_id,
                    "lora_rank": rank,
                    "lora_alpha": int(config["lora"]["alpha"]),
                    "epochs": float(summary["num_train_epochs"]),
                    "samples": int(summary["sample_count"]),
                    "optimizer_steps": int(summary["optimizer_global_step"]),
                    "effective_batch": int(summary["effective_train_batch_size"]),
                    "training_seconds": float(summary["training_loop_seconds"]),
                    "stage_seconds": float(summary["stage_wall_seconds"]),
                    "train_loss": float(summary["train_loss"]),
                    "processed_input_tokens": int(summary["fen_sft_processed_input_tokens"]),
                    "input_tokens_per_second": float(summary["fen_sft_processed_input_tokens_per_second"]),
                    "best_checkpoint": best.get("checkpoint"),
                    "selection_f1": best.get("fen_sft_val_all_moves_f1"),
                }
            )
            for point in history:
                if "loss" not in point:
                    continue
                histories.append(
                    {
                        "run": run_name,
                        "size": size,
                        "model": MODEL_LABELS[size],
                        "lora_rank": rank,
                        "step": int(point.get("step", 0)),
                        "epoch": float(point.get("epoch", 0)),
                        "loss": float(point["loss"]),
                    }
                )
    summary_frame = pd.DataFrame(summaries)
    history_frame = pd.DataFrame(histories)
    if summary_frame.empty or history_frame.empty:
        raise ValueError("Training inputs did not produce both summary and loss-history rows.")
    return summary_frame.sort_values(["size", "lora_rank"]), history_frame.sort_values(["size", "lora_rank", "step"])


def join_quality_throughput(quality: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    quality_r32 = quality[quality["variant"] == "r32"].copy()
    throughput = performance[performance["concurrency"] == 256].copy()
    joined = throughput.merge(
        quality_r32,
        on=["model", "size", "variant", "quantization"],
        validate="many_to_one",
        suffixes=("", "_quality"),
    )
    if len(joined) != 18 or joined["model"].nunique() != 9:
        raise ValueError("Expected nine rank-32 checkpoints in each of two throughput modes.")
    return joined.sort_values(["mode", "size", "quantization"])


def write_result_tables(
    output_dir: Path,
    quality: pd.DataFrame,
    performance_raw: pd.DataFrame,
    performance_median: pd.DataFrame,
    training: pd.DataFrame,
    training_history: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(output_dir / "quality_results.csv", index=False)
    performance_raw.to_csv(output_dir / "performance_repetitions.csv", index=False)
    performance_median.to_csv(output_dir / "performance_medians.csv", index=False)
    training.to_csv(output_dir / "training_runs.csv", index=False)
    training_history.to_csv(output_dir / "training_history.csv", index=False)
    join_quality_throughput(quality, performance_median).to_csv(
        output_dir / "quality_throughput_joined.csv", index=False
    )

