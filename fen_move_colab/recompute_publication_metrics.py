"""Recompute corrected metrics from saved benchmark predictions."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

from .publication_eval import aggregate_scores, score_completion, write_json, write_jsonl


def _prediction_sources(path: Path) -> Iterator[Tuple[str, Iterable[str]]]:
    if path.is_dir():
        for prediction_path in sorted(path.glob("*_predictions.jsonl")):
            yield prediction_path.name, prediction_path.read_text(encoding="utf-8").splitlines()
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("_predictions.jsonl"):
                    yield Path(name).name, archive.read(name).decode("utf-8").splitlines()
        return
    raise ValueError("--input must be a benchmark directory or ZIP archive.")


def _flat_summary(model_name: str, metrics: Dict) -> Dict:
    overall = metrics["overall"]
    check = metrics["in_check"]
    non_check = metrics["not_in_check"]
    return {
        "name": model_name,
        "samples": overall["samples"],
        "move_set_f1": overall["macro_f1"],
        "exact_position_accuracy": overall["set_exact_rate"],
        "precision": overall["macro_precision"],
        "recall": overall["macro_recall"],
        "illegal_move_rate": overall["illegal_move_rate"],
        "strict_format_rate": overall["strict_format_rate"],
        "end_of_turn_observed_rate": overall["end_of_turn_observed_rate"],
        "check_positions": check["samples"],
        "check_move_set_f1": check["macro_f1"],
        "check_exact_position_accuracy": check["set_exact_rate"],
        "non_check_positions": non_check["samples"],
        "non_check_move_set_f1": non_check["macro_f1"],
        "non_check_exact_position_accuracy": non_check["set_exact_rate"],
        "illegal_moves_from_check_rate": metrics["illegal_moves_from_check_rate"],
    }


def recompute(input_path: Path, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for filename, lines in _prediction_sources(input_path):
        model_name = filename[: -len("_predictions.jsonl")]
        details = []
        for line in lines:
            if not line.strip():
                continue
            saved = json.loads(line)
            record = {
                "puzzle_id": saved.get("puzzle_id"),
                "source_game_id": saved.get("source_game_id"),
                "fen": saved["fen"],
                "legal_moves": saved["expected_moves"],
            }
            details.append(score_completion(record, saved.get("completion", "")))
        metrics = aggregate_scores(details)
        report = {"name": model_name, "metrics": metrics}
        reports.append(report)
        write_json(output_dir / f"{model_name}_corrected.json", report)
        write_jsonl(output_dir / f"{model_name}_corrected_predictions.jsonl", details)

    summaries = [_flat_summary(report["name"], report["metrics"]) for report in reports]
    if summaries:
        with (output_dir / "corrected_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    payload = {
        "contract": "publication_eval_v1_eot_truncated_engine_verified",
        "source": str(input_path.resolve()),
        "models": reports,
    }
    write_json(output_dir / "corrected_comparison.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute EOT-corrected, engine-verified benchmark metrics."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    payload = recompute(args.input, args.output_dir)
    print(
        json.dumps(
            {
                "models": len(payload["models"]),
                "output": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
