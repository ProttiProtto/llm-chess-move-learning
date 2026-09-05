"""Validate split isolation and basic record integrity for benchmark datasets."""

import argparse
import json
import os
from collections import Counter
from typing import Dict, Iterable, Tuple


def _rows(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _canonical_fen(fen: str) -> str:
    return " ".join(str(fen).split()[:4])


def _inspect(path: str, require_output: bool) -> Tuple[int, set, set, Counter]:
    rows = 0
    games = set()
    fens = set()
    ratings = Counter()
    for record in _rows(path):
        rows += 1
        if require_output and not record.get("output"):
            raise ValueError(f"SFT record without output in {path}")
        if not record.get("fen") or not record.get("source_game_id"):
            raise ValueError(f"Record missing FEN or source game ID in {path}")
        games.add(str(record["source_game_id"]))
        fens.add(_canonical_fen(record["fen"]))
        ratings[str(record.get("rating_bucket", "unknown"))] += 1
    return rows, games, fens, ratings


def validate(dataset_dir: str) -> Dict:
    paths = {
        "sft_train": os.path.join(dataset_dir, "sft_train.jsonl"),
        "grpo_train": os.path.join(dataset_dir, "grpo_train.jsonl"),
        "validation": os.path.join(dataset_dir, "validation.jsonl"),
    }
    for path in paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    inspected = {
        "sft_train": _inspect(paths["sft_train"], require_output=True),
        "grpo_train": _inspect(paths["grpo_train"], require_output=False),
        "validation": _inspect(paths["validation"], require_output=False),
    }
    output = {}
    for split, (row_count, games, fens, ratings) in inspected.items():
        output[split] = {
            "rows": row_count,
            "unique_source_games": len(games),
            "unique_canonical_fens": len(fens),
            "rating_distribution": dict(ratings),
        }

    pairs = (("sft_train", "grpo_train"), ("sft_train", "validation"), ("grpo_train", "validation"))
    overlap = {}
    for left, right in pairs:
        _, left_games, left_fens, _ = inspected[left]
        _, right_games, right_fens, _ = inspected[right]
        games = left_games & right_games
        fens = left_fens & right_fens
        overlap[f"{left}__{right}"] = {
            "source_games": len(games),
            "canonical_fens": len(fens),
        }
        if games or fens:
            raise ValueError(f"Leakage detected between {left} and {right}: {overlap[f'{left}__{right}']}")
    output["overlap"] = overlap
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark dataset split isolation.")
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()
