"""Build a reproducible, leakage-aware FEN-to-UCI benchmark from Lichess.

The source archive is streamed once. Candidate puzzles are first filtered by
community-quality signals, then selected with deterministic hash reservoirs.
All puzzles from one Lichess game go to one split, and canonical FEN overlap is
removed between SFT, GRPO, and held-out validation data.
"""

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import chess
import pandas as pd

RATING_BUCKETS = (
    (0, 999, "under_1000"),
    (1000, 1199, "1000_1199"),
    (1200, 1399, "1200_1399"),
    (1400, 1599, "1400_1599"),
    (1600, 1799, "1600_1799"),
    (1800, 1999, "1800_1999"),
    (2000, 2199, "2000_2199"),
    (2200, 2399, "2200_2399"),
    (2400, 2599, "2400_2599"),
    (2600, 10**9, "2600_plus"),
)

# The full Lichess distribution is heavily concentrated below 1,200. These
# weights retain that beginner range without allowing it to dominate the demo.
RATING_WEIGHTS = {
    "under_1000": 0.060,
    "1000_1199": 0.090,
    "1200_1399": 0.130,
    "1400_1599": 0.160,
    "1600_1799": 0.150,
    "1800_1999": 0.150,
    "2000_2199": 0.121,
    "2200_2399": 0.080,
    "2400_2599": 0.050,
    "2600_plus": 0.009,
}

THEME_WEIGHTS = {
    "mate": 0.30,
    "tactical": 0.35,
    "endgame": 0.25,
    "other": 0.10,
}

TACTICAL_THEMES = {
    "advancedPawn",
    "attraction",
    "backRankMate",
    "clearance",
    "deflection",
    "defensiveMove",
    "discoveredAttack",
    "doubleCheck",
    "exposedKing",
    "fork",
    "hangingPiece",
    "interference",
    "kingsideAttack",
    "pin",
    "promotion",
    "quietMove",
    "sacrifice",
    "skewer",
    "trappedPiece",
}

SPLIT_BOUNDARIES = (
    ("sft_train", 0.68),
    ("grpo_train", 0.88),
    ("validation", 1.00),
)


def make_instruction(fen: str) -> str:
    """Keep data building independent of the Torch-backed training modules."""
    return (
        "You are playing chess. Given this position in FEN, output exactly one legal move "
        "in UCI notation.\n"
        f"FEN: {fen}\n\n"
        "Respond with ONLY the UCI move, such as e2e4, g1f3, or e7e8q."
    )

def _stable_hash(*parts: object) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\0")
    return int.from_bytes(hasher.digest(), "big")


def _rating_bucket(rating: float) -> str:
    for lower, upper, label in RATING_BUCKETS:
        if lower <= rating <= upper:
            return label
    return "unknown"


def _theme_bucket(themes: Iterable[str]) -> str:
    theme_set = set(themes)
    if "mate" in theme_set or any(theme.startswith("mateIn") for theme in theme_set):
        return "mate"
    if "endgame" in theme_set:
        return "endgame"
    if theme_set & TACTICAL_THEMES:
        return "tactical"
    return "other"


def _game_id(game_url: str, puzzle_id: str) -> str:
    parts = str(game_url).split("/")
    if len(parts) >= 4 and parts[3]:
        return parts[3]
    return f"puzzle-{puzzle_id}"


def _split_for_game(game_id: str, seed: int) -> str:
    fraction = _stable_hash("split", seed, game_id) / float(2**64)
    for split, boundary in SPLIT_BOUNDARIES:
        if fraction < boundary:
            return split
    return "validation"


def _largest_remainder(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    raw = {key: total * value for key, value in weights.items()}
    result = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    for key in sorted(weights, key=lambda item: (raw[item] - result[item], item), reverse=True)[:remaining]:
        result[key] += 1
    return result


def _as_text(value) -> str:
    return "" if pd.isna(value) else str(value)


def _candidate_from_row(row, seed: int) -> Optional[Dict]:
    moves = _as_text(row.Moves).split()
    if len(moves) < 2:
        return None
    try:
        rating = int(float(row.Rating))
        deviation = float(row.RatingDeviation)
        popularity = float(row.Popularity)
        plays = int(float(row.NbPlays))
    except (TypeError, ValueError):
        return None

    # These thresholds retain 3.15M of 6.06M current puzzles: enough breadth
    # for sampling while excluding new/uncertain and unpopular puzzle labels.
    if deviation > 80 or popularity < 50 or plays < 100:
        return None

    puzzle_id = _as_text(row.PuzzleId)
    game_id = _game_id(_as_text(row.GameUrl), puzzle_id)
    themes = _as_text(row.Themes).split()
    return {
        "priority": _stable_hash("candidate", seed, puzzle_id),
        "puzzle_id": puzzle_id,
        "source_game_id": game_id,
        "source_fen": _as_text(row.FEN),
        "moves": " ".join(moves),
        "rating": rating,
        "rating_deviation": deviation,
        "popularity": popularity,
        "nb_plays": plays,
        "themes": themes,
        "opening_tags": _as_text(row.OpeningTags).split(),
        "rating_bucket": _rating_bucket(rating),
        "theme_bucket": _theme_bucket(themes),
    }


def _add_reservoir_item(heap: List[Tuple[int, str, Dict]], candidate: Dict, capacity: int) -> None:
    # Store negative priorities so heap[0] is the worst candidate retained.
    item = (-candidate["priority"], candidate["puzzle_id"], candidate)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _position_from_candidate(candidate: Dict) -> Optional[Dict]:
    moves = candidate["moves"].split()
    try:
        board = chess.Board(candidate["source_fen"])
        opponent_move = chess.Move.from_uci(moves[0])
        solution_move = chess.Move.from_uci(moves[1])
    except ValueError:
        return None
    if opponent_move not in board.legal_moves:
        return None
    board.push(opponent_move)
    if solution_move not in board.legal_moves:
        return None

    fen = board.fen()
    legal_moves = sorted(move.uci() for move in board.legal_moves)
    if not legal_moves:
        return None
    return {
        "puzzle_id": candidate["puzzle_id"],
        "source_game_id": candidate["source_game_id"],
        "instruction": make_instruction(fen),
        "input": "",
        "fen": fen,
        # Clocks do not change legal moves, so use the board-relevant fields to
        # prevent a position from appearing in more than one split.
        "fen_key": " ".join(fen.split()[:4]),
        "solution_uci": solution_move.uci(),
        "legal_moves": legal_moves,
        "themes": candidate["themes"],
        "theme_bucket": candidate["theme_bucket"],
        "rating": candidate["rating"],
        "rating_bucket": candidate["rating_bucket"],
        "rating_deviation": candidate["rating_deviation"],
        "popularity": candidate["popularity"],
        "nb_plays": candidate["nb_plays"],
        "opening_tags": candidate["opening_tags"],
    }


def _select_positions(
    reservoirs: Dict[str, List[Tuple[int, str, Dict]]],
    total: int,
    blocked_fens: set,
) -> Tuple[List[Dict], Dict]:
    rating_targets = _largest_remainder(total, RATING_WEIGHTS)
    selections: List[Dict] = []
    seen_fens = set(blocked_fens)
    stats = {"requested": total, "by_rating": {}, "by_theme": Counter(), "invalid_candidates": 0}

    for rating_bucket, target in rating_targets.items():
        if target <= 0:
            stats["by_rating"][rating_bucket] = 0
            continue
        candidates = sorted((item[2] for item in reservoirs.get(rating_bucket, [])), key=lambda item: item["priority"])
        theme_targets = _largest_remainder(target, THEME_WEIGHTS)
        selected_ids = set()
        selected_for_rating = 0

        def try_add(candidate: Dict) -> bool:
            nonlocal selected_for_rating
            position = _position_from_candidate(candidate)
            if position is None:
                stats["invalid_candidates"] += 1
                return False
            if position["fen_key"] in seen_fens:
                return False
            seen_fens.add(position["fen_key"])
            selected_ids.add(candidate["puzzle_id"])
            selections.append(position)
            selected_for_rating += 1
            stats["by_theme"][position["theme_bucket"]] += 1
            return True

        for theme_bucket, theme_target in theme_targets.items():
            if theme_target <= 0:
                continue
            taken = 0
            for candidate in candidates:
                if candidate["theme_bucket"] != theme_bucket or candidate["puzzle_id"] in selected_ids:
                    continue
                if try_add(candidate):
                    taken += 1
                if taken >= theme_target:
                    break

        # Sparse high-rating/theme intersections are backfilled from the same
        # rating range instead of silently reducing the requested dataset size.
        if selected_for_rating < target:
            for candidate in candidates:
                if candidate["puzzle_id"] in selected_ids:
                    continue
                try_add(candidate)
                if selected_for_rating >= target:
                    break
        stats["by_rating"][rating_bucket] = selected_for_rating

    if len(selections) != total:
        missing = total - len(selections)
        raise RuntimeError(
            f"Could only select {len(selections):,} of {total:,} requested positions; "
            f"{missing:,} are missing. Increase the reservoir multiplier or relax the filter."
        )
    stats["by_theme"] = dict(stats["by_theme"])
    return selections, stats


def _write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1
    return count


def _sft_records(positions: Iterable[Dict]) -> Iterable[Dict]:
    for position in positions:
        for move in position["legal_moves"]:
            yield {
                "puzzle_id": position["puzzle_id"],
                "source_game_id": position["source_game_id"],
                "instruction": position["instruction"],
                "input": "",
                "fen": position["fen"],
                "output": move,
                "solution_uci": position["solution_uci"],
                "is_solution": move == position["solution_uci"],
                "rating": position["rating"],
                "rating_bucket": position["rating_bucket"],
                "theme_bucket": position["theme_bucket"],
                "themes": position["themes"],
            }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_datasets(
    puzzle_csv: str,
    output_dir: str,
    sft_positions: int,
    grpo_positions: int,
    validation_positions: int,
    seed: int,
    chunksize: int,
    reservoir_multiplier: int,
) -> Dict:
    requested = {
        "sft_train": int(sft_positions),
        "grpo_train": int(grpo_positions),
        "validation": int(validation_positions),
    }
    rating_targets = {
        split: _largest_remainder(total, RATING_WEIGHTS)
        for split, total in requested.items()
    }
    reservoirs = {
        split: {bucket: [] for _, _, bucket in RATING_BUCKETS}
        for split in requested
    }
    raw_rows = 0
    quality_candidates = 0
    candidates_by_split = {split: Counter() for split in requested}
    usecols = [
        "PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity",
        "NbPlays", "Themes", "GameUrl", "OpeningTags",
    ]

    for chunk_index, chunk in enumerate(
        pd.read_csv(puzzle_csv, usecols=usecols, chunksize=chunksize),
        start=1,
    ):
        raw_rows += len(chunk)
        for row in chunk.itertuples(index=False):
            candidate = _candidate_from_row(row, seed)
            if candidate is None:
                continue
            quality_candidates += 1
            split = _split_for_game(candidate["source_game_id"], seed)
            candidates_by_split[split][candidate["rating_bucket"]] += 1
            capacity = max(100, rating_targets[split][candidate["rating_bucket"]] * reservoir_multiplier)
            _add_reservoir_item(reservoirs[split][candidate["rating_bucket"]], candidate, capacity)

        if chunk_index == 1 or chunk_index % 10 == 0:
            print(f"Scanned {raw_rows:,} raw puzzles...", flush=True)

    sft, sft_stats = _select_positions(reservoirs["sft_train"], requested["sft_train"], blocked_fens=set())
    train_fens = {position["fen_key"] for position in sft}
    grpo, grpo_stats = _select_positions(reservoirs["grpo_train"], requested["grpo_train"], blocked_fens=train_fens)
    train_fens.update(position["fen_key"] for position in grpo)
    validation, validation_stats = _select_positions(
        reservoirs["validation"], requested["validation"], blocked_fens=train_fens
    )

    os.makedirs(output_dir, exist_ok=True)
    sft_path = os.path.join(output_dir, "sft_train.jsonl")
    grpo_path = os.path.join(output_dir, "grpo_train.jsonl")
    validation_path = os.path.join(output_dir, "validation.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")
    sft_record_count = _write_jsonl(sft_path, _sft_records(sft))
    _write_jsonl(grpo_path, ({key: value for key, value in position.items() if key != "fen_key"} for position in grpo))
    _write_jsonl(validation_path, ({key: value for key, value in position.items() if key != "fen_key"} for position in validation))

    manifest = {
        "version": "lichess-legal-move-v1",
        "source": {
            "puzzle_csv": os.path.abspath(puzzle_csv),
            "puzzle_csv_sha256": _sha256(puzzle_csv),
            "raw_rows_scanned": raw_rows,
            "quality_candidates": quality_candidates,
        },
        "quality_filter": {
            "rating_deviation_lte": 80,
            "popularity_gte": 50,
            "nb_plays_gte": 100,
        },
        "split_policy": {
            "group_key": "source_game_id",
            "group_allocation": {"sft_train": 0.68, "grpo_train": 0.20, "validation": 0.12},
            "canonical_fen_overlap": "removed from later splits",
            "position_policy": "first tactical solution position only",
            "rating_weights": RATING_WEIGHTS,
            "theme_weights": THEME_WEIGHTS,
            "seed": seed,
        },
        "candidates_by_split_and_rating": {split: dict(counts) for split, counts in candidates_by_split.items()},
        "outputs": {
            "sft_train": {"path": sft_path, "positions": len(sft), "records": sft_record_count, "selection": sft_stats},
            "grpo_train": {"path": grpo_path, "positions": len(grpo), "selection": grpo_stats},
            "validation": {"path": validation_path, "positions": len(validation), "selection": validation_stats},
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced SFT, GRPO, and validation datasets from full Lichess.")
    parser.add_argument("--puzzle-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sft-positions", type=int, default=60_000)
    parser.add_argument("--grpo-positions", type=int, default=15_000)
    parser.add_argument("--validation-positions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--reservoir-multiplier", type=int, default=3)
    args = parser.parse_args()
    manifest = build_datasets(
        puzzle_csv=args.puzzle_csv,
        output_dir=args.output_dir,
        sft_positions=args.sft_positions,
        grpo_positions=args.grpo_positions,
        validation_positions=args.validation_positions,
        seed=args.seed,
        chunksize=args.chunksize,
        reservoir_multiplier=args.reservoir_multiplier,
    )
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
