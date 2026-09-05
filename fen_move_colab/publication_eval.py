"""Reproducible scoring and dataset contracts for publication evaluation."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import chess

from .build_all_legal_moves_dataset import make_all_legal_moves_instruction
from .build_benchmark_datasets import make_instruction as make_legacy_one_move_instruction


END_OF_TURN = "<end_of_turn>"
PROMPT_SOURCE = (
    "fen_move_colab.build_all_legal_moves_dataset."
    "make_all_legal_moves_instruction"
)
PROMPT_WRAPPER = (
    "<start_of_turn>user\n{instruction}<end_of_turn>\n"
    "<start_of_turn>model\n"
)
UCI_TOKEN_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?", re.IGNORECASE)


def jsonl_rows(path: os.PathLike[str] | str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def write_json(path: os.PathLike[str] | str, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: os.PathLike[str] | str, rows: Iterable[Mapping]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), separators=(",", ":")) + "\n")


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fen(fen: str) -> str:
    board = chess.Board(str(fen))
    return " ".join(board.fen().split()[:4])


def build_all_moves_prompt(fen: str) -> str:
    instruction = make_all_legal_moves_instruction(fen)
    return PROMPT_WRAPPER.format(instruction=instruction)


def prompt_contract() -> Dict:
    instruction_source = inspect.getsource(make_all_legal_moves_instruction)
    wrapper_source = PROMPT_WRAPPER
    return {
        "prompt_source": PROMPT_SOURCE,
        "prompt_wrapper": wrapper_source,
        "instruction_source_sha256": hashlib.sha256(
            instruction_source.encode("utf-8")
        ).hexdigest(),
        "prompt_contract_sha256": hashlib.sha256(
            (instruction_source + "\n" + wrapper_source).encode("utf-8")
        ).hexdigest(),
    }


def truncate_at_end_of_turn(text: str, marker: str = END_OF_TURN) -> Tuple[str, bool, str]:
    raw = text or ""
    if marker not in raw:
        return raw, False, ""
    before, after = raw.split(marker, 1)
    return before, True, after


def resolve_stop_config(tokenizer, marker: str = END_OF_TURN) -> Dict:
    encoded = tokenizer.encode(marker, add_special_tokens=False)
    token_id = tokenizer.convert_tokens_to_ids(marker)
    unknown = tokenizer.unk_token_id
    stop_ids = []
    if token_id is not None and token_id != unknown:
        stop_ids.append(int(token_id))
    if len(encoded) == 1 and int(encoded[0]) not in stop_ids:
        stop_ids.append(int(encoded[0]))
    return {
        "stop_strings": [marker],
        "stop_token_ids": stop_ids,
        "end_of_turn_encoded_ids": [int(value) for value in encoded],
        "token_id_candidate": int(token_id) if token_id is not None else None,
        "token_id_candidate_is_unk": token_id is not None and token_id == unknown,
        "strategy": "stop_string_plus_verified_single_token_ids",
        "warning": (
            None
            if stop_ids
            else "No trustworthy standalone EOT token ID; using string stopping and model EOS."
        ),
    }


def parse_uci_moves(text: str, marker: str = END_OF_TURN) -> Dict:
    truncated, saw_eot, after_eot = truncate_at_end_of_turn(text, marker)
    tokens = truncated.lower().split()
    valid_tokens = [token for token in tokens if UCI_TOKEN_RE.fullmatch(token)]
    malformed_tokens = [token for token in tokens if not UCI_TOKEN_RE.fullmatch(token)]
    unique_moves: List[str] = []
    seen = set()
    duplicates: List[str] = []
    for move in valid_tokens:
        if move in seen:
            duplicates.append(move)
            continue
        seen.add(move)
        unique_moves.append(move)
    return {
        "truncated_completion": truncated,
        "saw_end_of_turn": saw_eot,
        "text_after_end_of_turn": after_eot,
        "moves": unique_moves,
        "duplicates": duplicates,
        "malformed_tokens": malformed_tokens,
        "strict_format_valid": bool(tokens) and not malformed_tokens and not duplicates,
        "empty_output": not tokens,
    }


def legal_moves_for_fen(fen: str) -> Tuple[chess.Board, List[str]]:
    board = chess.Board(str(fen))
    if not board.is_valid():
        raise ValueError(f"Invalid chess position: {fen}")
    return board, sorted(move.uci() for move in board.legal_moves)


def verify_record_ground_truth(record: Mapping) -> Tuple[chess.Board, List[str]]:
    if not record.get("fen"):
        raise ValueError("Evaluation record is missing fen.")
    board, generated = legal_moves_for_fen(str(record["fen"]))
    stored = sorted(set(str(move).lower() for move in record.get("legal_moves", [])))
    if not stored:
        raise ValueError(f"Evaluation record has no legal_moves: {record.get('puzzle_id')}")
    if stored != generated:
        missing = sorted(set(generated) - set(stored))
        extra = sorted(set(stored) - set(generated))
        raise ValueError(
            f"Ground-truth mismatch for {record.get('puzzle_id')}: "
            f"missing={missing}, extra={extra}"
        )
    return board, generated


def score_completion(record: Mapping, completion: str) -> Dict:
    board, expected = verify_record_ground_truth(record)
    parsed = parse_uci_moves(completion)
    predicted = parsed["moves"]
    expected_set = set(expected)
    predicted_set = set(predicted)
    correct = expected_set & predicted_set
    precision = len(correct) / len(predicted_set) if predicted_set else 0.0
    recall = len(correct) / len(expected_set) if expected_set else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    missed = sorted(expected_set - predicted_set)
    illegal = sorted(predicted_set - expected_set)
    return {
        "puzzle_id": record.get("puzzle_id"),
        "source_game_id": record.get("source_game_id"),
        "fen": record["fen"],
        "in_check": board.is_check(),
        "expected_moves": expected,
        "predicted_moves": predicted,
        "completion": completion,
        "scored_completion": parsed["truncated_completion"],
        "saw_end_of_turn": parsed["saw_end_of_turn"],
        "text_after_end_of_turn": parsed["text_after_end_of_turn"],
        "strict_format_valid": parsed["strict_format_valid"],
        "empty_output": parsed["empty_output"],
        "duplicate_moves": parsed["duplicates"],
        "malformed_tokens": parsed["malformed_tokens"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "set_exact": predicted_set == expected_set,
        "ordered_exact": predicted == expected,
        "missed_moves": missed,
        "illegal_moves": illegal,
        "correct_move_count": len(correct),
        "expected_move_count": len(expected_set),
        "predicted_move_count": len(predicted_set),
        "illegal_move_count": len(illegal),
    }


def _mean(rows: Sequence[Mapping], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _aggregate_group(rows: Sequence[Mapping]) -> Dict:
    expected = sum(int(row["expected_move_count"]) for row in rows)
    predicted = sum(int(row["predicted_move_count"]) for row in rows)
    correct = sum(int(row["correct_move_count"]) for row in rows)
    micro_precision = correct / predicted if predicted else 0.0
    micro_recall = correct / expected if expected else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "samples": len(rows),
        "macro_precision": _mean(rows, "precision"),
        "macro_recall": _mean(rows, "recall"),
        "macro_f1": _mean(rows, "f1"),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "set_exact_rate": _mean(rows, "set_exact"),
        "ordered_exact_rate": _mean(rows, "ordered_exact"),
        "strict_format_rate": _mean(rows, "strict_format_valid"),
        "empty_output_rate": _mean(rows, "empty_output"),
        "end_of_turn_observed_rate": _mean(rows, "saw_end_of_turn"),
        "predicted_moves": predicted,
        "expected_moves": expected,
        "correct_moves": correct,
        "illegal_moves": sum(int(row["illegal_move_count"]) for row in rows),
        "illegal_move_rate": (
            sum(int(row["illegal_move_count"]) for row in rows) / predicted
            if predicted
            else 0.0
        ),
    }


def aggregate_scores(details: Sequence[Mapping]) -> Dict:
    rows = list(details)
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation.")
    check_rows = [row for row in rows if row["in_check"]]
    non_check_rows = [row for row in rows if not row["in_check"]]
    overall = _aggregate_group(rows)
    in_check = _aggregate_group(check_rows)
    non_check = _aggregate_group(non_check_rows)
    total_illegal = int(overall["illegal_moves"])
    check_illegal = int(in_check["illegal_moves"])
    return {
        "metric_contract": "publication_eval_v1_eot_truncated_engine_verified",
        "overall": overall,
        "in_check": in_check,
        "not_in_check": non_check,
        "check_position_rate": len(check_rows) / len(rows),
        "illegal_moves_from_check_rate": check_illegal / total_illegal if total_illegal else 0.0,
    }


def score_records(records: Sequence[Mapping], completions: Sequence[str]) -> Tuple[Dict, List[Dict]]:
    if len(records) != len(completions):
        raise ValueError(
            f"Record/completion count mismatch: {len(records)} != {len(completions)}"
        )
    details = [score_completion(row, completion) for row, completion in zip(records, completions)]
    return aggregate_scores(details), details


def audit_prompt_contract(
    validation_records: Sequence[Mapping],
    training_path: os.PathLike[str] | str,
) -> Dict:
    validation_counts = {"all_moves": 0, "legacy_one_move": 0, "other": 0, "missing": 0}
    for row in validation_records:
        stored = row.get("instruction")
        if not stored:
            validation_counts["missing"] += 1
        elif stored == make_all_legal_moves_instruction(row["fen"]):
            validation_counts["all_moves"] += 1
        elif stored == make_legacy_one_move_instruction(row["fen"]):
            validation_counts["legacy_one_move"] += 1
        else:
            validation_counts["other"] += 1

    training_counts = {"rows": 0, "matching_all_moves": 0, "mismatched": 0}
    for row in jsonl_rows(training_path):
        training_counts["rows"] += 1
        if row.get("instruction") == make_all_legal_moves_instruction(row["fen"]):
            training_counts["matching_all_moves"] += 1
        else:
            training_counts["mismatched"] += 1

    if training_counts["mismatched"]:
        raise ValueError(
            f"Training prompt audit found {training_counts['mismatched']} mismatched rows."
        )
    return {
        **prompt_contract(),
        "validation_stored_instruction_counts": validation_counts,
        "training_instruction_counts": training_counts,
        "evaluation_prompt_is_regenerated_from_fen": True,
        "evaluation_prompt_matches_training": True,
        "legacy_validation_instruction_is_not_used": True,
        "evaluation_kind": "matched_training_prompt",
    }


def _identity_hash(rows: Sequence[Mapping]) -> str:
    identity = "\n".join(
        f"{row.get('puzzle_id', '')}\t{canonical_fen(row['fen'])}" for row in rows
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def prepare_publication_split(
    validation_path: os.PathLike[str] | str,
    training_paths: Sequence[os.PathLike[str] | str],
    output_dir: os.PathLike[str] | str,
    selection_count: int = 128,
) -> Dict:
    records = list(jsonl_rows(validation_path))
    if selection_count < 0 or selection_count >= len(records):
        raise ValueError(
            f"selection_count must be in [0, {len(records) - 1}], got {selection_count}."
        )
    selection_rows = records[:selection_count]
    test_rows = records[selection_count:]
    selection_fens = {canonical_fen(row["fen"]) for row in selection_rows}
    test_fens = {canonical_fen(row["fen"]) for row in test_rows}
    selection_games = {
        str(row["source_game_id"]) for row in selection_rows if row.get("source_game_id")
    }
    test_games = {
        str(row["source_game_id"]) for row in test_rows if row.get("source_game_id")
    }
    missing_validation_game_ids = sum(
        not row.get("source_game_id") for row in records
    )
    if missing_validation_game_ids:
        raise ValueError(
            f"Validation split has {missing_validation_game_ids} rows without source_game_id; "
            "source-game leakage cannot be audited completely."
        )
    if selection_fens & test_fens or selection_games & test_games:
        raise ValueError("Checkpoint-selection and final-test partitions overlap.")
    if len(test_fens) != len(test_rows):
        raise ValueError("Final test partition contains duplicate canonical FENs.")

    training_fen_hits = set()
    training_game_hits = set()
    missing_training_game_ids = 0
    checked_paths = []
    for training_path in training_paths:
        checked_paths.append(str(Path(training_path).resolve()))
        for row in jsonl_rows(training_path):
            fen = canonical_fen(row["fen"])
            game = str(row.get("source_game_id") or "")
            if not game:
                missing_training_game_ids += 1
            if fen in test_fens:
                training_fen_hits.add(fen)
            if game and game in test_games:
                training_game_hits.add(game)
    if training_fen_hits or training_game_hits:
        raise ValueError(
            "Training leakage detected: "
            f"{len(training_fen_hits)} FEN and {len(training_game_hits)} game collisions."
        )
    if missing_training_game_ids:
        raise ValueError(
            f"Training data has {missing_training_game_ids} rows without source_game_id; "
            "source-game leakage cannot be audited completely."
        )

    # Verify every published target independently with python-chess.
    for row in test_rows:
        verify_record_ground_truth(row)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    test_hash = _identity_hash(test_rows)
    test_path = destination / f"heldout_{len(test_rows)}_{test_hash[:12]}.jsonl"
    write_jsonl(test_path, test_rows)
    manifest = {
        "contract": "publication_split_v1",
        "source_validation_path": str(Path(validation_path).resolve()),
        "source_validation_sha256": sha256_file(validation_path),
        "source_validation_positions": len(records),
        "checkpoint_selection_positions": len(selection_rows),
        "checkpoint_selection_identity_sha256": _identity_hash(selection_rows),
        "checkpoint_selection_excluded_from_test": True,
        "final_test_positions": len(test_rows),
        "final_test_identity_sha256": test_hash,
        "final_test_file_sha256": sha256_file(test_path),
        "final_test_path": str(test_path.resolve()),
        "training_paths_checked": checked_paths,
        "training_fen_collisions": 0,
        "training_game_collisions": 0,
        "validation_rows_missing_source_game_id": 0,
        "training_rows_missing_source_game_id": 0,
        "ground_truth_engine": f"python-chess {chess.__version__}",
        "ground_truth_engine_verified": True,
        **prompt_contract(),
    }
    write_json(destination / "split_manifest.json", manifest)
    return {"manifest": manifest, "records": test_rows, "selection_records": selection_rows}


def checkpoint_content_manifest(path: os.PathLike[str] | str) -> Dict:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    included_suffixes = {".safetensors", ".json", ".jinja", ".model"}
    files = sorted(
        file for file in root.rglob("*")
        if file.is_file() and (file.suffix in included_suffixes or file.name == "tokenizer.json")
    )
    entries = []
    aggregate = hashlib.sha256()
    for file in files:
        relative = file.relative_to(root).as_posix()
        digest = sha256_file(file)
        size = file.stat().st_size
        entries.append({"path": relative, "size_bytes": size, "sha256": digest})
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))
    return {
        "checkpoint_path": str(root.resolve()),
        "checkpoint_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "checkpoint_content_sha256": aggregate.hexdigest(),
        "files": entries,
    }
