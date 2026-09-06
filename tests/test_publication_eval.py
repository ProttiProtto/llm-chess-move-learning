import json
import tempfile
import unittest
from pathlib import Path

import chess

from fen_move_colab.build_all_legal_moves_dataset import make_all_legal_moves_instruction
from fen_move_colab.build_benchmark_datasets import make_instruction
from fen_move_colab.publication_eval import (
    END_OF_TURN,
    aggregate_scores,
    audit_prompt_contract,
    legal_moves_for_fen,
    parse_uci_moves,
    prepare_publication_split,
    resolve_stop_config,
    score_completion,
)


START_FEN = chess.STARTING_FEN


def _position_after(move: str) -> str:
    board = chess.Board()
    board.push_uci(move)
    return board.fen()


def _record(fen: str, puzzle_id: str, game_id: str) -> dict:
    _, moves = legal_moves_for_fen(fen)
    return {
        "puzzle_id": puzzle_id,
        "source_game_id": game_id,
        "fen": fen,
        "instruction": make_instruction(fen),
        "legal_moves": moves,
    }


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PublicationEvaluationTests(unittest.TestCase):
    def test_gemma4_unknown_eot_id_uses_string_stop_without_failing(self):
        class Gemma4TokenizerStub:
            unk_token_id = 3

            @staticmethod
            def encode(text, add_special_tokens=False):
                self.assertEqual(text, END_OF_TURN)
                self.assertFalse(add_special_tokens)
                return [236820, 643, 236779, 1340, 236779, 887, 236813]

            @staticmethod
            def convert_tokens_to_ids(text):
                self.assertEqual(text, END_OF_TURN)
                return 3

        stop = resolve_stop_config(Gemma4TokenizerStub())
        self.assertEqual(stop["stop_strings"], [END_OF_TURN])
        self.assertEqual(stop["stop_token_ids"], [])
        self.assertTrue(stop["token_id_candidate_is_unk"])
        self.assertIn("string stopping", stop["warning"])

    def test_invalid_fen_is_rejected(self):
        with self.assertRaises(ValueError):
            legal_moves_for_fen("not a FEN")

    def test_parser_ignores_everything_after_first_end_of_turn(self):
        parsed = parse_uci_moves(f"a2a3 a2a4{END_OF_TURN} h1h8{END_OF_TURN} b1c3")
        self.assertEqual(parsed["moves"], ["a2a3", "a2a4"])
        self.assertTrue(parsed["saw_end_of_turn"])
        self.assertIn("h1h8", parsed["text_after_end_of_turn"])

    def test_post_eot_move_cannot_reduce_exact_accuracy(self):
        record = _record(START_FEN, "p0", "g0")
        completion = " ".join(record["legal_moves"]) + END_OF_TURN + " a1a8"
        scored = score_completion(record, completion)
        self.assertTrue(scored["set_exact"])
        self.assertEqual(scored["f1"], 1.0)
        self.assertEqual(scored["illegal_moves"], [])

    def test_aggregate_reports_check_and_non_check(self):
        normal = _record(START_FEN, "p0", "g0")
        checked_fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        checked = _record(checked_fen, "p1", "g1")
        details = [
            score_completion(normal, " ".join(normal["legal_moves"])),
            score_completion(checked, " ".join(checked["legal_moves"])),
        ]
        metrics = aggregate_scores(details)
        self.assertEqual(metrics["overall"]["samples"], 2)
        self.assertEqual(metrics["in_check"]["samples"], 1)
        self.assertEqual(metrics["not_in_check"]["samples"], 1)

    def test_prompt_audit_labels_legacy_validation_as_unused(self):
        validation = [_record(START_FEN, "p0", "g0")]
        training = dict(validation[0])
        training["instruction"] = make_all_legal_moves_instruction(START_FEN)
        with tempfile.TemporaryDirectory() as directory:
            training_path = Path(directory) / "train.jsonl"
            _write_jsonl(training_path, [training])
            audit = audit_prompt_contract(validation, training_path)
        self.assertEqual(audit["evaluation_kind"], "matched_training_prompt")
        self.assertEqual(
            audit["validation_stored_instruction_counts"]["legacy_one_move"], 1
        )
        self.assertTrue(audit["legacy_validation_instruction_is_not_used"])

    def test_split_excludes_selection_and_checks_training_leakage(self):
        validation = [
            _record(START_FEN, "p0", "g0"),
            _record(_position_after("e2e4"), "p1", "g1"),
            _record(_position_after("d2d4"), "p2", "g2"),
        ]
        training = [_record(_position_after("c2c4"), "train", "train-game")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_path = root / "validation.jsonl"
            training_path = root / "training.jsonl"
            _write_jsonl(validation_path, validation)
            _write_jsonl(training_path, training)
            split = prepare_publication_split(
                validation_path, [training_path], root / "split", selection_count=1
            )
            self.assertEqual(len(split["selection_records"]), 1)
            self.assertEqual(len(split["records"]), 2)
            self.assertTrue(
                split["manifest"]["checkpoint_selection_excluded_from_test"]
            )

            _write_jsonl(training_path, [validation[1]])
            with self.assertRaisesRegex(ValueError, "Training leakage detected"):
                prepare_publication_split(
                    validation_path, [training_path], root / "leaky", selection_count=1
                )

            _write_jsonl(training_path, [validation[0]])
            with self.assertRaisesRegex(ValueError, "checkpoint-selection"):
                prepare_publication_split(
                    validation_path, [training_path], root / "selection-leaky", selection_count=1
                )

    def test_split_requires_training_files_for_a_complete_leakage_audit(self):
        validation = [_record(START_FEN, "p0", "g0"), _record(_position_after("e2e4"), "p1", "g1")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_path = root / "validation.jsonl"
            _write_jsonl(validation_path, validation)
            with self.assertRaisesRegex(ValueError, "At least one training file"):
                prepare_publication_split(validation_path, [], root / "split", selection_count=1)


if __name__ == "__main__":
    unittest.main()
