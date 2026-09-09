import itertools
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.build_v2_report import validate_scenarios
from fen_move_colab.publication_eval import legal_moves_for_fen, score_completion


def scenarios():
    return [
        {"model": "e4b-r32-bf16", "mode": mode, "concurrency": concurrency, "repetition": repetition,
         "completed": 128, "requested_prompts": 128, "failed": 0, "duration": 1.0,
         "output_throughput": 128.0, "request_throughput": 128.0, "p50_ttft_ms": 1.0,
         "p95_ttft_ms": 2.0, "p50_e2el_ms": 10.0, "p95_e2el_ms": 20.0}
        for mode, concurrency, repetition in itertools.product(
            ("natural_fen", "fixed_tokens"), (1, 8, 32, 128, 256), (1, 2, 3)
        )
    ]


def test_complete_scenario_matrix():
    validate_scenarios(scenarios(), ["e4b-r32-bf16"])


@pytest.mark.parametrize("fault", ["duplicate", "missing", "incomplete", "failed", "nan", "zero"])
def test_rejects_invalid_benchmark_evidence(fault):
    rows = scenarios()
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "missing":
        rows.pop()
    elif fault == "incomplete":
        rows[0]["completed"] = 127
    elif fault == "failed":
        rows[0]["failed"] = 1
    elif fault == "nan":
        rows[0]["output_throughput"] = float("nan")
    else:
        rows[0]["p95_e2el_ms"] = 0
    with pytest.raises(ValueError):
        validate_scenarios(rows, ["e4b-r32-bf16"])


def test_readme_cpu_example():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    _, moves = legal_moves_for_fen(fen)
    result = score_completion({"fen": fen, "legal_moves": moves}, "a2a3 a2a4<end_of_turn> h1h8")
    assert result["precision"] == 1.0
    assert result["recall"] == 0.1
    assert result["illegal_moves"] == []
    assert not result["set_exact"]


def test_tracked_v2_evidence_is_complete_and_hash_linked():
    root = Path(__file__).resolve().parents[1] / "results" / "v2"
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    quality = {r["name"]: r for r in evidence["quality"]}
    assert len(quality) == 27
    assert all(r["status"] == "completed" for r in quality.values())
    assert all(r["quality"]["overall"]["samples"] == 9872 for r in quality.values())
    speed = pd.read_csv(root / "performance_repetitions.csv").to_dict("records")
    names = sorted({s["model"] for s in speed})
    assert len(names) == 9
    validate_scenarios(speed, names)
    assert sum(s["completed"] for s in speed) == 110592
    assert evidence["audit"]["rescored_predictions"] == 266544
    assert evidence["audit"]["checkpoint_hash_matches"] == 9
    for item in evidence["speed"]:
        provenance = item["checkpoint_provenance"]
        expected = quality[item["name"]]["checkpoint"]["checkpoint_content_sha256"]
        assert provenance["serving_checkpoint_content_sha256"] == expected
        assert item["quality_evidence"]["evaluation_contract_sha256"] == quality[item["name"]]["evaluation_contract_sha256"]
