import json

import pytest

from analysis.results import load_quality_comparison, parse_variant_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("270m-base-bf16", ("270m", "base", "bf16")),
        ("e2b-r16-fp8", ("e2b", "r16", "fp8")),
        ("e4b_r32_best-nvfp4", ("e4b", "r32", "nvfp4")),
    ],
)
def test_parse_variant_name(name, expected):
    assert parse_variant_name(name) == expected


def test_quality_loader_preserves_contract_and_checkpoint_hashes(tmp_path):
    comparison = {
        "results": [
            {
                "name": "e4b-r32-bf16",
                "evaluation_contract_sha256": "contract-hash",
                "checkpoint": {"checkpoint_content_sha256": "checkpoint-hash"},
                "quality": {
                    "overall": {
                        "samples": 10,
                        "macro_precision": 0.9,
                        "macro_recall": 0.8,
                        "macro_f1": 0.85,
                        "set_exact_rate": 0.5,
                        "illegal_move_rate": 0.1,
                    }
                },
            }
        ]
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison), encoding="utf-8")

    result = load_quality_comparison(path).iloc[0]

    assert result["model"] == "e4b-r32-bf16"
    assert result["evaluation_contract_sha256"] == "contract-hash"
    assert result["checkpoint_content_sha256"] == "checkpoint-hash"
