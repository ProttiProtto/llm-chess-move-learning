from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_historical_natural_fen_performance_is_not_a_final_claim():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "results" / "v2" / "REPORT.md").read_text(encoding="utf-8")

    assert "1.12x BF16 natural-FEN" not in readme
    assert "1.39x BF16 natural-FEN" not in readme
    assert "archived/provisional" in readme
    assert "archived/provisional" in report
    assert "correct-stop rerun" in readme
    assert "correct-stop rerun" in report


def test_v2_quality_provenance_explains_superseded_summaries():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "results" / "v2" / "REPORT.md").read_text(encoding="utf-8")

    for text in (readme, report):
        assert "pre-v2 release-review" in text
        assert "checkpoint/export matrix" in text
        assert "266,544 saved completions" in text
