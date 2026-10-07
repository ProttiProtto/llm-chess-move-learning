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


def test_natural_fen_timing_images_carry_the_release_warning():
    warning = "Archived/provisional: post-EOT generation may affect timing."
    for name in (
        "throughput_scaling",
        "parallelism_portfolio_summary",
        "quality_vs_throughput_natural_fen",
        "latency_natural_fen",
        "request_throughput",
    ):
        svg = (ROOT / "assets" / f"{name}.svg").read_text(encoding="utf-8")
        assert warning in svg, f"Missing in-image caveat: {name}"
        assert "Natural stopping" not in svg

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    findings = readme.split("## Main Findings", 1)[1].split("## Pipeline", 1)[0]
    assert "assets/check_status.png" in findings
    assert "assets/quality_vs_throughput_natural_fen.png" not in findings
