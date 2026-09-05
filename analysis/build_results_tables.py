from __future__ import annotations

import argparse
from pathlib import Path

from analysis.results import (
    load_performance_comparison,
    load_quality_comparison,
    load_training_metrics,
    write_result_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the publication outputs into report-ready CSV tables.")
    parser.add_argument("--quality-comparison", required=True, type=Path)
    parser.add_argument("--performance-csv", required=True, type=Path)
    parser.add_argument("--training-metrics", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/generated/tables"))
    args = parser.parse_args()

    quality = load_quality_comparison(args.quality_comparison)
    performance_raw, performance_median = load_performance_comparison(args.performance_csv)
    training, history = load_training_metrics(args.training_metrics)
    write_result_tables(args.output_dir, quality, performance_raw, performance_median, training, history)
    print(f"Wrote normalized result tables to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
