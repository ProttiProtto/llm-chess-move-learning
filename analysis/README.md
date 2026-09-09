# Rebuilding the v2 Report

Run from the repository root with Python 3.11 or newer:

```bash
python -m pip install -e ".[analysis]"
python -m analysis.build_v2_report
```

This redraws 15 PNG/SVG figures and regenerates results/v2/REPORT.md and derived CSVs from **tracked, previously audited compact evidence**. It needs no GPU or weights. It does not independently revalidate original predictions.

## Audit Raw Artifacts

Put the nine files in the [report's source table](../results/v2/REPORT.md#sources-and-reproduction) in one directory: v2 quality ZIP, v2 parallelism ZIP, six training-metrics ZIPs, and the executed corrected-selection notebook. Filenames and SHA-256 hashes identify the inputs. External download links are pending in [ARTIFACTS.md](../ARTIFACTS.md).

```bash
python -m analysis.build_v2_report --input-dir /path/to/artifacts
```

The audit reads ZIP members without extracting the archives and checks:

- 27 completed quality variants, 9,872 ordered predictions each, and engine-generated legal ground truth.
- All aggregate quality metrics against a rescore through the project's tested parser.
- Held-out hashes, prompt-match evidence and saved zero-overlap audit. Reconstructing training overlap needs the separate dataset.
- Nine serving checkpoints matched by content hash to quality evidence.
- Exactly 270 unique scenarios, all requests completed, finite positive metrics, and raw fixed input/output lengths of 128.
- Six training histories and corrected selection summaries; historical selection scores are not mixed into final quality.

Invalid/incomplete evidence raises an error rather than silently plotting a subset. Raw audit takes several minutes on CPU.

## Outputs

The results/v2 directory contains compact evidence, the comprehensive report, full quality metrics, check/complexity diagnostics, corrected selections, training history/cost, individual speed repetitions, medians, and quality/throughput joins.

The assets directory contains training loss/cost, all-variant precision/recall/F1/exact, quantization deltas, check status, legal-set size, generation limits, throughput/concurrency, request throughput, p50/p95 latency for both workloads, quality/throughput tradeoffs for all four metrics, and disk-footprint/startup figures. Disk footprint is not VRAM. Repetition bands are observed ranges, not confidence intervals.

No held-out loss curve is synthesized: these evaluations record generation-based quality, not cross-entropy loss.

## General-Purpose Helpers

The analysis.build_readme_figures and analysis.build_results_tables modules still accept arbitrary extracted quality comparisons, performance CSVs, and training metrics. They aggregate supplied data but do **not** perform the full v2 audit. Use analysis.build_v2_report for this publication's authoritative figures. Do not mix v1 and v2 inputs.
