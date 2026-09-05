# Rebuilding Results and Figures

The committed plots are generated from the raw outputs produced by the quality evaluator, parallelism benchmark, and SFT telemetry. Raw predictions, model weights, and benchmark archives are deliberately distributed outside Git rather than duplicated here.

Install the analysis dependencies from the repository root:

```bash
python -m pip install -e ".[analysis]"
```

Regenerate all six README figures and normalized CSV tables:

```bash
python -m analysis.build_readme_figures \
  --quality-comparison /path/to/publication_eval/comparison.json \
  --performance-csv /path/to/parallelism/performance_comparison.csv \
  --training-metrics /path/to/run1.zip /path/to/run2.zip /path/to/run3.zip \
  --output-dir assets \
  --tables-dir analysis/generated/tables
```

`--training-metrics` accepts one or more downloaded metrics ZIPs or extracted run directories. Each source must contain the final `step_*_config.json`, `step_*.json`, `step_*_history.jsonl`, and `checkpoint_evaluation.json` files under `metrics/sft/`.

To create only normalized report tables:

```bash
python -m analysis.build_results_tables \
  --quality-comparison /path/to/publication_eval/comparison.json \
  --performance-csv /path/to/parallelism/performance_comparison.csv \
  --training-metrics /path/to/metrics/*.zip
```

Generated intermediate tables are ignored under `analysis/generated/`. The six publication PNGs under `assets/` remain tracked because the root README embeds them.
