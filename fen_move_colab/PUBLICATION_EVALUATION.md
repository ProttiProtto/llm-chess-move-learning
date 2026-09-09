# Publication Evaluation

This evaluator fixes the two correctness risks in the exploratory benchmark: it resolves model-specific end-of-turn markers (rejecting unknown-token IDs) and independently truncates saved text at the first marker before parsing UCI moves. Published v2 results are documented in [the report](../results/v2/REPORT.md).

```mermaid
flowchart LR
    A[Lichess puzzle export] --> B[Leakage-aware split]
    B --> C[128-position checkpoint selection]
    B --> D[9,872-position final test]
    E[Base or merged LoRA checkpoint] --> F[BF16 / FP8 / NVFP4 export]
    D --> G[Exact training prompt]
    F --> H[Offline vLLM LLM.generate]
    G --> H
    H --> I[EOT truncation + UCI parser]
    I --> J[python-chess ground truth]
    J --> K[F1 / exact / check-stratified report]
    H --> L[Separate optional serving benchmark]
```

## Prompt audit

Training and evaluation both call `make_all_legal_moves_instruction(fen)` and use the same Gemma turn wrapper. The validation JSONL retains an older one-move `instruction` field, but that field is not sent to the model. The evaluator regenerates prompts from FEN and writes `prompt_audit.json`; therefore this is a matched-prompt evaluation, not a prompt-robustness test.

## Run in Colab

Upload the contents of `fen_move_colab` directly into `MyDrive/colab_drive_bundle`, alongside `runs`, `datasets`, and `vllm_exports`. A nested `fen_move_colab` folder on Drive is not required: the notebook creates the importable package on the local Colab disk. Open `VLLM_Publication_Evaluation_Colab.ipynb` and run its cells in order. Cell 4 executes the equivalent of:

```bash
python -m fen_move_colab.run_publication_evaluation \
  --config fen_move_colab/config_publication_eval.yaml
```

The default config evaluates all 27 base/rank/precision combinations. It excludes the first 128 validation positions used for checkpoint selection, then verifies that the remaining 9,872 positions share neither canonical FENs nor source games with training.

For this release, the quality notebook retains its Colab-tested `subprocess.check_call` launch behavior. Cell output may be quiet; progress remains inspectable in saved reports and worker logs. New logging and preflight conveniences are deferred. The v2 result paths are retained, and no additional GPU run is needed to publish the completed experiments.

## Saved evidence

Every run stores the exact runtime config, dataset and split hashes, checkpoint file hashes, package versions, model/tokenizer provenance, stop strings and any trustworthy stop IDs, generation settings, raw predictions, worker logs, and corrected quality metrics. Quality tables include move-set F1, exact-position accuracy, and separate check/non-check results.

Serving performance is a separate optional benchmark. It should use real FEN prompts with natural EOT stopping plus fixed input/output lengths, and report concurrency, requests/s, output tokens/s, TTFT, and end-to-end latency without blocking quality evaluation.

Where available, memory fields separate checkpoint weight bytes, vLLM model-loading memory, KV-cache reservation, CUDA-graph memory, and whole-GPU NVML peak. The offline v2 archive does not contain every one of these fields. Do not infer missing breakdowns or describe whole-GPU peak as model size.

## Offline correction

Existing prediction archives can be rescored without a GPU:

```bash
python -m fen_move_colab.recompute_publication_metrics \
  --input path/to/benchmark.zip \
  --output-dir corrected_metrics
```

This corrects quality metrics only. Throughput and latency must be rerun because the old benchmark generated unnecessary post-EOT tokens.

## Limitations

The default experiment has one training seed. Report rank comparisons as observations from these runs, not universal conclusions. High move-set F1 is not exact-position accuracy. Check positions remain a small but disproportionately difficult subset, and NVFP4 conclusions apply only to this short-context workload and serving stack.

## Licensing checklist

- Lichess database exports are published under CC0; cite the source and snapshot date.
- Gemma 3 and Gemma 4 may have different model terms. Preserve the exact model ID/revision and follow the license/model card that applies to each downloaded checkpoint or derivative.
- Stockfish is GPLv3. If a binary is redistributed, satisfy the corresponding source and notice obligations.
- The top-level MIT license covers original project code only; it does not replace model or dataset terms. See [ARTIFACTS.md](../ARTIFACTS.md) before distributing derived weights.
