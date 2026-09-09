# Artifact Publication Guide

The Git repository contains source code, tests, notebooks, documentation, 15 PNG/SVG figure pairs, and compact audited v2 evidence. Large datasets, raw predictions, adapters, and merged checkpoints should be published separately with immutable version tags and SHA-256 hashes.

## Publication Status

| Artifact | Destination | Status |
|---|---|---|
| Source, tests, notebooks, and plots | [GitHub repository](https://github.com/ProttiProtto/llm-chess-move-learning) | Included |
| Audited configurations, hashes, tables and report | [results/v2](results/v2/REPORT.md) | Included; figures can be redrawn without raw ZIPs |
| Raw predictions and benchmark reports | [GitHub Releases](https://github.com/ProttiProtto/llm-chess-move-learning/releases) or a Hugging Face dataset | Publish with the results release |
| Processed SFT, selection, and test splits | Hugging Face dataset | Publish after completing the dataset card and hash manifest |
| LoRA adapters | Hugging Face model repositories | Publish after completing the model cards and license review |
| Merged BF16, FP8, and NVFP4 checkpoints | Hugging Face model repositories | Publish after completing the model cards and license review |
| Original Lichess archive | [Official Lichess open database](https://database.lichess.org/) | Link only; do not duplicate |

Replace the pending statuses with immutable URLs after uploading. Do not add placeholder model URLs to the main README.

## Required Metadata

Every published artifact should record:

- Artifact name, semantic version, creation timestamp, byte size, and SHA-256 hash.
- Exact source checkpoint and, for an adapter, the compatible base-model revision.
- Dataset snapshot, split name, row count, and split/content hashes.
- Precision and quantization format, including whether NVFP4 is native W4A4 or weight-only.
- Minimum hardware and serving-library requirements.
- Prompt, tokenizer, generation, and evaluation contract versions where applicable.
- License, attribution, modification notice, and a link back to this repository.

## Model Repositories

The repository's MIT license applies to original project code only. It does not relicense Gemma base weights, LoRA adapters derived from them, or merged/quantized checkpoints.

Before distributing a Gemma-derived artifact:

1. Verify the exact terms linked by the corresponding Google base-model card. Do not assume Gemma 3 and Gemma 4 use identical terms.
2. Include the applicable agreement, restrictions, attribution, base-model link, and a prominent modification notice.
3. Include any required `NOTICE` file and identify this project as the fine-tuning and conversion source.
4. Document LoRA rank, alpha, training data, selected checkpoint, merge procedure, quantizer, calibration data, and serving requirements.
5. Avoid describing the derivative weights as MIT-licensed.

See the current [Google Gemma terms](https://ai.google.dev/gemma/terms) and the license linked by each model card before release. This checklist documents project practice and is not legal advice.

## Dataset Repository

The experiment used the official Lichess puzzle export snapshot dated **2026-07-05**. Lichess states that its database exports are released under CC0.

The dataset card should include:

- Source URL: `https://database.lichess.org/`
- Source filename: `lichess_db_puzzle.csv.zst`
- Snapshot date: `2026-07-05`
- Original compressed-file SHA-256 hash, recorded from the downloaded file
- Filtering, source-game grouping, split percentages, random seed, and builder commit
- Processed split filenames, row counts, and SHA-256 hashes
- A statement that the processed files are derived from the CC0 Lichess export

## Raw Results

The results release should preserve the unmodified evaluator and benchmark output directories, including configurations, contracts, environment reports, split manifests, checkpoint manifests, raw predictions, repetition-level performance files, consolidated comparisons, and their SHA-256 hashes.

The corrected publication sources are listed with exact filenames, sizes and SHA-256 hashes in [the v2 report](results/v2/REPORT.md#sources-and-reproduction). Publish those quality and parallelism ZIPs together with the six training-metrics ZIPs and executed checkpoint-selection notebook. The supplied quality ZIP does not include worker logs; do not claim that missing logs were archived. The selection table is recoverable from the executed notebook, but standalone corrected selection JSON files were not supplied here.

The tracked `results/v2/evidence.json` and CSVs are sufficient to regenerate figures and the report. They are not replacements for raw predictions or the original dataset in an independent re-audit. External artifact uploads are still pending; the release destination link above is not a claim that those files have been uploaded.

The plotting commands in [`analysis/README.md`](analysis/README.md) consume those raw outputs and regenerate the tracked README figures.
