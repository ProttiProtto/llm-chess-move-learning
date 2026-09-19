# Artifact Publication Guide

The Git repository contains source code, tests, notebooks, documentation, 15 PNG/SVG figure pairs, and compact audited v2 evidence. Large datasets, raw predictions, adapters, and merged checkpoints should be published separately with immutable version tags and SHA-256 hashes.

## Publication Status

| Artifact | Destination | Status |
|---|---|---|
| Source, tests, notebooks, and plots | [GitHub repository](https://github.com/ProttiProtto/llm-chess-move-learning) | Included |
| Audited configurations, hashes, tables and report | [results/v2](results/v2/REPORT.md) | Included; figures can be redrawn without raw ZIPs |
| Raw predictions and benchmark reports | [GitHub Releases](https://github.com/ProttiProtto/llm-chess-move-learning/releases) or a Hugging Face dataset | Not uploaded; compact audited evidence is tracked in Git |
| Processed SFT, selection, and test splits | [Hugging Face dataset](https://huggingface.co/datasets/ProttiProtto/lichess-fen-legal-moves) | Uploaded privately and remotely verified |
| LoRA adapters | [Hugging Face repositories](#lora-adapters) | Six uploaded privately and remotely verified |
| Merged BF16, FP8, and NVFP4 checkpoints | [Hugging Face repositories](#rank-32-serving-checkpoints) | Nine uploaded privately and remotely verified |
| Original Lichess archive | [Official Lichess open database](https://database.lichess.org/) | Link only; do not duplicate |

The Hugging Face links below are private release candidates. They are accessible to authorized reviewers and become public download links only after repository visibility is changed deliberately.

## Hugging Face Repositories

### Dataset

| Artifact | Repository | Immutable revision |
|---|---|---|
| Processed legal-move dataset and splits | [lichess-fen-legal-moves](https://huggingface.co/datasets/ProttiProtto/lichess-fen-legal-moves) | [8ca99d1](https://huggingface.co/datasets/ProttiProtto/lichess-fen-legal-moves/tree/8ca99d1ce87740121b919d3e6fec76ce1ae35a07) |

### LoRA Adapters

| Model | Rank 16 | Rank 32 |
|---|---|---|
| Gemma 3 270M | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-lora-r16) / [54e46ad](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-lora-r16/tree/54e46ad92b3304f93fa6c2f053e397ed1bd21877) | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-lora-r32) / [aeda621](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-lora-r32/tree/aeda6210b570528ff41a4307e7c832c1a2d140de) |
| Gemma 4 E2B | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-lora-r16) / [6725b5f](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-lora-r16/tree/6725b5f6105d691fd0ff8f8bf4dc82975934f4bb) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-lora-r32) / [9c4fdc7](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-lora-r32/tree/9c4fdc7abe2c4a3757c90460ffb81114b19f61fd) |
| Gemma 4 E4B | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-lora-r16) / [c1147df](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-lora-r16/tree/c1147df1f22f1fb2ed0fed5d1a6347325b3e0cc5) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-lora-r32) / [db0e62b](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-lora-r32/tree/db0e62b87991f0a4b511a99935bf06bb92756d01) |

Load an adapter with its gated Gemma base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("google/gemma-4-e4b-it")
model = PeftModel.from_pretrained(
    base,
    "ProttiProtto/gemma-4-e4b-chess-legal-moves-lora-r32",
)
```

### Rank-32 Serving Checkpoints

| Model | BF16 | FP8 | NVFP4 |
|---|---|---|---|
| Gemma 3 270M | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-bf16) / [a005008](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-bf16/tree/a005008f9323fdba24025a4eb8f70d19da00e4d3) | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-fp8) / [fa4655d](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-fp8/tree/fa4655d2938eab154643b65d8dbf6aff6eff8e4d) | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-nvfp4) / [355a423](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-nvfp4/tree/355a42378a88557fb72cbabb1675dc5688a912c9) |
| Gemma 4 E2B | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-bf16) / [4b7ef5a](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-bf16/tree/4b7ef5a6b61f71aebf850855472a5e483b5f1a9a) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-fp8) / [572cb45](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-fp8/tree/572cb45bc90f3a91e106b7bf0368effe6c58916e) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-nvfp4) / [a99d273](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-nvfp4/tree/a99d273f3302ba9c18a6251e88f860984421acf5) |
| Gemma 4 E4B | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-bf16) / [d0669da](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-bf16/tree/d0669daa8fae1684a2c5feef292ff09c0cf5df0c) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-fp8) / [273003c](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-fp8/tree/273003c4eb2ec6c0136c4c1cd8581a62eec9019c) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-nvfp4) / [cafd99f](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-nvfp4/tree/cafd99f9da5701e4d33687d300092a4a58eb4c59) |

Serve a merged checkpoint with vLLM after accepting the applicable Gemma terms and installing a compatible GPU stack:

```bash
vllm serve ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-fp8
```

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

The tracked `results/v2/evidence.json` and CSVs are sufficient to regenerate figures and the report. They are not replacements for raw predictions in an independent re-audit. The processed dataset, selected adapters, and rank-32 serving checkpoints are uploaded privately and remotely verified; the raw quality and parallelism archives are still not externally published.

The plotting commands in [`analysis/README.md`](analysis/README.md) consume those raw outputs and regenerate the tracked README figures.
