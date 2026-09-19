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
Each linked revision includes `LICENSE`, `NOTICE`, `SHA256SUMS`, and `artifact_manifest.json`; the model or dataset card also declares the applicable license identifier.

## Hugging Face Repositories

### Dataset

| Artifact | Repository | Immutable revision |
|---|---|---|
| Processed legal-move dataset and splits | [lichess-fen-legal-moves](https://huggingface.co/datasets/ProttiProtto/lichess-fen-legal-moves) | [14c1ba8](https://huggingface.co/datasets/ProttiProtto/lichess-fen-legal-moves/tree/14c1ba80dc090c5842b6fba114b9f3bbae1b8077) |

### LoRA Adapters

| Model | Rank 16 | Rank 32 |
|---|---|---|
| Gemma 3 270M | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-lora-r16) / [c4f464e](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-lora-r16/tree/c4f464e62bbdc6f91daf91fb26fa759294620faa) | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-lora-r32) / [1e471c6](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-lora-r32/tree/1e471c6842cf1d9795496efe81758cede2e69f18) |
| Gemma 4 E2B | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-lora-r16) / [4476967](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-lora-r16/tree/4476967fbd8c7578db0951fed14fd2d126a0e432) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-lora-r32) / [c0b0c7b](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-lora-r32/tree/c0b0c7bb618c82c4d0f81b0efa9e389b1f4465d8) |
| Gemma 4 E4B | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-lora-r16) / [204f377](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-lora-r16/tree/204f377250835deceaae8977dad911e45bbc7801) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-lora-r32) / [d5c0a33](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-lora-r32/tree/d5c0a33deaeb8f14d573d86af5b70a3f50adb800) |

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
| Gemma 3 270M | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-bf16) / [0adb99b](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-bf16/tree/0adb99b1ece1ed9133946ed90d7b83e435dc3364) | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-fp8) / [b3307c0](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-fp8/tree/b3307c029e3a7d29c480a3424a40bd5062dcf044) | [repository](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-nvfp4) / [973635e](https://huggingface.co/ProttiProtto/gemma-3-270m-chess-legal-moves-r32-nvfp4/tree/973635ee6bc2fea45038f8e3475304bebb63cadc) |
| Gemma 4 E2B | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-bf16) / [7eab247](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-bf16/tree/7eab2472d7e268e135a345fe3ccb0578635090f5) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-fp8) / [9dfc556](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-fp8/tree/9dfc556eadb2628f46265725b99357ee8e4bf400) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-nvfp4) / [41607a8](https://huggingface.co/ProttiProtto/gemma-4-e2b-chess-legal-moves-r32-nvfp4/tree/41607a826eb74c69867a16d5f2229ae1944e4819) |
| Gemma 4 E4B | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-bf16) / [e40a1dd](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-bf16/tree/e40a1ddd806703d4b0318950c632d734ceea5517) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-fp8) / [3a57fa5](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-fp8/tree/3a57fa5b86b818f20324dcb7d621cb08eae508d7) | [repository](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-nvfp4) / [0329e73](https://huggingface.co/ProttiProtto/gemma-4-e4b-chess-legal-moves-r32-nvfp4/tree/0329e73e2bff63e577853ed902664636e4a7a789) |

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
