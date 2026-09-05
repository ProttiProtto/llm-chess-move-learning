# Colab Workflows

This package contains the supported implementation for training and evaluating models that enumerate every legal UCI move from a FEN position. The public release intentionally excludes earlier one-move, GRPO, generic-pipeline, and superseded evaluation experiments.

## Supported Notebooks

| Notebook | Purpose |
|---|---|
| `SFT_All_Legal_Moves_Colab.ipynb` | Build the leakage-aware dataset and run resumable LoRA SFT with telemetry and intermediate checkpoints |
| `VLLM_Checkpoint_Export_Colab.ipynb` | Merge a LoRA adapter into BF16 weights and export independent BF16, FP8, and NVFP4 checkpoints |
| `VLLM_Publication_Evaluation_Colab.ipynb` | Run the corrected 27-variant held-out quality evaluation and preserve predictions, hashes, prompt audits, and metrics |
| `VLLM_Parallelism_Benchmark_Colab.ipynb` | Benchmark real-FEN and fixed-length serving workloads across configured concurrency levels |

Run each notebook from top to bottom. They mount Drive, copy this package to local Colab storage for execution, and keep datasets, checkpoints, exports, and reports on Drive.

## Build a Drive Bundle

From the repository root:

```bash
python -m fen_move_colab.package_colab_bundle \
  --output-dir colab_drive_bundle \
  --raw-csv data/raw/lichess_db_puzzle.csv.zst \
  --overwrite
```

Upload `colab_drive_bundle/` directly under `MyDrive`. See `README_UPLOAD_TO_DRIVE.md` for the expected layout.

## Dependency Layers

- `requirements-colab.txt` installs training and dataset dependencies without replacing Colab's CUDA-matched PyTorch build.
- `requirements-colab-fp8.txt` adds TorchAO for optional FP8 and low-precision training paths.
- `requirements-publication-eval.txt` defines the CPU-side publication evaluator dependencies; the notebooks install the pinned vLLM/Torch/CUDA serving stack separately.
- The repository-level `pyproject.toml` defines reproducible local and CI extras such as `.[test]`, `.[training]`, and `.[publication]`.

Hardware-specific PyTorch, TorchAO, vLLM, CUDA, and ModelOpt packages are deliberately not folded into one universal environment. The notebooks pin those stacks where the target GPU and Colab image are known.

## Main Modules

- `build_benchmark_datasets.py` streams and filters the Lichess puzzle export, then creates source-game-disjoint splits.
- `build_all_legal_moves_dataset.py` generates canonical alphabetically sorted legal-move targets with `python-chess`.
- `train_sft_chunk.py` runs resumable LoRA SFT and records exact token, timing, memory, and configuration telemetry.
- `evaluate_sft_checkpoints.py` evaluates intermediate checkpoints after training.
- `publication_eval.py` owns the shared prompt, EOT truncation, parsing, split-audit, and metric contracts.
- `run_publication_evaluation.py` and `offline_vllm_worker.py` run corrected batched quality evaluation.
- `run_parallelism_benchmark.py` runs reproducible online vLLM throughput and latency scenarios.

The root `README.md` documents the research question, experimental design, results, limitations, and full reproduction workflow.
