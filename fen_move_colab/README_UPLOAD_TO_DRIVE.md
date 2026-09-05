# Upload to Google Drive

Create the bundle from the repository root:

```bash
python -m fen_move_colab.package_colab_bundle \
  --output-dir colab_drive_bundle \
  --raw-csv data/raw/lichess_db_puzzle.csv.zst \
  --overwrite
```

Upload the generated `colab_drive_bundle` folder directly under `MyDrive` with this exact name:

```text
MyDrive/colab_drive_bundle/
  SFT_All_Legal_Moves_Colab.ipynb
  VLLM_Checkpoint_Export_Colab.ipynb
  VLLM_Publication_Evaluation_Colab.ipynb
  VLLM_Parallelism_Benchmark_Colab.ipynb
  train_sft_chunk.py
  evaluate_sft_checkpoints.py
  run_publication_evaluation.py
  run_parallelism_benchmark.py
  ...other bundled dependencies...
  raw/
    lichess_db_puzzle.csv.zst
  datasets/
    lichess_legal_v1/
  runs/
  vllm_exports/
```

Open the desired notebook from this Drive folder and run its cells in order. Change `DRIVE_BUNDLE_DIR` in the setup cell only if you uploaded the bundle elsewhere.

The training notebook can generate the leakage-aware split and canonical all-legal-moves targets from the bundled compressed Lichess export. It saves checkpoints and telemetry under `runs/<run-name>/`. The export notebook writes merged and quantized checkpoints under `vllm_exports/`; the quality and speed notebooks write self-contained reports under their configured output roots.
