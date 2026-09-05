#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_LOW_PRECISION="${INSTALL_LOW_PRECISION:-${INSTALL_FP8:-1}}"

if python - "$INSTALL_LOW_PRECISION" <<'PY'
import sys

try:
    import torch
    print(f"PyTorch already available: {torch.__version__}")
except Exception:
    raise SystemExit(1)

if sys.argv[1] == "1":
    version = torch.__version__.split("+", 1)[0]
    parts = tuple(int(part) for part in version.split(".")[:2])
    if parts < (2, 11):
        print("TorchAO 0.17 compiled kernels require PyTorch>=2.11; upgrading PyTorch.")
        raise SystemExit(1)
PY
then
  :
else
  python -m pip install --upgrade pip
  python -m pip install --upgrade --index-url "${PIPELINE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" torch torchvision torchaudio
fi

python -m pip install --upgrade pip
python -m pip install -r "$SCRIPT_DIR/requirements-colab.txt"

if [[ "$INSTALL_LOW_PRECISION" == "1" ]]; then
  python -m pip install -r "$SCRIPT_DIR/requirements-colab-fp8.txt"
  python - <<'PY'
import torch
import torchao

print(f"Low-precision dependency versions: torch={torch.__version__}, torchao={torchao.__version__}")
PY
fi

echo "Colab install complete."
