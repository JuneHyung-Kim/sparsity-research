#!/usr/bin/env bash
# Run on the LOGIN node (has internet, no GPU). One-time setup after cloning:
#   1) build both uv venvs on scratch from the pinned requirements
#   2) pre-download the model into HF_HOME so compute nodes can run OFFLINE
#
# Usage:
#   bash vulcan/setup.sh                  # Qwen/Qwen3-8B
#   MODEL=Qwen/Qwen3-14B bash vulcan/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source vulcan/env.sh

MODEL="${MODEL:-Qwen/Qwen3-8B}"

command -v uv >/dev/null 2>&1 || {
    echo "uv not found on PATH. Install it on scratch first, e.g.:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "(env.sh expects it at \$UV_INSTALL_DIR=$UV_INSTALL_DIR)"
    exit 1
}

mkdir -p vulcan/logs

echo "[setup] research venv -> $VENV"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -r requirements-research.txt

echo "[setup] bfcl venv -> $BFCL_VENV (core only, no vllm)"
uv venv --python 3.12 "$BFCL_VENV"
uv pip install --python "$BFCL_VENV/bin/python" -r requirements-bfcl.txt

echo "[setup] pre-downloading $MODEL into $HF_HOME"
"$VENV/bin/python" - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY

echo
echo "[setup] done. Submit the sweep with:"
echo "  sbatch vulcan/bfcl_sweep.slurm"
