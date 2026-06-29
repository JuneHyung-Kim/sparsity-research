#!/usr/bin/env bash
# Run on the LOGIN node (has internet, no GPU). One-time setup after cloning:
#   1) build the three uv venvs on scratch from the pinned requirements
#   2) pre-download the model into HF_HOME so compute nodes can run OFFLINE
#
# Idempotent: a venv that already has bin/python is skipped, so re-running after
# adding a new benchmark only builds what's missing (set FORCE=1 to rebuild all).
#
# Usage:
#   bash vulcan/setup.sh                  # Qwen/Qwen3-8B
#   MODEL=Qwen/Qwen3-14B bash vulcan/setup.sh
#   FORCE=1 bash vulcan/setup.sh          # rebuild every venv from scratch
set -euo pipefail
cd "$(dirname "$0")/.."
source vulcan/env.sh

MODEL="${MODEL:-Qwen/Qwen3-8B}"
FORCE="${FORCE:-0}"

command -v uv >/dev/null 2>&1 || {
    echo "uv not found on PATH. Install it on scratch first, e.g.:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "(env.sh expects it at \$UV_INSTALL_DIR=$UV_INSTALL_DIR)"
    exit 1
}

mkdir -p vulcan/logs

# build_venv <venv_dir> <label>; the rest of the args are passed to uv pip install
build_venv() {
    local venv="$1" label="$2"; shift 2
    if [ "$FORCE" = "0" ] && [ -x "$venv/bin/python" ]; then
        echo "[setup] $label venv -> $venv (exists, skipping; FORCE=1 to rebuild)"
        return
    fi
    echo "[setup] $label venv -> $venv"
    uv venv --python 3.12 "$venv"
    uv pip install --python "$venv/bin/python" "$@"
}

build_venv "$VENV"      "research" -r requirements-research.txt
build_venv "$BFCL_VENV" "bfcl"     -r benchmarks/bfcl/requirements.txt

# tau2 needs its upstream repo cloned first (it holds the package AND the domain
# data). Pin a tag with --branch <tag> for strict reproducibility; default branch
# otherwise. tau2 is a CLI that only talks to LLM endpoints over HTTP (no torch),
# so its venv coexists with the research venv. Editable so the `tau2` entry point
# resolves the in-repo data/ (TAU2_DATA_DIR also points there).
if [ ! -d "$TAU2_REPO/.git" ]; then
    echo "[setup] cloning tau2-bench -> $TAU2_REPO"
    git clone --depth 1 https://github.com/sierra-research/tau2-bench.git "$TAU2_REPO"
fi
build_venv "$TAU2_VENV" "tau2" -e "$TAU2_REPO"

echo "[setup] pre-downloading $MODEL into $HF_HOME"
"$VENV/bin/python" - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY

echo
echo "[setup] done. Submit a sweep with:"
echo "  sbatch vulcan/bfcl_sweep.slurm     # function-calling accuracy (BFCL)"
echo "  sbatch vulcan/tau2_sweep.slurm     # agentic tool+user (tau2-bench)"
