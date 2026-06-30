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

# vLLM serving venv (the tau2 Gemma-4 path). vLLM 0.24.0 hard-pins torch 2.11.
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"
VLLM_PYTHON="${VLLM_PYTHON:-3.12.13}"
# torch wheel backend. MUST match the cluster GPU's CUDA. Default cu130 = Vulcan
# L40S (driver 595 / CUDA 13.2; `module avail cuda` shows up to 13.2). Do NOT use
# 'auto' here: setup runs on the GPU-less LOGIN node, where uv detects no CUDA and
# silently installs a CPU-only torch (no libtorch_cuda.so -> `import vllm` dies with
# "libtorch_cuda.so: cannot open shared object file" at run time on the GPU node).
# Other cluster: set to its CUDA, e.g. TORCH_BACKEND=cu126 bash vulcan/setup.sh
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"

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

# vLLM venv: built differently from the others (special wheel backend + a
# uv-MANAGED python). System python often lacks dev headers (Python.h), which
# breaks Triton's runtime JIT used by vLLM's attention + sampler -> managed python
# ships them. Then install the actsparse plugin editable (registers the
# vllm.general_plugins entry point; it imports src.actsparse at runtime via
# PYTHONPATH=repo root, set by run_vllm.sh).
build_vllm_venv() {
    if [ "$FORCE" = "0" ] && [ -x "$VLLM_VENV/bin/python" ]; then
        echo "[setup] vllm venv -> $VLLM_VENV (exists, skipping; FORCE=1 to rebuild)"
        return
    fi
    echo "[setup] vllm venv -> $VLLM_VENV (vllm $VLLM_VERSION, torch-backend=$TORCH_BACKEND)"
    uv venv --python "$VLLM_PYTHON" --managed-python "$VLLM_VENV"
    uv pip install --python "$VLLM_VENV/bin/python" \
        "vllm==$VLLM_VERSION" --torch-backend="$TORCH_BACKEND"
    uv pip install --python "$VLLM_VENV/bin/python" -e vllm_actsparse_plugin
    # sanity. The login node has no GPU/CUDA, so we canNOT truly `import vllm` here
    # (it would fail on libcuda/libcudart regardless of the wheel). Only confirm the
    # packages are INSTALLED via find_spec. The real vLLM CUDA import is validated at
    # run time on a GPU node, where run_vllm.sh puts the wheel's bundled CUDA libs
    # (site-packages/nvidia/*/lib) on LD_LIBRARY_PATH; the node driver must support
    # the wheel's CUDA (Vulcan L40S = driver 595 / CUDA 13.2, wheel = cu13).
    "$VLLM_VENV/bin/python" - <<'PY'
import importlib.util as u
assert u.find_spec("vllm"), "vllm not installed"
assert u.find_spec("vllm.model_executor.models.gemma4"), "vllm gemma4 model missing"
import vllm_actsparse; assert hasattr(vllm_actsparse, "register")
print("[setup] vllm venv: package + gemma4 model + actsparse plugin INSTALLED")
print("[setup] NOTE: validate `import vllm` on a GPU node (login node has no CUDA).")
PY
}
build_vllm_venv

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
