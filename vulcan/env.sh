# Source me FIRST on the Vulcan/DRAC cluster (login or compute node):
#   source vulcan/env.sh
#
# Code, venvs, and run outputs stay inside the repo (on the project space).
# Only re-downloadable data (the HF model cache) and the uv toolchain/cache
# live on scratch. All paths are overridable: export them before sourcing.

# --- repo (project space): venvs live alongside the code, outputs too ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VENV="${VENV:-$REPO_ROOT/.venv}"             # masker + HF server + plot
export BFCL_VENV="${BFCL_VENV:-$REPO_ROOT/.venv-bfcl}"   # the bfcl CLI

# --- scratch: data + tooling (re-downloadable / re-installable) ---
export SCRATCH="${SCRATCH:-/scratch/$USER}"
SCRATCH_DATA="${SCRATCH_DATA:-$SCRATCH/jhkim}"
export TMPDIR="${TMPDIR:-$SCRATCH_DATA/tmp}"
# HF model cache, shared between login (download) and compute (offline read).
export HF_HOME="${HF_HOME:-$SCRATCH_DATA/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$SCRATCH_DATA/uv/bin}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_DATA/uv/cache}"
export PATH="$UV_INSTALL_DIR:$PATH"

# SLURM account (DRAC/Alliance).
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-aip-nanditav}"
export SALLOC_ACCOUNT="$SLURM_ACCOUNT"
export SBATCH_ACCOUNT="$SLURM_ACCOUNT"

mkdir -p "$TMPDIR" "$HF_HOME" "$UV_CACHE_DIR"
