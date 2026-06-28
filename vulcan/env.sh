# Source me FIRST on the Vulcan/DRAC cluster (login or compute node):
#   source vulcan/env.sh
#
# Puts this repo's venvs, HF cache, and outputs under the project space
# ($MYPROJ). All paths are overridable: export them before sourcing.

# Project allocation (~/projects is the /project filesystem).
export PROJ_BASE="${PROJ_BASE:-$HOME/projects/aip-nanditav/sankeert}"
# Our own directory under it (everything this repo writes goes here).
export MYPROJ="${MYPROJ:-$PROJ_BASE/jhkim}"

export TMPDIR="${TMPDIR:-$MYPROJ/tmp}"

# HuggingFace cache, shared between login (download) and compute (offline read).
# Re-downloadable, so override HF_HOME before sourcing if you want it elsewhere.
export HF_HOME="${HF_HOME:-$MYPROJ/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=0

# uv toolchain (kept under the project space).
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$MYPROJ/uv/bin}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$MYPROJ/uv/cache}"
export PATH="$UV_INSTALL_DIR:$PATH"

# This repo's two venvs. run_bfcl.sh reads $VENV (masker + HF server) and
# $BFCL_VENV (the bfcl CLI).
export VENV="${VENV:-$MYPROJ/sparsity-research-venv}"
export BFCL_VENV="${BFCL_VENV:-$MYPROJ/sparsity-research-bfcl-venv}"

# Where bfcl_run_s* roots and the figure/CSV land.
export ACTIVATION_SPARSITY_OUTPUTS="${ACTIVATION_SPARSITY_OUTPUTS:-$MYPROJ/outputs/sparsity-research}"

# SLURM account (DRAC/Alliance).
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-aip-nanditav}"
export SALLOC_ACCOUNT="$SLURM_ACCOUNT"
export SBATCH_ACCOUNT="$SLURM_ACCOUNT"

mkdir -p "$TMPDIR" "$HF_HOME" "$UV_CACHE_DIR" "$ACTIVATION_SPARSITY_OUTPUTS"
