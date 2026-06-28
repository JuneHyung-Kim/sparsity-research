# Source me FIRST on the Vulcan/DRAC cluster (login or compute node):
#   source vulcan/env.sh
#
# Sets scratch-backed caches (home has tight quotas), the uv toolchain path,
# the two venv locations this repo uses, the BFCL output dir, and the SLURM
# account. Everything is overridable: `export MYSCRATCH=... ` before sourcing
# (or edit below) if your cluster layout differs.

export SCRATCH="${SCRATCH:-/scratch/$USER}"
# Personal subdir inside the (possibly shared) scratch allocation.
export MYSCRATCH="${MYSCRATCH:-$SCRATCH/jhkim}"
export TMPDIR="${TMPDIR:-$MYSCRATCH/tmp}"

# HuggingFace cache on scratch, shared between login (download) and compute
# (offline read). hf_transfer off (not installed).
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=0

# uv lives on scratch and goes on PATH.
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$SCRATCH/uv/bin}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv/cache}"
export PATH="$UV_INSTALL_DIR:$PATH"

# This repo's two venvs (kept on scratch, not in $HOME). run_bfcl.sh reads
# $VENV (masker + HF server) and $BFCL_VENV (the bfcl CLI).
export VENV="${VENV:-$MYSCRATCH/sparsity-research-venv}"
export BFCL_VENV="${BFCL_VENV:-$MYSCRATCH/sparsity-research-bfcl-venv}"

# Where bfcl_run_s* roots and the figure/CSV land (off home).
export ACTIVATION_SPARSITY_OUTPUTS="${ACTIVATION_SPARSITY_OUTPUTS:-$MYSCRATCH/outputs/sparsity-research}"

# SLURM account (DRAC/Alliance).
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-aip-nanditav}"
export SALLOC_ACCOUNT="$SLURM_ACCOUNT"
export SBATCH_ACCOUNT="$SLURM_ACCOUNT"

mkdir -p "$TMPDIR" "$HF_HOME" "$UV_CACHE_DIR" "$ACTIVATION_SPARSITY_OUTPUTS"
