#!/usr/bin/env bash
#SBATCH --job-name=bfcl-sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:55:00
#SBATCH --output=vulcan/logs/%x-%j.out
#
# BFCL accuracy-vs-sparsity sweep on one L40S, fully offline.
# Run vulcan/setup.sh on the login node FIRST (venvs + model download).
#
# Submit:
#   sbatch vulcan/bfcl_sweep.slurm
# Override knobs (all optional):
#   sbatch --export=ALL,SPARSITIES="0 0.3 0.5 0.7 0.8",CATS=simple_python,irrelevance,METHOD=oracle_gate \
#          vulcan/bfcl_sweep.slurm
#
# NOTE: --time=02:55:00 fits the default 3h partition (gpubase_bygpu_b1). For a
# longer sweep, also raise the partition, e.g. add `#SBATCH --partition=gpubase_bygpu_b2`
# and bump --time.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source vulcan/env.sh

# Compute nodes have no internet: read the model from the scratch HF cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHOD="${METHOD:-oracle_gate}"
CATS="${CATS:-simple_python,irrelevance}"
SPARSITIES="${SPARSITIES:-0 0.5 0.8}"
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
# Run artifacts (bfcl_run_s*) land inside the repo; plot reads them from here.
export BFCL_RUN_BASE="${BFCL_RUN_BASE:-$PWD}"

echo "[sweep] model=$MODEL method=$METHOD cats=$CATS sparsities='$SPARSITIES'"
echo "[sweep] outputs -> $BFCL_RUN_BASE"

for s in $SPARSITIES; do
    echo "===== sparsity=$s ====="
    ./run_bfcl.sh "$s" "$METHOD" "$CATS"
done

# Gather scores across the sweep into a figure + CSV (committed under results/).
"$VENV/bin/python" plot_bfcl.py \
    --runs-dir "$BFCL_RUN_BASE" \
    --out results/bfcl_acc_vs_sparsity.png \
    --csv results/bfcl_acc_vs_sparsity.csv

echo "[sweep] done -> results/bfcl_acc_vs_sparsity.png"
