#!/usr/bin/env bash
#SBATCH --job-name=bfcl-sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --time=2-00:00:00
#SBATCH --output=vulcan/logs/%x-%j.out
#
# BFCL accuracy-vs-sparsity sweep on one L40S, fully offline.
# Run vulcan/setup.sh on the login node FIRST (venvs + model download).
#
# Submit:
#   sbatch vulcan/bfcl_sweep.slurm
# Override knobs (all optional):
#   sbatch --export=ALL,SPARSITIES="0 0.5 0.8",CATS=single_turn,METHOD=oracle_gate \
#          vulcan/bfcl_sweep.slurm
#
# Defaults run every key-free function-calling category (single_turn = non_live
# + live, plus multi_turn) across the full sparsity sweep — a long, multi-hour
# job, hence the 7-day partition (b5) and 2-day wall clock. web_search (needs a
# SerpAPI key), memory (needs agentic backends), and format_sensitivity
# (non-scoring) are excluded. Trim CATS/SPARSITIES for a quick run.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source vulcan/env.sh

# Compute nodes have no internet: read the model from the scratch HF cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHOD="${METHOD:-oracle_gate}"
CATS="${CATS:-single_turn,multi_turn}"
SPARSITIES="${SPARSITIES:-0 0.5 0.6 0.7 0.8}"
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
# Run artifacts (bfcl_run_s*) land inside the repo; plot reads them from here.
export BFCL_RUN_BASE="${BFCL_RUN_BASE:-$PWD}"

echo "[sweep] model=$MODEL method=$METHOD cats=$CATS sparsities='$SPARSITIES'"
echo "[sweep] outputs -> $BFCL_RUN_BASE"

# Each sparsity point is independent: a failure (e.g. one category erroring out)
# is logged and the sweep continues, so a long run isn't lost wholesale. Scores
# are written per (sparsity, category) as they complete.
for s in $SPARSITIES; do
    echo "===== sparsity=$s ====="
    ./run_bfcl.sh "$s" "$METHOD" "$CATS" || echo "[sweep] WARN: sparsity=$s failed; continuing"
done

# Gather scores across the sweep into a figure + CSV (committed under results/).
"$VENV/bin/python" plot_bfcl.py \
    --runs-dir "$BFCL_RUN_BASE" \
    --out results/bfcl_acc_vs_sparsity.png \
    --csv results/bfcl_acc_vs_sparsity.csv

# Remove BFCL's transient HF lock dirs (pure cruft; scores/results are kept).
find "$BFCL_RUN_BASE"/bfcl_run_s* -type d -name .file_locks -exec rm -rf {} + 2>/dev/null || true

echo "[sweep] done. results -> results/bfcl_acc_vs_sparsity.{png,md,csv}"
