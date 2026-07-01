#!/usr/bin/env bash
#SBATCH --job-name=tau2-plot
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=vulcan/logs/%x-%j.out
#
# Aggregate the tau2 array sweep (tau2_run_s*/<domain>.json) into one figure + table
# + CSV under results/. One line per domain. Submit with a dependency so it runs
# once the compute array is done:
#   sbatch --dependency=afterok:<ARRAY_JOBID> vulcan/tau2_vllm_plot.slurm
# (Or just run the python below by hand -- it only reads JSON, no GPU.)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source vulcan/env.sh
export TAU2_RUN_BASE="${TAU2_RUN_BASE:-$PWD}"

"$VENV/bin/python" benchmarks/tau2/plot.py \
    --runs-dir "$TAU2_RUN_BASE" \
    --metric pass_hat_1 \
    --out results/tau2_passk_vs_sparsity.png \
    --csv results/tau2_passk_vs_sparsity.csv

echo "[plot] results -> results/tau2_passk_vs_sparsity.{png,md,csv}"
