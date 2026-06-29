#!/usr/bin/env bash
#SBATCH --job-name=tau2-sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --time=12:00:00
#SBATCH --output=vulcan/logs/%x-%j.out
#
# tau2-bench pass^k-vs-sparsity sweep on one L40S, fully offline.
# Run vulcan/setup.sh on the login node FIRST (venvs + tau2 clone + model
# download). tau2 drives BOTH the agent (masked) and the user simulator (dense)
# against our local tau2_server.py, so no external API keys are needed.
#
# Submit:
#   sbatch vulcan/tau2_sweep.slurm
# Override knobs (all optional):
#   sbatch --export=ALL,SPARSITIES="0 0.5 0.8",DOMAINS="retail airline",\
#          METHOD=oracle_gate,TRIALS=1,CONC=8 vulcan/tau2_sweep.slurm
#
# Defaults: oracle_gate (the ceiling masker), retail domain (114 tasks; the
# domain with the most headroom for an 8B agent), pass^1 over the full sparsity
# sweep. retail at concurrency 8 is ~10-15 min/point, so the full sweep is ~1-2h;
# the 12h wall clock leaves margin for adding airline or raising TRIALS (pass^k).
# multi-domain: list DOMAINS to sweep each. Raise TRIALS for pass^2..k.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source vulcan/env.sh

# Compute nodes have no internet: read the model from the scratch HF cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHOD="${METHOD:-oracle_gate}"
DOMAINS="${DOMAINS:-retail}"
SPARSITIES="${SPARSITIES:-0 0.5 0.6 0.7 0.8}"
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export TRIALS="${TRIALS:-1}"
export CONC="${CONC:-8}"
# Run artifacts (tau2_run_s*) land inside the repo; plot reads them from here.
export TAU2_RUN_BASE="${TAU2_RUN_BASE:-$PWD}"

echo "[sweep] model=$MODEL method=$METHOD domains='$DOMAINS' sparsities='$SPARSITIES' trials=$TRIALS conc=$CONC"
echo "[sweep] outputs -> $TAU2_RUN_BASE"

# Each (domain, sparsity) point is independent: a failure is logged and the sweep
# continues, so a long run isn't lost wholesale. Metrics are written per point.
for dom in $DOMAINS; do
    for s in $SPARSITIES; do
        echo "===== domain=$dom sparsity=$s ====="
        ./run_tau2.sh "$s" "$METHOD" "$dom" \
            || echo "[sweep] WARN: domain=$dom sparsity=$s failed; continuing"
    done
done

# Gather metrics across the sweep into a figure + table + CSV (under results/).
"$VENV/bin/python" plot_tau2.py \
    --runs-dir "$TAU2_RUN_BASE" \
    --metric pass_hat_1 \
    --out results/tau2_passk_vs_sparsity.png \
    --csv results/tau2_passk_vs_sparsity.csv

echo "[sweep] done. results -> results/tau2_passk_vs_sparsity.{png,md,csv}"
