#!/usr/bin/env bash
#SBATCH --job-name=tau2-vllm
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --time=12:00:00
#SBATCH --output=vulcan/logs/%x-%j.out
#
# tau2-bench on Gemma-4-12B served by vLLM (benchmarks/tau2/run_vllm.sh), offline.
# Run vulcan/setup.sh on the login node FIRST (it now also builds .venv-vllm + the
# actsparse plugin; MODEL=google/gemma-4-12B-it bash vulcan/setup.sh to predownload
# the weights). vLLM handles Gemma-4 tool-calls + reasoning natively, so no proxy.
#
# DEFAULT = the dense baseline GATE: bf16, full retail (114 tasks), sparsity 0, on
# ONE L40S. This is the number to compare against the published Gemma-4-12B
# tau2 anchor (~69%). Submit:
#   sbatch vulcan/tau2_vllm_sweep.slurm
#
# SPARSITY SWEEP (phase b): each masked point runs TWO bf16 engines (agent=masked,
# user-sim+judge=dense) that do NOT fit on one L40S, so request TWO GPUs and list
# the sparsities:
#   sbatch --gres=gpu:l40s:2 --export=ALL,SPARSITIES="0 0.5 0.7",METHOD=oracle_gate \
#          vulcan/tau2_vllm_sweep.slurm
# (run_vllm.sh pins agent->AGENT_GPU=0, user-sim->USER_GPU=1. The dense point in a
# sweep still uses one engine; the 2-GPU request just sits idle for that point.)
#
# Other knobs: DOMAINS (retail), NTASKS (unset=all; small N for a quick check),
# CONC (8), MAXLEN (32768), GPU_UTIL (0.9), TRIALS (1; raise for pass^k), THINK
# stays off (vLLM template default).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source vulcan/env.sh

# Compute nodes have no internet: read weights from the HF cache, no telemetry.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

export MODEL="${MODEL:-google/gemma-4-12B-it}"
METHOD="${METHOD:-oracle_gate}"
DOMAINS="${DOMAINS:-retail}"
SPARSITIES="${SPARSITIES:-0}"          # gate by default; list values for a sweep
export QUANT="${QUANT:-bf16}"          # bf16 on the L40S (fp8 only for the 24GB dev box)
export MAXLEN="${MAXLEN:-32768}"
export GPU_UTIL="${GPU_UTIL:-0.9}"
export CONC="${CONC:-8}"
export TRIALS="${TRIALS:-1}"
export TAU2_RUN_BASE="${TAU2_RUN_BASE:-$PWD}"

echo "[sweep] model=$MODEL quant=$QUANT method=$METHOD domains='$DOMAINS' sparsities='$SPARSITIES' conc=$CONC maxlen=$MAXLEN"
echo "[sweep] outputs -> $TAU2_RUN_BASE"

# Each (domain, sparsity) point is independent: a failure is logged and the sweep
# continues so a long run isn't lost wholesale. run_vllm.sh launches/tears down its
# own vLLM engine(s) per point (the masker is fixed per process).
for dom in $DOMAINS; do
    for s in $SPARSITIES; do
        echo "===== domain=$dom sparsity=$s ====="
        ./benchmarks/tau2/run_vllm.sh "$s" "$METHOD" "$dom" \
            || echo "[sweep] WARN: domain=$dom sparsity=$s failed; continuing"
    done
done

# Gather metrics across the sweep into a figure + table + CSV (under results/).
"$VENV/bin/python" benchmarks/tau2/plot.py \
    --runs-dir "$TAU2_RUN_BASE" \
    --metric pass_hat_1 \
    --out results/tau2_passk_vs_sparsity.png \
    --csv results/tau2_passk_vs_sparsity.csv

echo "[sweep] done. results -> results/tau2_passk_vs_sparsity.{png,md,csv}"
