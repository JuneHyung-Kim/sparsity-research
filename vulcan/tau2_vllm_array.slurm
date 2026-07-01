#!/usr/bin/env bash
#SBATCH --job-name=tau2-arr
#SBATCH --array=0-11
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --time=04:00:00
#SBATCH --output=vulcan/logs/%x-%A_%a.out
#
# Fixed-subset oracle_gate sparsity sweep for tau2 on Gemma-4-12B (vLLM), as a JOB
# ARRAY: one array task per (domain, sparsity) POINT (3 domains x 4 sparsities = 12).
# SLURM packs as many points concurrently as the allocation allows, so makespan
# adapts to free GPUs and each point is independent -- a failure re-runs just that
# index (`sbatch --array=<idx> ...`). This beats splitting by domain (which piles the
# slow telecom domain's 4 points into one job) or by sparsity.
#
# Prereqs on the login node (has internet):
#   source vulcan/env.sh
#   MODEL=google/gemma-4-12B-it bash vulcan/setup.sh          # venvs + weights
#   .venv-tau2/bin/python benchmarks/tau2/make_subsets.py     # freeze the 40-task subsets
# Submit:
#   sbatch vulcan/tau2_vllm_array.slurm
#   # cap concurrency (e.g. 6 points at once -> up to 12 GPUs): sbatch --array=0-11%6 ...
#   # single calibration point first (retail dense): sbatch --array=0 vulcan/tau2_vllm_array.slurm
# Aggregate AFTER the array finishes:
#   sbatch --dependency=afterok:<ARRAY_JOBID> vulcan/tau2_vllm_plot.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source vulcan/env.sh

# Compute nodes are offline: read weights from cache, no telemetry.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

# ---- the 3x4 grid: array index -> (domain, sparsity) ----
DOMAINS=(retail airline telecom)
SPARSITIES=(0 0.5 0.7 0.8)
NS=${#SPARSITIES[@]}                                    # 4
i="${SLURM_ARRAY_TASK_ID:?run me as an array job (sbatch vulcan/tau2_vllm_array.slurm)}"
DOM="${DOMAINS[$(( i / NS ))]}"
SP="${SPARSITIES[$(( i % NS ))]}"

# ---- fixed run config (see the sweep plan) ----
export MODEL="${MODEL:-google/gemma-4-12B-it}"
export QUANT="${QUANT:-bf16}"              # bf16 on the L40S
export MAXLEN="${MAXLEN:-32768}"           # MUST exceed max input + MAX_NEW: input+output<=max_model_len,
                                           # else long turns 400 with ContextWindowExceededError. 16384 was too
                                           # tight (retail hits ~12k input + 4096 out > 16384). Paged KV, so a
                                           # bigger window doesn't shrink concurrency for short requests.
export GPU_UTIL="${GPU_UTIL:-0.9}"
export CONC="${CONC:-8}"                   # don't over-crank: high CONC slows per-sim decode -> more timeouts
export TRIALS="${TRIALS:-1}"
export THINK="${THINK:-1}"                 # agent-only Gemma-4 reasoning
export MAX_NEW="${MAX_NEW:-4096}"          # per-turn cap: bounds thinking runaway (single-turn timeout guard)
export TIMEOUT="${TIMEOUT:-1200}"          # per-sim wallclock = tau2's own DEFAULT_MAX_STEPS_SECONDS
export FRESH="${FRESH:-1}"                 # regenerate; never reuse an old full-114 / different-config run
export TAU2_RUN_BASE="${TAU2_RUN_BASE:-$PWD}"
export TASK_IDS_FILE="$PWD/benchmarks/tau2/subsets/${DOM}_40.txt"

# Unique ports per array task so two points that land on the SAME node don't both
# bind 8001/8002. SLURM cgroup-isolates each task's 2 GPUs, so run_vllm's
# AGENT_GPU=0 / USER_GPU=1 still address this task's own pair.
export PORT=$(( 8001 + SLURM_ARRAY_TASK_ID * 10 ))
export USER_PORT=$(( PORT + 1 ))

[ -f "$TASK_IDS_FILE" ] || { echo "missing subset $TASK_IDS_FILE -- run benchmarks/tau2/make_subsets.py on the login node first"; exit 1; }

echo "[array $SLURM_ARRAY_TASK_ID] domain=$DOM sparsity=$SP method=oracle_gate quant=$QUANT think=$THINK max_new=$MAX_NEW timeout=$TIMEOUT conc=$CONC ports=$PORT/$USER_PORT"
echo "[array $SLURM_ARRAY_TASK_ID] subset=$TASK_IDS_FILE run_base=$TAU2_RUN_BASE"

./benchmarks/tau2/run_vllm.sh "$SP" oracle_gate "$DOM"

echo "[array $SLURM_ARRAY_TASK_ID] done: domain=$DOM sparsity=$SP (metrics under $TAU2_RUN_BASE/tau2_run_s*/${DOM}.json)"
