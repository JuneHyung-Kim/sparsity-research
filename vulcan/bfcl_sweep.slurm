#!/usr/bin/env bash
#SBATCH --job-name=bfcl-vllm
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --time=24:00:00
#SBATCH --output=vulcan/logs/%x-%j.out
#
# BFCL accuracy-vs-sparsity sweep on Gemma-4-12B *served by vLLM*, one L40S, offline.
# Replaces the old HF server.py path (run.sh) with benchmarks/bfcl/sweep.sh ->
# run_vllm.sh + the actsparse plugin. BFCL has ONE role (the model under test), so
# dense AND sparse points each use ONE L40S -- no 2-GPU case (unlike tau2).
#
# Run vulcan/setup.sh on the login node FIRST (venvs + .venv-vllm + actsparse plugin +
# model download). Then submit. DEFAULT = the plan's multi_turn+memory subset: bf16,
# THINK=1, 50 RANDOM cases/subcat (seed 0), oracle_gate (the only reported oracle),
# round grid dense + 0.5/0.7/0.8/0.9:
#   sbatch vulcan/bfcl_sweep.slurm
#
# single_turn is a SEPARATE submission (THINK=0, and it is run in full, not subset):
#   sbatch --export=ALL,CATS=single_turn,THINK=0,NCASES=,SUBSET_SEED=,SWEEP_BASE=$PWD/bfcl_runs/single_turn_bf16 \
#          vulcan/bfcl_sweep.slurm
#
# Knobs (-> sweep.sh/run_vllm.sh): CATS, METHODS (oracle_gate), SPARSITIES (sparse
# points; sweep.sh adds dense), THINK, NCASES + SUBSET_SEED (random subset; empty
# SUBSET_SEED -> first-N, empty NCASES -> full set), QUANT (bf16), MAXLEN (32768;
# reasoning + multi_turn_long_context accumulate), GPU_UTIL, NUM_THREADS, SWEEP_BASE,
# SKIP_EXISTING (1 = resumable; safe to resubmit), SKIP_PLOT.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source vulcan/env.sh

# Compute nodes have no internet: read weights from the HF cache, no telemetry.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

export MODEL="${MODEL:-google/gemma-4-12B-it}"
export METHODS="${METHODS:-oracle_gate}"          # the fixed reported oracle (no gateonly)
export CATS="${CATS:-multi_turn,memory}"          # THINK=1 tiers; single_turn -> separate job
export SPARSITIES="${SPARSITIES:-0.5 0.7 0.8 0.9}"  # sparse points; sweep.sh runs dense too
export QUANT="${QUANT:-bf16}"                     # bf16 on the L40S (fp8 = the 24GB dev box)
export MAXLEN="${MAXLEN:-32768}"                  # thinking + multi_turn_long_context need it
export GPU_UTIL="${GPU_UTIL:-0.9}"
export THINK="${THINK:-1}"                        # multi_turn/memory reason; single_turn=0
export NCASES="${NCASES:-50}"                     # random cases per subcategory
export SUBSET_SEED="${SUBSET_SEED:-0}"            # reproducible random subset
export NUM_THREADS="${NUM_THREADS:-24}"
export SWEEP_BASE="${SWEEP_BASE:-$PWD/bfcl_runs/mt_mem_bf16}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"        # resume: skip already-scored points
# Unique server port per job so concurrent jobs on one node don't collide on 8003.
export PORT="${PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 40000))}"

echo "[bfcl-vllm] model=$MODEL quant=$QUANT methods='$METHODS' cats='$CATS' sparsities='$SPARSITIES'"
echo "[bfcl-vllm] think=$THINK ncases=$NCASES seed=$SUBSET_SEED maxlen=$MAXLEN -> $SWEEP_BASE"

# sweep.sh runs dense (s=0, plugin no-op) + METHODS x SPARSITIES, each as its own vLLM
# engine (the masker is fixed per process), into $SWEEP_BASE/{dense,<method>}/.
./benchmarks/bfcl/sweep.sh

# Aggregate (method x sparsity) -> CSV + per-group/per-category tables + plot (results/).
# SKIP_PLOT=1 when chopping the sweep across jobs, then run sweep_plot.py once by hand.
if [ "${SKIP_PLOT:-0}" = "0" ]; then
    OUT="results/bfcl_gemma4/$(basename "$SWEEP_BASE")"
    "$VENV/bin/python" benchmarks/bfcl/sweep_plot.py --base "$SWEEP_BASE" --out-dir "$OUT" \
        --title "BFCL vs oracle_gate sparsity (Gemma-4-12B $QUANT, THINK=$THINK)"
    # Drop BFCL's transient HF lock dirs (cruft; scores/results kept).
    find "$SWEEP_BASE"/*/bfcl_run_s* -type d -name .file_locks -exec rm -rf {} + 2>/dev/null || true
    echo "[bfcl-vllm] done -> $OUT"
else
    echo "[bfcl-vllm] done (SKIP_PLOT=1; aggregate later with sweep_plot.py --base $SWEEP_BASE)"
fi
