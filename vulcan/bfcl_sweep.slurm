#!/usr/bin/env bash
#SBATCH --job-name=bfcl-vllm
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --time=12:00:00
#SBATCH --output=vulcan/logs/%x-%j.out
#
# BFCL accuracy-vs-sparsity sweep on Gemma-4-12B *served by vLLM*, offline, one L40S
# per point (BFCL is single-engine -- dense AND sparse alike; no 2-GPU case like tau2).
# Replaces the old HF server.py/run.sh path with benchmarks/bfcl/{sweep,run_vllm}.sh
# + the actsparse plugin.
#
# SETUP (login node, once per worktree):
#   MODEL=google/gemma-4-12B-it bash vulcan/setup.sh   # builds .venv/.venv-bfcl/.venv-vllm + plugin, downloads weights
#
# THREE MODES (auto-detected):
#   1) PARALLEL job array -- one (sparsity) point per task, run concurrently:
#        sbatch --array=0-4 vulcan/bfcl_sweep.slurm            # POINTS "0 0.5 0.7 0.8 0.9"
#        sbatch --array=0-4%3 ...                              # cap 3 concurrent tasks
#      then aggregate once the array finishes (dependency job):
#        sbatch --dependency=afterok:<arrayJobID> --export=ALL,AGG_ONLY=1 vulcan/bfcl_sweep.slurm
#   2) SEQUENTIAL single job -- dense + all sparsities in one process (no --array):
#        sbatch vulcan/bfcl_sweep.slurm
#   3) AGG_ONLY=1 -- just (re)aggregate an existing SWEEP_BASE into results/.
#
# DEFAULT experiment = the plan's multi_turn+memory subset: bf16, THINK=1, 50 RANDOM
# cases/subcat (seed 0), oracle_gate, points dense+0.5/0.7/0.8/0.9. single_turn is a
# SEPARATE submission (THINK=0, full set):
#   sbatch --array=0-4 --export=ALL,CATS=single_turn,THINK=0,NCASES=,SUBSET_SEED=,SWEEP_BASE=$PWD/bfcl_runs/single_turn_bf16 \
#          vulcan/bfcl_sweep.slurm
#
# Knobs: CATS, METHODS (oracle_gate), POINTS (incl dense 0), THINK, NCASES+SUBSET_SEED
# (empty SUBSET_SEED->first-N, empty NCASES->full), QUANT (bf16), MAXLEN (32768;
# reasoning + multi_turn_long_context accumulate), GPU_UTIL, NUM_THREADS, SWEEP_BASE,
# SKIP_EXISTING (resume), SKIP_PLOT (mode 2).
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
export QUANT="${QUANT:-bf16}"                     # bf16 on the L40S (fp8 = the 24GB dev box)
export MAXLEN="${MAXLEN:-32768}"                  # thinking + multi_turn_long_context need it
export GPU_UTIL="${GPU_UTIL:-0.9}"
export THINK="${THINK:-1}"                        # multi_turn/memory reason; single_turn=0
export NCASES="${NCASES:-50}"                     # random cases per subcategory
export SUBSET_SEED="${SUBSET_SEED:-0}"            # reproducible random subset
export NUM_THREADS="${NUM_THREADS:-24}"
export SWEEP_BASE="${SWEEP_BASE:-$PWD/bfcl_runs/mt_mem_bf16}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"        # resume: skip already-scored points
export PORT="${PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 40000))}"
POINTS="${POINTS:-0 0.5 0.7 0.8 0.9}"             # all points incl dense; array indexes this

OUT="results/bfcl_gemma4/$(basename "$SWEEP_BASE")"
aggregate() {
    "$VENV/bin/python" benchmarks/bfcl/sweep_plot.py --base "$SWEEP_BASE" --out-dir "$OUT" \
        --title "BFCL vs oracle_gate sparsity (Gemma-4-12B $QUANT, THINK=$THINK)"
    find "$SWEEP_BASE"/*/bfcl_run_s* -type d -name .file_locks -exec rm -rf {} + 2>/dev/null || true
}

echo "[bfcl-vllm] model=$MODEL quant=$QUANT methods='$METHODS' cats='$CATS' points='$POINTS'"
echo "[bfcl-vllm] think=$THINK ncases=$NCASES seed=$SUBSET_SEED maxlen=$MAXLEN -> $SWEEP_BASE"

# --- Mode 3: aggregate-only (final step / dependency after an array) ---
if [ "${AGG_ONLY:-0}" != "0" ]; then
    echo "[bfcl-vllm] AGG_ONLY -> $OUT"; aggregate; exit 0
fi

# --- Mode 1: job array -- one (sparsity) point per task, in parallel ---
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -ra PTS <<< "$POINTS"
    s="${PTS[$SLURM_ARRAY_TASK_ID]:?array index $SLURM_ARRAY_TASK_ID out of range for POINTS='$POINTS'}"
    case "$s" in 0|0.0|0.00) dir=dense ;; *) dir="$METHODS" ;; esac   # dense is method-independent
    base="$SWEEP_BASE/$dir"
    tag="s$(awk "BEGIN{printf \"%02d\", int($s*100+0.5)}")"
    if [ "$SKIP_EXISTING" != "0" ] &&
       [ -n "$(find "$base/bfcl_run_$tag/score" -name 'BFCL_v4_*_score.json' 2>/dev/null | head -1)" ]; then
        echo "[bfcl-vllm] task $SLURM_ARRAY_TASK_ID: s=$s scores exist -> skip"; exit 0
    fi
    echo "[bfcl-vllm] task $SLURM_ARRAY_TASK_ID: s=$s method=$METHODS -> $base"
    BFCL_RUN_BASE="$base" FRESH=1 ./benchmarks/bfcl/run_vllm.sh "$s" "$METHODS" "$CATS"
    exit 0
fi

# --- Mode 2: single sequential job -- full sweep (sweep.sh) + aggregate ---
# sweep.sh runs dense itself, then loops the SPARSE points, into $SWEEP_BASE/{dense,<method>}/.
SPARSE=""
for p in $POINTS; do case "$p" in 0|0.0|0.00) ;; *) SPARSE="$SPARSE $p" ;; esac; done
SPARSITIES="$SPARSE" ./benchmarks/bfcl/sweep.sh
[ "${SKIP_PLOT:-0}" = "0" ] && aggregate
echo "[bfcl-vllm] done -> $OUT"
