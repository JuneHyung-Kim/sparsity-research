#!/usr/bin/env bash
#SBATCH --job-name=bfcl-sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --time=48:00:00
#SBATCH --output=vulcan/logs/%x-%j.out
#
# BFCL accuracy-vs-sparsity sweep on one L40S, fully offline.
# Run vulcan/setup.sh on the login node FIRST (venvs + model download).
#
# Submit (one job per sparsity is recommended so each fits the wall; SKIP_PLOT=1
# so the parallel jobs don't race on results/, then plot once at the end):
#   for s in 0 0.5 0.6 0.7 0.8; do
#     SKIP_PLOT=1 SPARSITIES="$s" sbatch --export=ALL vulcan/bfcl_sweep.slurm
#   done
#   # after all finish, on the login node:
#   source vulcan/env.sh && "$VENV/bin/python" benchmarks/bfcl/plot.py --runs-dir "$PWD" \
#     --out results/bfcl_acc_vs_sparsity.png --csv results/bfcl_acc_vs_sparsity.csv
#
# Each sparsity runs single_turn (non_live + live) then multi_turn. They have very
# different prompt lengths -- multi_turn_long_context reaches ~13-17k tokens (and more
# with thinking ON) while single_turn stays <=10k -- so they run at DIFFERENT server
# batch sizes (BATCH_ST/BATCH_MT) to use the L40S (48GB) without OOM. Sizes were chosen
# from vulcan/probe_vram.py; re-probe if the model or token caps change. web_search
# (needs a SerpAPI key), memory (needs agentic backends), and format_sensitivity
# (non-scoring) are excluded.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source vulcan/env.sh

# Compute nodes have no internet: read the model from the scratch HF cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHOD="${METHOD:-oracle_gate}"
SPARSITIES="${SPARSITIES:-0 0.5 0.6 0.7 0.8}"
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
# Run artifacts (bfcl_run_s*) land inside the repo; plot reads them from here.
export BFCL_RUN_BASE="${BFCL_RUN_BASE:-$PWD}"
# Match the official Qwen3-8B (FC) baseline: thinking ON + 4096-token/step cap. The
# earlier sweep used run.sh's old defaults (THINK=0 / MAX_NEW=1024), which floored
# multi_turn (~1-2% vs official ~50%) and ran live ~5-10pp low.
export THINK="${THINK:-1}"
export MAX_NEW="${MAX_NEW:-4096}"
# Unique server port per job (derived from the SLURM job id) so several chopped jobs
# can run concurrently -- even if two land on the same node -- without colliding on
# the default 1053. run.sh passes this to both the server and BFCL's client.
export PORT="${PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 40000))}"
# Per-group server batch (thinking inflates the KV cache; long multi_turn prompts need
# a much smaller batch than short single_turn ones). Set CATS_ST="" or CATS_MT="" to
# run only one group. NUM_THREADS is matched to BATCH so the micro-batch stays full.
# NOTE: ${VAR-default} (no colon) so an explicit empty value (CATS_ST="" / CATS_MT="")
# skips that group -- ${VAR:-default} would wrongly treat "" as unset and re-add it.
CATS_ST="${CATS_ST-single_turn}"
CATS_MT="${CATS_MT-multi_turn}"
BATCH_ST="${BATCH_ST:-8}"    # single_turn: prompts <=10k  -> batch 8 (~41GB on L40S)
BATCH_MT="${BATCH_MT:-4}"    # multi_turn:  prompts ~17-20k -> batch 4 (~39-44GB)

echo "[sweep] model=$MODEL method=$METHOD sparsities='$SPARSITIES' think=$THINK max_new=$MAX_NEW"
echo "[sweep] groups: '$CATS_ST'@batch=$BATCH_ST then '$CATS_MT'@batch=$BATCH_MT -> $BFCL_RUN_BASE"

# Each (sparsity, group) is independent: a failure is logged and the sweep continues,
# so a long run isn't lost wholesale. Scores are written per (sparsity, category) as
# they complete. Single_turn and multi_turn write disjoint category dirs under the same
# bfcl_run_s<NN> root, so running them sequentially here keeps one root per sparsity.
for s in $SPARSITIES; do
    echo "===== sparsity=$s ====="
    if [ -n "$CATS_ST" ]; then
        BATCH="$BATCH_ST" NUM_THREADS="$BATCH_ST" \
            ./benchmarks/bfcl/run.sh "$s" "$METHOD" "$CATS_ST" \
            || echo "[sweep] WARN: sparsity=$s group=$CATS_ST failed; continuing"
    fi
    if [ -n "$CATS_MT" ]; then
        BATCH="$BATCH_MT" NUM_THREADS="$BATCH_MT" \
            ./benchmarks/bfcl/run.sh "$s" "$METHOD" "$CATS_MT" \
            || echo "[sweep] WARN: sparsity=$s group=$CATS_MT failed; continuing"
    fi
done

# Gather scores across the sweep into a figure + CSV (committed under results/).
# SKIP_PLOT=1 when splitting the run across several jobs (one per sparsity): each job
# would otherwise overwrite results/ with a partial table and race the others. Skip
# here and run plot.py once by hand after all jobs finish (see the header).
if [ "${SKIP_PLOT:-0}" = "0" ]; then
    "$VENV/bin/python" benchmarks/bfcl/plot.py \
        --runs-dir "$BFCL_RUN_BASE" \
        --out results/bfcl_acc_vs_sparsity.png \
        --csv results/bfcl_acc_vs_sparsity.csv
    # Remove BFCL's transient HF lock dirs (pure cruft; scores/results are kept).
    # ONLY in the single-job case: when chopped jobs share a root concurrently, one
    # job deleting .file_locks could corrupt another's in-flight writes, so the
    # chopped flow (SKIP_PLOT=1) leaves them and the final manual plot step cleans up.
    find "$BFCL_RUN_BASE"/bfcl_run_s* -type d -name .file_locks -exec rm -rf {} + 2>/dev/null || true
fi

echo "[sweep] done. results -> results/bfcl_acc_vs_sparsity.{png,md,csv}"
