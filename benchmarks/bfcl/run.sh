#!/usr/bin/env bash
# Run BFCL on Qwen3-8B with a given activation-sparsity setting, end to end.
#
# Serves the HF model (with the per-token FFN masker from src/actsparse.py) via
# benchmarks/bfcl/server.py, points BFCL at it with --skip-server-setup, generates
# + scores one or more categories, then tears the server down. Results/scores land
# in a per-sparsity project root so dense and sparse runs don't clobber each other.
#
# Usage:
#   ./benchmarks/bfcl/run.sh <sparsity> [method] [categories]
# Examples:
#   ./benchmarks/bfcl/run.sh 0.0                              # dense baseline
#   ./benchmarks/bfcl/run.sh 0.5 oracle_gate simple_python,irrelevance
#   ./benchmarks/bfcl/run.sh 0.7 oracle_contrib non_live
#
# Env knobs:
#   PORT (1053), MODEL (Qwen/Qwen3-8B), FC_MODEL (=${MODEL}-FC, the BFCL handler
#   id), VENV (.venv), BFCL_VENV (.venv-bfcl), NUM_THREADS/BATCH (throughput),
#   THINK (1 = enable Qwen3 reasoning; needed for multi_turn), MAX_NEW (1024).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root (this script lives in benchmarks/bfcl/)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # so server.py finds src.actsparse

SPARSITY="${1:?usage: run.sh <sparsity> [method] [categories]}"
METHOD="${2:-oracle_gate}"
CATS="${3:-single_turn,multi_turn}"
PORT="${PORT:-1053}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
FC_MODEL="${FC_MODEL:-${MODEL}-FC}"   # BFCL handler id (must be registered)
VENV="${VENV:-.venv}"
BFCL_VENV="${BFCL_VENV:-.venv-bfcl}"
# Throughput: the server micro-batches concurrent requests, so BFCL must keep
# many in flight. NUM_THREADS (BFCL's parallel requests) should ~match the
# server's --batch; lower BATCH if the GPU OOMs (bigger model / thinking on).
NUM_THREADS="${NUM_THREADS:-24}"
BATCH="${BATCH:-24}"
# Qwen3 reasoning: off by default (fast, single-turn). multi_turn needs it on,
# with a higher token cap for the chain-of-thought.
THINK="${THINK:-0}"
MAX_NEW="${MAX_NEW:-1024}"
THINK_FLAG=""
[ "$THINK" != "0" ] && THINK_FLAG="--think"

# tag like "s00" / "s50" / "s70" for the project-root dir
TAG="s$(printf '%02d' "$("$VENV/bin/python" -c "print(round(float('$SPARSITY')*100))")")"
# Outputs default to the repo dir, but on a cluster point BFCL_RUN_BASE at
# scratch (home has tight quotas). benchmarks/bfcl/plot.py --runs-dir reads here.
BASE="${BFCL_RUN_BASE:-$PWD}"
mkdir -p "$BASE"
ROOT="$BASE/bfcl_run_${TAG}"
SNAP="$("$VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))")"

echo "[run_bfcl] model=$MODEL fc=$FC_MODEL sparsity=$SPARSITY method=$METHOD cats=$CATS think=$THINK batch=$BATCH max_new=$MAX_NEW root=$ROOT"

# BFCL skips generation when result files already exist (its resume feature).
# That silently re-scores stale results after a code change, so FRESH=1 wipes
# this sparsity's prior generations/scores to force a clean re-run.
if [ "${FRESH:-0}" != "0" ]; then
    echo "[run_bfcl] FRESH=1 -> clearing $ROOT/{result,score}"
    rm -rf "$ROOT/result" "$ROOT/score"
fi

# NCASES=N -> run only the first N test cases per category (a fast subset, for
# baseline/iteration). Writes BFCL's test_case_ids_to_generate.json (ids are
# "<category>_<i>", contiguous from 0) and adds --run-ids. Use explicit category
# names in CATS (not group aliases like single_turn) so the ids resolve.
RUN_IDS_FLAG=""
PARTIAL_FLAG=""
if [ -n "${NCASES:-}" ]; then
    mkdir -p "$ROOT"
    "$VENV/bin/python" - "$ROOT" "$CATS" "$NCASES" <<'PY'
import json, os, sys
root, cats, n = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
ids = {c: [f"{c}_{i}" for i in range(n)] for c in cats}
json.dump(ids, open(os.path.join(root, "test_case_ids_to_generate.json"), "w"), indent=2)
PY
    RUN_IDS_FLAG="--run-ids"
    # evaluate refuses a partial result set unless told it's intentional.
    PARTIAL_FLAG="--partial-eval"
    echo "[run_bfcl] NCASES=$NCASES -> first $NCASES ids/category via --run-ids"
fi

# Reduce CUDA fragmentation OOMs — multi_turn/thinking grow the KV cache a lot.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- start server, wait for readiness, ensure cleanup ---
"$VENV/bin/python" benchmarks/bfcl/server.py --model "$MODEL" --served-name "$FC_MODEL" \
    --port "$PORT" --method "$METHOD" --sparsity "$SPARSITY" \
    --batch "$BATCH" --max-new "$MAX_NEW" $THINK_FLAG &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; exit 1; fi
    sleep 1
done

# --- generate + evaluate ---
export BFCL_PROJECT_ROOT="$ROOT"
export LOCAL_SERVER_PORT="$PORT"
"$BFCL_VENV/bin/bfcl" generate --model "$FC_MODEL" --test-category "$CATS" \
    --skip-server-setup --local-model-path "$SNAP" --num-threads "$NUM_THREADS" $RUN_IDS_FLAG
"$BFCL_VENV/bin/bfcl" evaluate --model "$FC_MODEL" --test-category "$CATS" $PARTIAL_FLAG

echo "[run_bfcl] scores -> $ROOT/score/"
