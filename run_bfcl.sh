#!/usr/bin/env bash
# Run BFCL on Qwen3-8B with a given activation-sparsity setting, end to end.
#
# Serves the HF model (with the per-token FFN masker from src/actsparse.py) via
# bfcl_server.py, points BFCL at it with --skip-server-setup, generates + scores
# one or more categories, then tears the server down. Results/scores land in a
# per-sparsity project root so dense and sparse runs don't clobber each other.
#
# Usage:
#   ./run_bfcl.sh <sparsity> [method] [categories]
# Examples:
#   ./run_bfcl.sh 0.0                              # dense baseline
#   ./run_bfcl.sh 0.5 oracle_gate simple_python,irrelevance
#   ./run_bfcl.sh 0.7 oracle_contrib non_live
#
# Env knobs: PORT (1053), MODEL (Qwen/Qwen3-8B), VENV (.venv), BFCL_VENV (.venv-bfcl)
set -euo pipefail
cd "$(dirname "$0")"

SPARSITY="${1:?usage: run_bfcl.sh <sparsity> [method] [categories]}"
METHOD="${2:-oracle_gate}"
CATS="${3:-single_turn,multi_turn}"
PORT="${PORT:-1053}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
VENV="${VENV:-.venv}"
BFCL_VENV="${BFCL_VENV:-.venv-bfcl}"
# Throughput: the server micro-batches concurrent requests, so BFCL must keep
# many in flight. NUM_THREADS (BFCL's parallel requests) should ~match the
# server's --batch; lower BATCH if the GPU OOMs on long-prompt batches.
NUM_THREADS="${NUM_THREADS:-24}"
BATCH="${BATCH:-24}"

# tag like "s00" / "s50" / "s70" for the project-root dir
TAG="s$(printf '%02d' "$("$VENV/bin/python" -c "print(round(float('$SPARSITY')*100))")")"
# Outputs default to the repo dir, but on a cluster point BFCL_RUN_BASE at
# scratch (home has tight quotas). plot_bfcl.py --runs-dir reads from here.
BASE="${BFCL_RUN_BASE:-$PWD}"
mkdir -p "$BASE"
ROOT="$BASE/bfcl_run_${TAG}"
SNAP="$("$VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))")"

echo "[run_bfcl] sparsity=$SPARSITY method=$METHOD cats=$CATS root=$ROOT"

# --- start server, wait for readiness, ensure cleanup ---
"$VENV/bin/python" bfcl_server.py --model "$MODEL" --port "$PORT" \
    --method "$METHOD" --sparsity "$SPARSITY" --batch "$BATCH" &
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
"$BFCL_VENV/bin/bfcl" generate --model Qwen/Qwen3-8B-FC --test-category "$CATS" \
    --skip-server-setup --local-model-path "$SNAP" --num-threads "$NUM_THREADS"
"$BFCL_VENV/bin/bfcl" evaluate --model Qwen/Qwen3-8B-FC --test-category "$CATS"

echo "[run_bfcl] scores -> $ROOT/score/"
