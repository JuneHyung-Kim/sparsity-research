#!/usr/bin/env bash
# Run tau2-bench on Qwen3-8B with a given activation-sparsity setting, end to end.
#
# Serves the HF model (per-token FFN masker from src/actsparse.py) via
# benchmarks/tau2/server.py as an OpenAI chat+tools endpoint, points tau2-bench at
# it for BOTH roles -- the AGENT (masked, the policy under test) and the USER
# simulator (dense, part of the environment) -- runs one domain, scores pass^k /
# avg_reward, then tears the server down. Each sparsity point writes to its own dir
# so dense and sparse runs don't clobber each other.
#
# Usage:
#   ./benchmarks/tau2/run.sh <sparsity> [method] [domain]
# Examples:
#   ./benchmarks/tau2/run.sh 0.0                       # dense baseline, retail
#   ./benchmarks/tau2/run.sh 0.5 oracle_gate retail
#   ./benchmarks/tau2/run.sh 0.7 oracle_gate airline
#
# Env knobs:
#   PORT (1055), MODEL (Qwen/Qwen3-8B), VENV (.venv), TAU2_VENV (.venv-tau2),
#   TAU2_REPO (the cloned tau2-bench, has data/), TRIALS (1; raise for pass^k),
#   NTASKS (all; set to a small N for a fast subset), CONC (max-concurrency, 8),
#   THINK (0; 1 enables Qwen3 reasoning), MAX_NEW (1024), SEED (300),
#   TAU2_RUN_BASE (output base; defaults to repo dir).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root (this script lives in benchmarks/tau2/)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # so server.py finds src.actsparse

SPARSITY="${1:?usage: run.sh <sparsity> [method] [domain]}"
METHOD="${2:-oracle_gate}"
DOMAIN="${3:-retail}"
PORT="${PORT:-1055}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
VENV="${VENV:-.venv}"
TAU2_VENV="${TAU2_VENV:-.venv-tau2}"
TAU2_REPO="${TAU2_REPO:-$PWD/tau2-bench}"      # cloned repo (holds data/ domains)
AGENT_NAME="Qwen3-8B-agent"
USER_NAME="Qwen3-8B-user"
TRIALS="${TRIALS:-1}"
CONC="${CONC:-8}"
THINK="${THINK:-0}"
MAX_NEW="${MAX_NEW:-1024}"
SEED="${SEED:-300}"
THINK_FLAG=""
[ "$THINK" != "0" ] && THINK_FLAG="--think"

# tag like s00 / s50 / s70 for the per-sparsity output dir
TAG="s$(printf '%02d' "$("$VENV/bin/python" -c "print(round(float('$SPARSITY')*100))")")"
BASE="${TAU2_RUN_BASE:-$PWD}"
ROOT="$BASE/tau2_run_${TAG}"
mkdir -p "$ROOT"
SAVE_TO="tau2_${TAG}_${DOMAIN}"                # under $TAU2_DATA_DIR/simulations/

# tau2 reads domains and writes simulations under TAU2_DATA_DIR.
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_REPO/data}"
SIM_DIR="$TAU2_DATA_DIR/simulations/$SAVE_TO"
RESULTS_JSON="$SIM_DIR/results.json"

# --auto-resume reuses an existing results file (tau2's resume feature). After a
# code change that silently re-scores stale conversations, so FRESH=1 wipes this
# point's prior simulation to force a clean re-run.
if [ "${FRESH:-0}" != "0" ]; then
    echo "[run_tau2] FRESH=1 -> clearing $SIM_DIR"
    rm -rf "$SIM_DIR"
fi

# llm-args: point litellm's openai provider at our local server (both roles).
API_BASE="http://localhost:$PORT/v1"
AGENT_ARGS="{\"api_base\": \"$API_BASE\", \"api_key\": \"local\", \"temperature\": 0.0}"
USER_ARGS="{\"api_base\": \"$API_BASE\", \"api_key\": \"local\", \"temperature\": 0.0}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"   # belt-and-suspenders for litellm

NTASKS_FLAG=""
[ -n "${NTASKS:-}" ] && NTASKS_FLAG="--num-tasks $NTASKS"

echo "[run_tau2] model=$MODEL sparsity=$SPARSITY method=$METHOD domain=$DOMAIN trials=$TRIALS conc=$CONC think=$THINK root=$ROOT"

# Reduce CUDA fragmentation OOMs (multi-turn KV cache grows).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- start server, wait for readiness, ensure cleanup ---
"$VENV/bin/python" benchmarks/tau2/server.py --model "$MODEL" \
    --served-name "$AGENT_NAME" --user-served-name "$USER_NAME" \
    --port "$PORT" --method "$METHOD" --sparsity "$SPARSITY" \
    --batch "$CONC" --max-new "$MAX_NEW" $THINK_FLAG &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; exit 1; fi
    sleep 2
done

# --- run the benchmark (tau2 drives both roles against our server) ---
"$TAU2_VENV/bin/tau2" run \
    --domain "$DOMAIN" \
    --agent-llm "openai/$AGENT_NAME" --agent-llm-args "$AGENT_ARGS" \
    --user-llm  "openai/$USER_NAME"  --user-llm-args  "$USER_ARGS" \
    --num-trials "$TRIALS" --max-concurrency "$CONC" --seed "$SEED" \
    --save-to "$SAVE_TO" --auto-resume $NTASKS_FLAG

# --- score: results.json -> compact metrics (pass^k, avg_reward) in our tree ---
"$TAU2_VENV/bin/python" benchmarks/tau2/score.py "$RESULTS_JSON" \
    --domain "$DOMAIN" --sparsity "$SPARSITY" --method "$METHOD" \
    --out "$ROOT/${DOMAIN}.json"

echo "[run_tau2] metrics -> $ROOT/${DOMAIN}.json"
