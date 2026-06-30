#!/usr/bin/env bash
# Run tau2-bench on Gemma-4-12B with a given activation-sparsity setting, end to end.
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
#   THINK (0; 1 enables Gemma 4 reasoning), MAX_NEW (1024), SEED (300),
#   LOAD_4BIT (0; 1 = nf4 load for a 24GB dev GPU like this box, bf16 on Vulcan),
#   TIMEOUT (600; per-simulation wallclock cap in s, 0 = none — tau2's default of
#     no timeout lets a looping agent/user run to --max-steps and clog the
#     concurrency pipe for an hour+, so we cap it), MAXSTEPS (unset = tau2's 200),
#   TAU2_RUN_BASE (output base; defaults to repo dir).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root (this script lives in benchmarks/tau2/)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # so server.py finds src.actsparse

SPARSITY="${1:?usage: run.sh <sparsity> [method] [domain]}"
METHOD="${2:-oracle_gate}"
DOMAIN="${3:-retail}"
PORT="${PORT:-1055}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
VENV="${VENV:-.venv}"
TAU2_VENV="${TAU2_VENV:-.venv-tau2}"
TAU2_REPO="${TAU2_REPO:-$PWD/tau2-bench}"      # cloned repo (holds data/ domains)
AGENT_NAME="gemma-4-12b-agent"
USER_NAME="gemma-4-12b-user"
TRIALS="${TRIALS:-1}"
CONC="${CONC:-8}"
THINK="${THINK:-0}"
MAX_NEW="${MAX_NEW:-1024}"
SEED="${SEED:-300}"
TIMEOUT="${TIMEOUT:-600}"
MAXSTEPS="${MAXSTEPS:-}"
LOAD_4BIT="${LOAD_4BIT:-0}"            # nf4 load for a 24GB dev GPU (this box); 0 = bf16 (Vulcan)
THINK_FLAG=""
[ "$THINK" != "0" ] && THINK_FLAG="--think"
FOURBIT_FLAG=""
[ "$LOAD_4BIT" != "0" ] && FOURBIT_FLAG="--load-4bit"
# Per-simulation wallclock cap: a stuck (looping / non-terminating) conversation
# would otherwise hold a concurrency slot until --max-steps, starving the rest.
TIMEOUT_FLAG=""
[ "$TIMEOUT" != "0" ] && TIMEOUT_FLAG="--timeout $TIMEOUT"
MAXSTEPS_FLAG=""
[ -n "$MAXSTEPS" ] && MAXSTEPS_FLAG="--max-steps $MAXSTEPS"

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

# Point litellm's openai provider at our local server. We set OPENAI_API_BASE as
# the GLOBAL default (not just per-call --*-llm-args), because litellm's internal
# num_retries path can drop a per-request api_base and fall back to the real
# api.openai.com -> AuthenticationError("Incorrect API key: local"). With the env
# base set, every openai/* call (first try AND retries) targets our server.
API_BASE="http://localhost:$PORT/v1"
export OPENAI_API_BASE="$API_BASE"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
AGENT_ARGS="{\"api_base\": \"$API_BASE\", \"api_key\": \"local\", \"temperature\": 0.0}"
USER_ARGS="{\"api_base\": \"$API_BASE\", \"api_key\": \"local\", \"temperature\": 0.0}"
# Pre-register these model ids with litellm at zero cost (via _tau2_cli.py) so its
# cost lookup doesn't log a 'model isn't mapped' ERROR per call for our endpoint.
export TAU2_LOCAL_MODELS="$AGENT_NAME,$USER_NAME"

NTASKS_FLAG=""
[ -n "${NTASKS:-}" ] && NTASKS_FLAG="--num-tasks $NTASKS"

echo "[run_tau2] model=$MODEL sparsity=$SPARSITY method=$METHOD domain=$DOMAIN trials=$TRIALS conc=$CONC think=$THINK timeout=$TIMEOUT root=$ROOT"

# Reduce CUDA fragmentation OOMs (multi-turn KV cache grows).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- start server, wait for readiness, ensure cleanup ---
"$VENV/bin/python" benchmarks/tau2/server.py --model "$MODEL" \
    --served-name "$AGENT_NAME" --user-served-name "$USER_NAME" \
    --port "$PORT" --method "$METHOD" --sparsity "$SPARSITY" \
    --batch "$CONC" --max-new "$MAX_NEW" $THINK_FLAG $FOURBIT_FLAG &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; exit 1; fi
    sleep 2
done

# --- run the benchmark (tau2 drives both roles against our server) ---
# Invoke tau2 via _tau2_cli.py (registers our local model ids with litellm first,
# then delegates to tau2's CLI) instead of the bare `tau2` entry point.
"$TAU2_VENV/bin/python" benchmarks/tau2/_tau2_cli.py run \
    --domain "$DOMAIN" \
    --agent-llm "openai/$AGENT_NAME" --agent-llm-args "$AGENT_ARGS" \
    --user-llm  "openai/$USER_NAME"  --user-llm-args  "$USER_ARGS" \
    --num-trials "$TRIALS" --max-concurrency "$CONC" --seed "$SEED" \
    --save-to "$SAVE_TO" --auto-resume $NTASKS_FLAG $TIMEOUT_FLAG $MAXSTEPS_FLAG

# --- score: results.json -> compact metrics (pass^k, avg_reward) in our tree ---
"$TAU2_VENV/bin/python" benchmarks/tau2/score.py "$RESULTS_JSON" \
    --domain "$DOMAIN" --sparsity "$SPARSITY" --method "$METHOD" \
    --out "$ROOT/${DOMAIN}.json"

echo "[run_tau2] metrics -> $ROOT/${DOMAIN}.json"
