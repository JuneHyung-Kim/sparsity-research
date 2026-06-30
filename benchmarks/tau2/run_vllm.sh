#!/usr/bin/env bash
# Run tau2-bench on Gemma-4-12B *served by vLLM*, end to end, at one sparsity point.
#
# This is the fast replacement for the HF-generate server (benchmarks/tau2/run.sh):
# vLLM does continuous batching, and (validated, see the project memory) it handles
# Gemma-4's tool-call + reasoning natively, so NO proxy/parser of ours is needed --
# tau2 talks straight to vLLM's OpenAI server with `--tool-call-parser gemma4
# --reasoning-parser gemma4`.
#
# Two roles, masker-per-process:
#   * sparsity == 0 (dense): ONE vLLM engine, aliased to BOTH model ids
#     (agent + user). The actsparse plugin no-ops at sparsity 0, so this is stock
#     vLLM -- the baseline-reproduction gate.
#   * sparsity  > 0 (sparse): TWO engines, because the masker monkeypatch is
#     process-global. agent engine = vLLM + actsparse plugin (ACTSPARSE_SPARSITY);
#     user engine = stock dense vLLM. tau2's two roles are routed by per-role
#     api_base to the two ports.
#
# Usage:
#   ./benchmarks/tau2/run_vllm.sh <sparsity> [method] [domain]
# Examples:
#   ./benchmarks/tau2/run_vllm.sh 0.0                  # dense baseline, retail
#   ./benchmarks/tau2/run_vllm.sh 0.5 oracle_gate retail
#
# Env knobs:
#   MODEL (google/gemma-4-12B-it), VLLM_VENV (.venv-vllm), TAU2_VENV (.venv-tau2),
#   TAU2_REPO (cloned tau2-bench, has data/), QUANT (fp8 | bf16/none; fp8 for the
#     24GB 4090, bf16 on the L40S), MAXLEN (16384), GPU_UTIL (0.92),
#   PORT (8001 agent), USER_PORT (8002 user-sim, sparse only),
#   TRIALS (1), NTASKS (all; small N for a fast subset), CONC (max-concurrency, 4),
#   MAX_NEW (1024), SEED (300), TIMEOUT (600s per-sim cap, 0=none), MAXSTEPS (tau2's 200),
#   FRESH (1 wipes this point's prior sims), TAU2_RUN_BASE (output base; repo dir).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root (this script lives in benchmarks/tau2/)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # actsparse plugin -> src.actsparse

SPARSITY="${1:?usage: run_vllm.sh <sparsity> [method] [domain]}"
METHOD="${2:-oracle_gate}"
DOMAIN="${3:-retail}"

MODEL="${MODEL:-google/gemma-4-12B-it}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
# May arrive absolute (vulcan/env.sh exports it) or relative (bare default, resolved
# against the repo root we cd'd into). Canonicalize to absolute so PATH/VLLM_BIN are
# never built as "$PWD/<absolute>".
case "$VLLM_VENV" in /*) ;; *) VLLM_VENV="$PWD/$VLLM_VENV" ;; esac
TAU2_VENV="${TAU2_VENV:-.venv-tau2}"
TAU2_REPO="${TAU2_REPO:-$PWD/tau2-bench}"
AGENT_NAME="gemma-4-12b-agent"
USER_NAME="gemma-4-12b-user"
# tau2 hardcodes its NL-assertion / env-interface judge to this OpenAI id
# (tau2 config.py DEFAULT_LLM_NL_ASSERTIONS, no CLI override). We alias it onto the
# DENSE engine so the reward judge runs on local dense Gemma -- a fixed judge across
# all sparsity points. (Offline => can't use the real gpt-4.1; the dense->sparse
# delta uses the same judge, so the signal holds; absolute pass^k may deviate from
# the published 69% which judged with gpt-4.1.)
JUDGE_NAME="gpt-4.1-2025-04-14"

PORT="${PORT:-8001}"
USER_PORT="${USER_PORT:-8002}"
QUANT="${QUANT:-fp8}"
MAXLEN="${MAXLEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"
# GPU pinning. Dense (1 engine) uses AGENT_GPU only. Sparse runs TWO engines; in
# bf16 they don't share one card (2x ~24GB > 46GB L40S), so the dense user-sim goes
# on USER_GPU -- request gpu:l40s:2 for a bf16 sweep. On a single-GPU box only the
# dense gate fits.
AGENT_GPU="${AGENT_GPU:-0}"
USER_GPU="${USER_GPU:-1}"

TRIALS="${TRIALS:-1}"
CONC="${CONC:-4}"
MAX_NEW="${MAX_NEW:-1024}"
SEED="${SEED:-300}"
TIMEOUT="${TIMEOUT:-600}"
MAXSTEPS="${MAXSTEPS:-}"

# fp8 -> --quantization fp8; bf16/none -> let vLLM load native dtype.
QUANT_FLAG=""
case "$QUANT" in
    ""|none|bf16|float16|fp16) : ;;
    *) QUANT_FLAG="--quantization $QUANT" ;;
esac
TIMEOUT_FLAG=""; [ "$TIMEOUT" != "0" ] && TIMEOUT_FLAG="--timeout $TIMEOUT"
MAXSTEPS_FLAG=""; [ -n "$MAXSTEPS" ] && MAXSTEPS_FLAG="--max-steps $MAXSTEPS"
NTASKS_FLAG=""; [ -n "${NTASKS:-}" ] && NTASKS_FLAG="--num-tasks $NTASKS"

# tag like s00 / s50 / s70 for the per-sparsity output dir
TAG="s$(printf '%02d' "$("$VLLM_VENV/bin/python" -c "print(round(float('$SPARSITY')*100))")")"
BASE="${TAU2_RUN_BASE:-$PWD}"
ROOT="$BASE/tau2_run_${TAG}"
mkdir -p "$ROOT"
SAVE_TO="tau2_${TAG}_${DOMAIN}"

export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_REPO/data}"
SIM_DIR="$TAU2_DATA_DIR/simulations/$SAVE_TO"
RESULTS_JSON="$SIM_DIR/results.json"

if [ "${FRESH:-0}" != "0" ]; then
    echo "[run_tau2_vllm] FRESH=1 -> clearing $SIM_DIR"
    rm -rf "$SIM_DIR"
fi

# --- vLLM serving ---
# .venv-vllm/bin on PATH so the ninja shim is found; native triton sampler (the
# flashinfer sampler wants a separate JIT toolchain).
export PATH="$VLLM_VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM's _C_stable_libtorch loads its deps (libtorch_cuda.so/libc10.so from
# torch/lib; libcudart.so.NN/cublas/cudnn/nccl from the cu* wheels' nvidia/*/lib)
# BEFORE torch wires up its own lib search path, so those dirs must be on
# LD_LIBRARY_PATH or import dies with "lib...: cannot open shared object file" (seen
# on the L40S nodes; only the local 4090 had a system CUDA + worked via RPATH). The
# node's driver must support that CUDA: Vulcan L40S = driver 595 / CUDA 13.2, wheel
# = cu13. Self-contained, no `module load` needed.
_VLLM_LIBS="$(ls -d "$VLLM_VENV"/lib/python*/site-packages/torch/lib \
                    "$VLLM_VENV"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)"
[ -n "$_VLLM_LIBS" ] && export LD_LIBRARY_PATH="${_VLLM_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
VLLM_BIN="$VLLM_VENV/bin/vllm"

# start_vllm <port> <sparsity> <gpu> <logfile> <served-name...>
start_vllm() {
    local port="$1" sp="$2" gpu="$3" logf="$4"; shift 4
    CUDA_VISIBLE_DEVICES="$gpu" \
    ACTSPARSE_SPARSITY="$sp" ACTSPARSE_METHOD="$METHOD" \
    "$VLLM_BIN" serve "$MODEL" \
        --served-model-name "$@" \
        --port "$port" \
        $QUANT_FLAG \
        --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' \
        --enforce-eager \
        --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
        --max-model-len "$MAXLEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --tensor-parallel-size 1 \
        > "$logf" 2>&1 &
    echo $!
}

# wait_ready <port> <pid> <logfile>
wait_ready() {
    local port="$1" pid="$2" logf="$3"
    for _ in $(seq 1 360); do        # vLLM load+quantize can take minutes
        if curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[run_tau2_vllm] vLLM on :$port died -- last log lines:"; tail -30 "$logf"; exit 1
        fi
        sleep 5
    done
    echo "[run_tau2_vllm] vLLM on :$port not ready in time"; tail -30 "$logf"; exit 1
}

SERVER_PIDS=()
cleanup() { for p in "${SERVER_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

IS_DENSE="$("$VLLM_VENV/bin/python" -c "print(1 if float('$SPARSITY')<=0 else 0)")"
echo "[run_tau2_vllm] model=$MODEL sparsity=$SPARSITY method=$METHOD domain=$DOMAIN quant=$QUANT trials=$TRIALS conc=$CONC dense=$IS_DENSE root=$ROOT"

if [ "$IS_DENSE" = "1" ]; then
    # One engine, aliased to agent + user + judge ids; stock vLLM (plugin no-ops at 0).
    pid=$(start_vllm "$PORT" 0 "$AGENT_GPU" "$ROOT/vllm_agent.log" "$AGENT_NAME" "$USER_NAME" "$JUDGE_NAME")
    SERVER_PIDS+=("$pid")
    AGENT_BASE="http://localhost:$PORT/v1"; USER_BASE="$AGENT_BASE"
    wait_ready "$PORT" "$pid" "$ROOT/vllm_agent.log"
else
    # agent = masked engine (AGENT_GPU); user-sim + reward judge = dense engine (USER_GPU).
    pid=$(start_vllm "$PORT" "$SPARSITY" "$AGENT_GPU" "$ROOT/vllm_agent.log" "$AGENT_NAME")
    SERVER_PIDS+=("$pid")
    pid2=$(start_vllm "$USER_PORT" 0 "$USER_GPU" "$ROOT/vllm_user.log" "$USER_NAME" "$JUDGE_NAME")
    SERVER_PIDS+=("$pid2")
    AGENT_BASE="http://localhost:$PORT/v1"; USER_BASE="http://localhost:$USER_PORT/v1"
    wait_ready "$PORT" "$pid" "$ROOT/vllm_agent.log"
    wait_ready "$USER_PORT" "$pid2" "$ROOT/vllm_user.log"
fi

# litellm's openai provider base. tau2's NL-assertion judge calls litellm WITHOUT a
# per-call api_base, so the global MUST be a DENSE engine -- the reward judge must
# never be sparsified. USER_BASE is always dense (== agent base only when the whole
# run is dense). agent/user roles carry their own per-call api_base below. NOTE for
# the sparse sweep: litellm's num_retries path can drop a per-call api_base and fall
# back to this global; with global=dense, the only leak is an AGENT *retry* running
# dense (rare, slightly optimistic). The alternative (global=masked agent) would
# sparsify the judge on EVERY point -- strictly worse -- so dense is the right global.
export OPENAI_API_BASE="$USER_BASE"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
AGENT_ARGS="{\"api_base\": \"$AGENT_BASE\", \"api_key\": \"local\", \"temperature\": 0.0, \"max_tokens\": $MAX_NEW}"
USER_ARGS="{\"api_base\": \"$USER_BASE\", \"api_key\": \"local\", \"temperature\": 0.0, \"max_tokens\": $MAX_NEW}"
export TAU2_LOCAL_MODELS="$AGENT_NAME,$USER_NAME,$JUDGE_NAME"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- run the benchmark (tau2 drives both roles against vLLM) ---
"$TAU2_VENV/bin/python" benchmarks/tau2/_tau2_cli.py run \
    --domain "$DOMAIN" \
    --agent-llm "openai/$AGENT_NAME" --agent-llm-args "$AGENT_ARGS" \
    --user-llm  "openai/$USER_NAME"  --user-llm-args  "$USER_ARGS" \
    --num-trials "$TRIALS" --max-concurrency "$CONC" --seed "$SEED" \
    --save-to "$SAVE_TO" --auto-resume $NTASKS_FLAG $TIMEOUT_FLAG $MAXSTEPS_FLAG

# --- score: results.json -> compact metrics (pass^k, avg_reward) ---
"$TAU2_VENV/bin/python" benchmarks/tau2/score.py "$RESULTS_JSON" \
    --domain "$DOMAIN" --sparsity "$SPARSITY" --method "$METHOD" \
    --out "$ROOT/${DOMAIN}.json"

echo "[run_tau2_vllm] metrics -> $ROOT/${DOMAIN}.json"
