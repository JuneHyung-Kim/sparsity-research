#!/usr/bin/env bash
# Run BFCL on Gemma-4-12B *served by vLLM*, end to end, at one sparsity point.
#
# The fast, unified replacement for the HF micro-batching server (benchmarks/bfcl/
# server.py + run.sh): vLLM does continuous batching and BFCL attaches to it with
# --skip-server-setup, POSTing raw prompts to /v1/completions (so NO vLLM-side
# tool-call/reasoning parser is needed -- BFCL renders the prompt and parses the
# reply client-side via benchmarks/bfcl/gemma4_handler.py).
#
# Single engine (BFCL has ONE role -- the model under test; no user-sim/judge):
#   * sparsity == 0 (dense): stock vLLM. The actsparse plugin no-ops at 0, so the
#     dense condition is numerically identical to an un-plugged engine.
#   * sparsity  > 0 (sparse): same one engine + the actsparse plugin
#     (ACTSPARSE_SPARSITY), which patches Gemma4MLP.forward.
#
# Usage:
#   ./benchmarks/bfcl/run_vllm.sh <sparsity> [method] [categories]
# Examples:
#   ./benchmarks/bfcl/run_vllm.sh 0.0                                   # dense baseline
#   ./benchmarks/bfcl/run_vllm.sh 0.5 oracle_gate simple_python,irrelevance
#   THINK=1 ./benchmarks/bfcl/run_vllm.sh 0.0 oracle_gate multi_turn    # reasoning on
#
# Env knobs:
#   PORT (8003), MODEL (google/gemma-4-12B-it), VLLM_VENV (.venv-vllm),
#   BFCL_VENV (.venv-bfcl), QUANT (fp8 | bf16/none; fp8 for the 24GB 4090, bf16 on
#   the L40S), MAXLEN (16384), GPU_UTIL (0.92), GPU (0),
#   NUM_THREADS (BFCL's concurrent requests; vLLM batches them -- 24),
#   THINK (0; 1 => Gemma-4 reasoning. Policy: single_turn OFF, multi_turn ON -- run
#     the two category groups as separate invocations with different THINK),
#   NCASES (first N ids/category, fast subset), FRESH (1 wipes this point's results),
#   BFCL_RUN_BASE (output base; repo dir).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root (this script lives in benchmarks/bfcl/)

SPARSITY="${1:?usage: run_vllm.sh <sparsity> [method] [categories]}"
METHOD="${2:-oracle_gate}"
CATS="${3:-single_turn,multi_turn}"

MODEL="${MODEL:-google/gemma-4-12B-it}"
# BFCL's model-registry id (handler lookup). Usually == MODEL, but some models are
# registered under a distinct id whose weights are MODEL, e.g. Qwen3-8B served for
# BFCL's function-calling handler: MODEL=Qwen/Qwen3-8B, BFCL_MODEL=Qwen/Qwen3-8B-FC.
BFCL_MODEL="${BFCL_MODEL:-$MODEL}"
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"
case "$VLLM_VENV" in /*) ;; *) VLLM_VENV="$PWD/$VLLM_VENV" ;; esac
BFCL_VENV="${BFCL_VENV:-.venv-bfcl}"

PORT="${PORT:-8003}"
QUANT="${QUANT:-fp8}"
MAXLEN="${MAXLEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"
GPU="${GPU:-0}"
NUM_THREADS="${NUM_THREADS:-24}"
# Gemma-4 reasoning: OFF by default (single_turn AST needs no CoT and parses more
# stably). Turn ON for multi_turn, where a low no-think baseline would leave little
# headroom to read sparsity degradation (the Qwen3 lesson). The handler reads
# BFCL_THINK; export THINK=1 to flip it.
THINK="${THINK:-0}"

# fp8 -> --quantization fp8; bf16/none -> native dtype.
QUANT_FLAG=""
case "$QUANT" in
    ""|none|bf16|float16|fp16) : ;;
    *) QUANT_FLAG="--quantization $QUANT" ;;
esac

# tag like s00 / s50 / s70 for the per-sparsity output dir
TAG="s$(printf '%02d' "$("$BFCL_VENV/bin/python" -c "print(round(float('$SPARSITY')*100))")")"
BASE="${BFCL_RUN_BASE:-$PWD}"
mkdir -p "$BASE"
ROOT="$BASE/bfcl_run_${TAG}"
SNAP="$("$BFCL_VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))")"

echo "[run_bfcl_vllm] model=$MODEL sparsity=$SPARSITY method=$METHOD cats=$CATS think=$THINK quant=$QUANT threads=$NUM_THREADS root=$ROOT"

# BFCL resumes by skipping categories whose result files exist; FRESH=1 forces a
# clean re-run of this sparsity point (else a code change silently re-scores stale
# generations).
if [ "${FRESH:-0}" != "0" ]; then
    echo "[run_bfcl_vllm] FRESH=1 -> clearing $ROOT/{result,score}"
    rm -rf "$ROOT/result" "$ROOT/score"
fi

# NCASES=N -> N ids/category (fast subset) via BFCL's test_case_ids_to_generate.json
# + --run-ids. Default = first N (contiguous). SUBSET_SEED=<int> -> a RANDOM N/category
# instead (reproducible per seed), sampled from the category's real ids. CATS may be a
# group alias (e.g. multi_turn) -- it is expanded to concrete categories here.
RUN_IDS_FLAG=""
PARTIAL_FLAG=""
if [ -n "${NCASES:-}" ]; then
    mkdir -p "$ROOT"
    "$BFCL_VENV/bin/python" - "$ROOT" "$CATS" "$NCASES" "${SUBSET_SEED:-}" <<'PY'
import json, os, re, sys, random
from bfcl_eval._llm_response_generation import parse_test_category_argument, load_dataset_entry
root, cats_arg, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = sys.argv[4] if len(sys.argv) > 4 else ""
cats = parse_test_category_argument(                   # expand aliases -> real categories
    [c.strip() for c in cats_arg.split(",") if c.strip()])   # CATS may be comma-joined (multi_turn,memory)
ids = {}
if seed != "":
    rng = random.Random(int(seed))
    for c in cats:
        all_ids = [e["id"] for e in load_dataset_entry(c)]      # dataset order (stable)
        # the SET is fixed by the seed; sort only for a stable/readable file. Ids are
        # not always <cat>_<int> (memory uses <cat>_<int>-<name>-<int>), so sort by a
        # natural key that tolerates non-numeric suffixes.
        ids[c] = sorted(rng.sample(all_ids, min(n, len(all_ids))),
                        key=lambda s: [int(t) if t.isdigit() else t
                                       for t in re.split(r"(\d+)", s)])
else:
    ids = {c: [f"{c}_{i}" for i in range(n)] for c in cats}
json.dump(ids, open(os.path.join(root, "test_case_ids_to_generate.json"), "w"), indent=2)
PY
    RUN_IDS_FLAG="--run-ids"
    PARTIAL_FLAG="--partial-eval"     # evaluate refuses a partial set unless told it's intentional
    echo "[run_bfcl_vllm] NCASES=$NCASES seed='${SUBSET_SEED:-first-N}' -> $NCASES ids/category via --run-ids"
fi

# --- vLLM serving env (self-contained; needed on the L40S nodes) ---
# .venv-vllm/bin on PATH for the ninja shim; native triton sampler (flashinfer
# sampler wants a separate JIT toolchain). The vLLM wheel's bundled CUDA libs must
# be on LD_LIBRARY_PATH or _C_stable_libtorch import dies. PYTHONPATH=repo root so
# the actsparse plugin can `from src.actsparse import build_masker`.
export PATH="$VLLM_VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
_VLLM_LIBS="$(ls -d "$VLLM_VENV"/lib/python*/site-packages/torch/lib \
                    "$VLLM_VENV"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)"
[ -n "$_VLLM_LIBS" ] && export LD_LIBRARY_PATH="${_VLLM_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Serve from the snapshot path so its default served-model-name == the id BFCL
# sends (--local-model-path SNAP -> model=SNAP on /v1/completions). No tool-call/
# reasoning parser: BFCL uses the raw completions endpoint.
LOG="$ROOT/vllm.log"
mkdir -p "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" \
ACTSPARSE_SPARSITY="$SPARSITY" ACTSPARSE_METHOD="$METHOD" \
"$VLLM_VENV/bin/vllm" serve "$SNAP" \
    --port "$PORT" \
    $QUANT_FLAG \
    --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' \
    --enforce-eager \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --tensor-parallel-size 1 \
    > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 360); do        # vLLM load+quantize can take minutes
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run_bfcl_vllm] vLLM died -- last log lines:"; tail -30 "$LOG"; exit 1
    fi
    sleep 5
done

# --- generate + evaluate (BFCL client attaches to our vLLM) ---
export BFCL_PROJECT_ROOT="$ROOT"
export REMOTE_OPENAI_BASE_URL="http://localhost:$PORT/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"
export BFCL_THINK="$THINK"
export BFCL_MAX_CTX="$MAXLEN"

"$BFCL_VENV/bin/python" benchmarks/bfcl/_bfcl_cli.py generate --model "$BFCL_MODEL" \
    --test-category "$CATS" --skip-server-setup --local-model-path "$SNAP" \
    --num-threads "$NUM_THREADS" $RUN_IDS_FLAG
"$BFCL_VENV/bin/python" benchmarks/bfcl/_bfcl_cli.py evaluate --model "$BFCL_MODEL" \
    --test-category "$CATS" $PARTIAL_FLAG

echo "[run_bfcl_vllm] scores -> $ROOT/score/"
