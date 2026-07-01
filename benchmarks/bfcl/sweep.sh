#!/usr/bin/env bash
# Drive a BFCL activation-sparsity sweep on Gemma-4-12B via run_vllm.sh.
#
# Runs dense once (s=0 is method-independent -- the masker no-ops), then each
# (method, sparsity) as its own vLLM launch (the masker is baked in at server
# start, so every point needs a fresh engine). Each method gets its own output
# subtree; sweep_plot.py reads the lot and treats dense as the s=0 point for all.
#
#   <SWEEP_BASE>/dense/bfcl_run_s00/...
#   <SWEEP_BASE>/<method>/bfcl_run_s{50,70,85}/...
#
# Usage:  ./benchmarks/bfcl/sweep.sh
# Env knobs (defaults = the T1 local fp8 plan):
#   CATS (single_turn = non_live+live), METHODS ("oracle_gate oracle_gateonly"),
#   SPARSITIES ("0.5 0.7 0.8 0.9"), QUANT (fp8), THINK (0), NUM_THREADS,
#   SWEEP_BASE (output root), FRESH (1 = wipe each point before running),
#   SKIP_EXISTING (1 = resume: skip points that already have scores),
#   NCASES (N/category subset) + SUBSET_SEED (random N/category; reproducible per seed).
# multi_turn/memory example (THINK on, 50 random/subcat):
#   CATS=multi_turn THINK=1 NCASES=50 SUBSET_SEED=0 SWEEP_BASE=bfcl_runs/multi_turn ./sweep.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

CATS="${CATS:-single_turn}"                       # T1 = non_live + live (single-turn AST)
# oracle_gate is the fixed reported oracle for every sparsity sweep (see the
# project memory: oracle_gateonly is not carried as a comparison arm downstream).
METHODS="${METHODS:-oracle_gate}"
# Round sparsity points; the cliff sits high, so resolve it with 0.8 AND 0.9.
SPARSITIES="${SPARSITIES:-0.5 0.7 0.8 0.9}"
QUANT="${QUANT:-fp8}"
THINK="${THINK:-0}"
SWEEP_BASE="${SWEEP_BASE:-$PWD/bfcl_sweep}"
mkdir -p "$SWEEP_BASE"

echo "[sweep] base=$SWEEP_BASE cats=$CATS methods='$METHODS' sparsities='$SPARSITIES' quant=$QUANT think=$THINK"

run_point() {   # <sparsity> <method> <out_base>
    local s="$1" m="$2" base="$3"
    # SKIP_EXISTING=1 makes the sweep resumable: a point already carrying scores
    # is left as-is (run_vllm.sh's TAG = s<round(s*100)>).
    local tag; tag="$(awk "BEGIN{printf \"s%02d\", int($s*100+0.5)}")"
    if [ "${SKIP_EXISTING:-0}" != "0" ] &&
       [ -n "$(find "$base/bfcl_run_$tag/score" -name 'BFCL_v4_*_score.json' 2>/dev/null | head -1)" ]; then
        echo "[sweep] skip (scores exist): s=$s method=$m -> $base/bfcl_run_$tag"
        return 0
    fi
    echo "[sweep] ===== s=$s method=$m -> $base ====="
    BFCL_RUN_BASE="$base" FRESH="${FRESH:-1}" QUANT="$QUANT" THINK="$THINK" \
        NUM_THREADS="${NUM_THREADS:-24}" NCASES="${NCASES:-}" SUBSET_SEED="${SUBSET_SEED:-}" \
        ./benchmarks/bfcl/run_vllm.sh "$s" "$m" "$CATS"
}

# dense (shared s=0 for every method; method arg is irrelevant at s=0)
run_point 0.0 oracle_gate "$SWEEP_BASE/dense"

for m in $METHODS; do
    for s in $SPARSITIES; do
        run_point "$s" "$m" "$SWEEP_BASE/$m"
    done
done

echo "[sweep] DONE -> $SWEEP_BASE"
