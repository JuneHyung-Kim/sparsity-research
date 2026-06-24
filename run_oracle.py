#!/usr/bin/env python
"""PPL vs activation-sparsity for the oracle ceiling + random floor.

For each (method, sparsity) we swap the per-token FFN masker and measure
WikiText-2 PPL. Weights are never modified, so the model is loaded once and
nothing is restored between points. The dense (sparsity=0) point is shared.

  .venv/bin/python run_oracle.py \
      --model NousResearch/Llama-2-7b-hf \
      --methods random oracle_gate oracle_contrib \
      --sparsities 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
      --out results/oracle_llama2-7b.csv
"""
import argparse
import csv
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.actsparse import METHODS, build_masker, install_sparse_mlps
from src.data import get_wikitext2_testenc
from src.eval_ppl import eval_ppl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--sparsities", nargs="+", type=float,
                   default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/oracle.csv")
    args = p.parse_args()

    for m in args.methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method '{m}'. known: {METHODS}")

    dtype = getattr(torch, args.dtype)
    print(f"[load] {args.model} ({args.dtype})")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True).to(args.device)
    model.eval()
    model.config.use_cache = False

    testenc = get_wikitext2_testenc(tok)
    ctrl, _ = install_sparse_mlps(model)

    rows = []
    sparsities = sorted(set(args.sparsities))

    def record(method, sparsity, ppl, secs):
        rows.append({"method": method, "sparsity": round(sparsity, 4),
                     "ppl": ppl, "seconds": round(secs, 1)})
        print(f"  {method:15s} sparsity={sparsity:.2f}  ppl={ppl:.4f}  [{secs:.0f}s]")
        _write_csv(args.out, rows)

    if 0 in sparsities:
        ctrl["masker"] = None
        t0 = time.time()
        ppl = eval_ppl(model, testenc, args.seqlen, args.device)
        record("dense", 0.0, ppl, time.time() - t0)

    for method in args.methods:
        for sp in sparsities:
            if sp == 0:
                continue
            ctrl["masker"] = build_masker(method, sp, args.device, seed=args.seed)
            t0 = time.time()
            ppl = eval_ppl(model, testenc, args.seqlen, args.device)
            record(method, sp, ppl, time.time() - t0)
    ctrl["masker"] = None

    print(f"[done] wrote {args.out}  ({len(rows)} rows)")


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "sparsity", "ppl", "seconds"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
