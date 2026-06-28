#!/usr/bin/env python
"""Zero-shot downstream accuracy vs activation-sparsity (companion to run_oracle.py).

PPL is teacher-forced: it scores next-token prediction given gold context and
misses error accumulation under real decoding. This sweep re-uses the same
per-token FFN masker but scores lm-eval-harness multiple-choice / LAMBADA tasks,
so we can check whether the PPL headroom holds up on downstream accuracy.

The sparse MLPs are installed once and wrapped in a single HFLM; only the shared
ctrl["masker"] changes between points, so the model is never reloaded.

  .venv/bin/python run_zeroshot.py \
      --model NousResearch/Llama-2-7b-hf \
      --methods oracle_gate oracle_contrib \
      --sparsities 0 0.25 0.5 0.75 0.9 \
      --tasks lambada_openai piqa arc_easy winogrande hellaswag \
      --out results/zeroshot_llama2-7b.csv

Each downstream point is far heavier than a PPL point, so the sparsity grid
defaults to coarse; use --limit N for a quick smoke run (N examples per task).
"""
import argparse
import csv
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.actsparse import METHODS, build_masker, install_sparse_mlps

DEFAULT_TASKS = ["lambada_openai", "piqa", "arc_easy", "arc_challenge",
                 "winogrande", "hellaswag"]


def _parse_batch_size(s):
    return int(s) if s.isdigit() else s          # "auto" / "auto:N" pass through


def _scores(results, task):
    """Pull every accuracy metric (acc / acc_norm) and its stderr for a task."""
    md = results["results"].get(task, {})
    vals, errs = {}, {}
    for key, v in md.items():
        if not isinstance(v, (int, float)):
            continue
        name = key.split(",")[0]                 # "acc,none" -> "acc"
        if name.endswith("_stderr"):
            errs[name[:-len("_stderr")]] = float(v)
        elif name in ("acc", "acc_norm"):
            vals[name] = float(v)
    return [(m, vals[m], errs.get(m)) for m in vals]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--sparsities", nargs="+", type=float,
                   default=[0, 0.25, 0.5, 0.75, 0.9])
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--limit", type=float, default=None,
                   help="examples (or fraction) per task; None = full")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/zeroshot.csv")
    args = p.parse_args()

    for m in args.methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method '{m}'. known: {METHODS}")

    # import here so --help works without the lm-eval stack loaded
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    dtype = getattr(torch, args.dtype)
    print(f"[load] {args.model} ({args.dtype})")
    try:
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    except Exception:  # Llama-3+ ships no slow (sentencepiece) tokenizer
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True).to(args.device)
    model.eval()
    model.config.use_cache = False

    ctrl, _ = install_sparse_mlps(model)
    lm = HFLM(pretrained=model, tokenizer=tok,
              batch_size=_parse_batch_size(args.batch_size))

    rows = []
    sparsities = sorted(set(args.sparsities))

    def evaluate(method, sparsity):
        t0 = time.time()
        res = simple_evaluate(model=lm, tasks=args.tasks, limit=args.limit,
                              random_seed=args.seed, numpy_random_seed=args.seed,
                              torch_random_seed=args.seed, verbosity="ERROR")
        secs = time.time() - t0
        accs = []
        for task in args.tasks:
            for metric, val, err in _scores(res, task):
                rows.append({"method": method, "sparsity": round(sparsity, 4),
                             "task": task, "metric": metric,
                             "value": round(val, 4),
                             "stderr": round(err, 4) if err is not None else "",
                             "seconds": round(secs, 1)})
                if metric == "acc":
                    accs.append(val)
        mean = sum(accs) / len(accs) if accs else float("nan")
        print(f"  {method:15s} sparsity={sparsity:.2f}  mean_acc={mean:.4f}  [{secs:.0f}s]")
        _write_csv(args.out, rows)

    if 0 in sparsities:
        ctrl["masker"] = None
        evaluate("dense", 0.0)

    for method in args.methods:
        for sp in sparsities:
            if sp == 0:
                continue
            ctrl["masker"] = build_masker(method, sp, args.device, seed=args.seed)
            evaluate(method, sp)
    ctrl["masker"] = None

    print(f"[done] wrote {args.out}  ({len(rows)} rows)")


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "sparsity", "task", "metric",
                                          "value", "stderr", "seconds"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
