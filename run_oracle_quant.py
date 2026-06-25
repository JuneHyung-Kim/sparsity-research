#!/usr/bin/env python
"""expr3 / Q2: does the activation-sparsity headroom survive weight quantization?

expr1 measured the oracle ceiling (drop the bottom-s neurons per token by |a|) on
*fp16* LLaMA-2-7B: ~70% of FFN neurons skippable at +5% PPL. Real deployments ship
weight-only-quantized models, so here we re-run that exact sweep on the same model
loaded fp16 / int8 / NF4 (bitsandbytes) and overlay the curves.

Two readings of the result:
  * absolute PPL vs sparsity  -> the deployed quality at each operating point
  * PPL / (that config's dense PPL) vs sparsity -> the *shape*. If the relative
    curves coincide across precisions, sparsity and weight-quant are orthogonal
    (they compose: 4-bit + 70%-sparse ~ additive). If the quantized curve rises
    faster, quant already ate some of the redundancy sparsity was exploiting.

Weight-only quant leaves the activations `a` in fp16, so the ranking *signal* is
untouched; only the weights (hence the true contributions and the baseline) move.
We rank by |a| (oracle_gate) — expr1 showed |a| ~ |a|*||down|| on this model, and
under bnb the packed integer down_proj has no usable float column norm anyway.
"""
import argparse
import csv
import gc
import time

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.actsparse import build_masker, install_sparse_mlps
from src.data import get_wikitext2_testenc
from src.eval_ppl import eval_ppl

QUANTS = ["none", "int8", "nf4"]


def load_model(model_id, quant, device, dtype):
    if quant == "none":
        m = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True).to(device)
    elif quant == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        m = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map={"": 0},
            low_cpu_mem_usage=True)
    elif quant == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        m = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map={"": 0},
            low_cpu_mem_usage=True)
    else:
        raise ValueError(quant)
    m.eval()
    m.config.use_cache = False
    return m


@torch.no_grad()
def sweep(model, testenc, sparsities, seqlen, device):
    """oracle_gate PPL at each sparsity; sparsity 0 = this config's dense PPL."""
    ctrl, _ = install_sparse_mlps(model)
    out = []
    for sp in sparsities:
        ctrl["masker"] = None if sp == 0 else build_masker("oracle_gate", sp, device)
        t0 = time.time()
        ppl = eval_ppl(model, testenc, seqlen, device)
        out.append((sp, ppl, time.time() - t0))
        print(f"    sparsity={sp:.2f}  ppl={ppl:.4f}  [{out[-1][2]:.0f}s]", flush=True)
    ctrl["masker"] = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--quants", nargs="+", default=QUANTS, choices=QUANTS)
    p.add_argument("--sparsities", nargs="+", type=float,
                   default=[0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-segments", type=int, default=0,
                   help="cap WikiText-2 segments (0 = full test set)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16"])
    p.add_argument("--out", default="results/oracle_quant.csv")
    p.add_argument("--png", default="results/ppl_vs_sparsity_quant.png")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    testenc = get_wikitext2_testenc(tok)
    if args.max_segments:
        testenc = testenc[:, :args.max_segments * args.seqlen]
    sparsities = sorted(set(args.sparsities))

    rows = []
    for q in args.quants:
        print(f"[load] {args.model}  quant={q}", flush=True)
        model = load_model(args.model, q, args.device, dtype)
        for sp, ppl, secs in sweep(model, testenc, sparsities, args.seqlen, args.device):
            rows.append({"quant": q, "sparsity": round(sp, 4),
                         "ppl": ppl, "seconds": round(secs, 1)})
            _write_csv(args.out, rows)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[free]  quant={q}  (gpu freed)", flush=True)

    _plot(rows, args.png, args.model)
    print(f"[done] wrote {args.out} and {args.png}  ({len(rows)} rows)")


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quant", "sparsity", "ppl", "seconds"])
        w.writeheader()
        w.writerows(rows)


def _plot(rows, png, model_id):
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quant"], []).append((r["sparsity"], r["ppl"]))
    colors = {"none": "#1f77b4", "int8": "#ff7f0e", "nf4": "#d62728"}
    labels = {"none": "fp16", "int8": "int8", "nf4": "NF4 (4-bit)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for q, pts in by_q.items():
        pts = sorted(pts)
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        dense = next((y for x, y in pts if x == 0), ys[0])
        c = colors.get(q, None)
        ax1.plot(xs, ys, "o-", color=c, lw=2, label=f"{labels.get(q, q)} (dense {dense:.2f})")
        ax2.plot(xs, [y / dense for y in ys], "o-", color=c, lw=2, label=labels.get(q, q))
    ax1.set_xlabel("intra-FFN activation sparsity (oracle drop fraction)")
    ax1.set_ylabel("WikiText-2 PPL")
    ax1.set_title("Absolute PPL")
    ax2.set_xlabel("intra-FFN activation sparsity (oracle drop fraction)")
    ax2.set_ylabel("PPL / (this config's dense PPL)")
    ax2.set_title("Relative to own dense baseline (curve shape)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Oracle activation sparsity vs weight quantization ({model_id.split('/')[-1]})")
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
