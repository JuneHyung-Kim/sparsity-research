#!/usr/bin/env python
"""Measure how much contextual sparsity *exists* in a SwiGLU model.

Runs WikiText-2 once with a recorder, then plots, per layer, the mean fraction
of a token's total FFN output contribution captured by its top-k neurons.
A curve that saturates early = lots of skippable neurons (high headroom).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.actsparse import MassRecorder, install_sparse_mlps
from src.data import get_wikitext2_testenc


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-segments", type=int, default=20,
                   help="cap WikiText-2 segments for speed (0 = all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gate-only", action="store_true",
                   help="rank by |a| instead of |a|*||down_i||")
    p.add_argument("--out", default="results/sparsity_headroom.png")
    p.add_argument("--npz", default="results/sparsity_headroom.npz")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    model.config.use_cache = False

    testenc = get_wikitext2_testenc(tok)
    ctrl, _ = install_sparse_mlps(model)
    rec = MassRecorder(contribution_aware=not args.gate_only)
    ctrl["recorder"] = rec

    nseg = testenc.shape[1] // args.seqlen
    if args.max_segments:
        nseg = min(nseg, args.max_segments)
    for i in range(nseg):
        batch = testenc[:, i * args.seqlen:(i + 1) * args.seqlen].to(args.device)
        model(batch)
    ctrl["recorder"] = None

    layers = sorted(rec.curves)
    curves = np.stack([rec.curves[i].cpu().numpy() for i in layers])  # [L, I]
    inter = curves.shape[1]
    frac_kept = np.arange(1, inter + 1) / inter
    np.savez(args.npz, curves=curves, frac_kept=frac_kept, layers=np.array(layers))

    fig, ax = plt.subplots(figsize=(7, 5))
    for li in layers:
        a = 0.25 + 0.65 * li / max(layers)
        ax.plot(frac_kept, curves[li], color=plt.cm.viridis(li / max(layers)),
                alpha=a, lw=1)
    ax.plot(frac_kept, curves.mean(0), color="red", lw=2.5, label="mean over layers")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="uniform (no sparsity)")
    ax.set_xlabel("fraction of neurons kept (top-k by contribution)")
    ax.set_ylabel("mean captured contribution mass")
    ax.set_title(f"Contextual sparsity headroom ({args.model.split('/')[-1]})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} and {args.npz}")

    for q in (0.9, 0.95, 0.99):
        mean_curve = curves.mean(0)
        k = int(np.searchsorted(mean_curve, q)) + 1
        print(f"  capture {q:.0%} of mass -> keep {k}/{inter} neurons "
              f"({k / inter:.1%}); skippable ~{1 - k / inter:.1%}")


if __name__ == "__main__":
    main()
