#!/usr/bin/env python
"""expr2 / Q1: how sparse are an MoE's *active* experts, per token?

Runs WikiText-2 once through Qwen3-MoE with a recorder on each active expert's
SwiGLU activation, then plots — per layer — the mean fraction of a routed
token's expert-output contribution captured by that expert's top-k neurons.
Early saturation = lots of skippable neurons *inside* each active expert.

The model (30B bf16) does not fit in 24GB, so device_map="auto" spreads it over
GPU + CPU + disk. This is a one-shot characterization (~tens of forwards), so the
offload is slow but fine.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import get_wikitext2_testenc
from src.moe import ExpertMassRecorder, install_expert_recorder


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-segments", type=int, default=20,
                   help="cap WikiText-2 segments (0 = all); stats converge fast")
    p.add_argument("--offload-folder", default="/home/jhkim/workdir/offload")
    p.add_argument("--gate-only", action="store_true",
                   help="rank by |a| instead of |a|*||down_i||")
    p.add_argument("--out", default="results/moe_expert_headroom.png")
    p.add_argument("--npz", default="results/moe_expert_headroom.npz")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        offload_folder=args.offload_folder, low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False

    testenc = get_wikitext2_testenc(tok)
    rec = ExpertMassRecorder(contribution_aware=not args.gate_only)
    ctrl, restore = install_expert_recorder(model, rec)

    nseg = testenc.shape[1] // args.seqlen
    if args.max_segments:
        nseg = min(nseg, args.max_segments)
    for i in range(nseg):
        batch = testenc[:, i * args.seqlen:(i + 1) * args.seqlen].to(0)
        model(batch)
        print(f"  segment {i + 1}/{nseg}", flush=True)
    restore()

    layers = sorted(rec.curves)
    curves = np.stack([rec.curves[i].cpu().numpy() for i in layers])  # [L, moe_inter]
    inter = curves.shape[1]
    frac_kept = np.arange(1, inter + 1) / inter
    np.savez(args.npz, curves=curves, frac_kept=frac_kept, layers=np.array(layers))

    fig, ax = plt.subplots(figsize=(7, 5))
    for li, lyr in enumerate(layers):
        ax.plot(frac_kept, curves[li], color=plt.cm.viridis(lyr / max(layers)),
                alpha=0.25 + 0.65 * lyr / max(layers), lw=1)
    ax.plot(frac_kept, curves.mean(0), color="red", lw=2.5, label="mean over layers")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="uniform (no sparsity)")
    ax.set_xlabel("fraction of an active expert's neurons kept (top-k by contribution)")
    ax.set_ylabel("mean captured contribution mass")
    ax.set_title(f"Intra-expert sparsity headroom ({args.model.split('/')[-1]})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} and {args.npz}")

    mean_curve = curves.mean(0)
    for q in (0.9, 0.95, 0.99):
        k = int(np.searchsorted(mean_curve, q)) + 1
        print(f"  capture {q:.0%} of mass -> keep {k}/{inter} neurons "
              f"({k / inter:.1%}); skippable ~{1 - k / inter:.1%}")


if __name__ == "__main__":
    main()
