#!/usr/bin/env python
"""Combine results/ppl_sweep/*.csv into one figure.

Left: absolute WikiText-2 PPL vs oracle_gate sparsity (note: Gemma and Qwen use
different tokenizers, so absolute PPL is only comparable within a model family).
Right: PPL relative to each config's own dense baseline — the curve *shape*. If
bf16 and int4 coincide here, quantization and activation sparsity compose.
"""
import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt

STYLE = {  # color by model family, line style by precision
    "gemma4-12b-bf16": ("#1f77b4", "-",  "Gemma-4-12B base bf16"),
    "gemma4-12b-int4": ("#1f77b4", "--", "Gemma-4-12B base int4 (NF4)"),
    "qwen3-8b-bf16":   ("#d62728", "-",  "Qwen3-8B bf16"),
    "qwen3-8b-int4":   ("#d62728", "--", "Qwen3-8B int4 (official AWQ)"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/ppl_sweep")
    p.add_argument("--png", default=None, help="default: <dir>/ppl_vs_sparsity.png")
    args = p.parse_args()
    png = args.png or os.path.join(args.dir, "ppl_vs_sparsity.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for path in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            pts = sorted((float(r["sparsity"]), float(r["ppl"]))
                         for r in csv.DictReader(f))
        if not pts:
            continue
        color, ls, label = STYLE.get(name, (None, "-", name))
        xs, ys = zip(*pts)
        dense = dict(pts).get(0.0, ys[0])
        ax1.plot(xs, ys, "o", ls=ls, color=color, lw=2,
                 label=f"{label} (dense {dense:.2f})")
        ax2.plot(xs, [y / dense for y in ys], "o", ls=ls, color=color, lw=2,
                 label=label)

    ax1.set_ylabel("WikiText-2 PPL")
    ax1.set_title("Absolute PPL")
    ax2.set_ylabel("PPL / (own dense PPL)")
    ax2.set_title("Relative to own dense baseline (curve shape)")
    for ax in (ax1, ax2):
        ax.set_xlabel("FFN activation sparsity (oracle_gate drop fraction)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Oracle activation sparsity vs precision, Gemma-4-12B / Qwen3-8B")
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
