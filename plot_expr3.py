#!/usr/bin/env python
"""expr3 figures from the saved CSV/npz (no GPU; re-runnable any time).

Reads:
  results/oracle_quant.csv              (a) PPL vs sparsity per precision
  results/ranking_stability.csv         (b) kept-set overlap vs sparsity
  results/ranking_stability_perlayer.npz (b) per-layer overlap
Writes the two figures the SUMMARY references.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"none": "#1f77b4", "int8": "#ff7f0e", "nf4": "#d62728"}
LABELS = {"none": "fp16", "int8": "int8", "nf4": "NF4 (4-bit)"}


def _read_oracle(path):
    by_q = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_q[r["quant"]].append((float(r["sparsity"]), float(r["ppl"])))
    for q in by_q:
        by_q[q].sort()
    return by_q


def plot_headroom(by_q, png):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))
    for q, pts in by_q.items():
        xs = np.array([x for x, _ in pts])
        ys = np.array([y for _, y in pts])
        dense = dict(pts).get(0.0, ys[0])
        c, lab = COLORS.get(q), LABELS.get(q, q)
        ax1.plot(xs, ys, "o-", color=c, lw=2, label=f"{lab} (dense {dense:.2f})")
        rel = ys / dense
        ax2.plot(xs, rel, "o-", color=c, lw=2, label=lab)
        m = xs <= 0.8
        ax3.plot(xs[m], rel[m], "o-", color=c, lw=2, label=lab)

    ax1.set_yscale("log")
    ax1.set_ylabel("WikiText-2 PPL (log)")
    ax1.set_title("Absolute PPL")
    ax2.set_ylabel("PPL / own dense PPL")
    ax2.set_title("Relative to own dense (full range)")
    ax3.set_ylabel("PPL / own dense PPL")
    ax3.set_title("Relative, zoom s≤0.8 (curves coincide)")
    ax3.axhline(1.05, color="gray", ls=":", lw=1)
    ax3.annotate("+5%", (0.02, 1.052), color="gray", fontsize=9)
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("intra-FFN activation sparsity s (oracle drop fraction)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.suptitle("expr3 (a): oracle activation-sparsity headroom is precision-invariant "
                 "— weight quant ⊥ sparsity (LLaMA-2-7B)", y=1.02)
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


def _read_overlap(path):
    by_q = defaultdict(list)
    floor = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            s = float(r["sparsity"])
            by_q[r["quant"]].append((s, float(r["overlap_mean"]),
                                     float(r["overlap_min"])))
            floor[s] = float(r["random_floor"])
    for q in by_q:
        by_q[q].sort()
    return by_q, floor


def plot_stability(ov_path, npz_path, png, layer_s=0.7):
    by_q, floor = _read_overlap(ov_path)
    npz = np.load(npz_path)
    layers = npz["layers"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fs = sorted(floor)
    ax1.plot(fs, [floor[s] for s in fs], "k--", lw=1.2, label="random floor (1−s)")
    for q, pts in by_q.items():
        xs = [s for s, _, _ in pts]
        mean = np.array([m for _, m, _ in pts])
        mn = np.array([lo for _, _, lo in pts])
        c, lab = COLORS.get(q), LABELS.get(q, q)
        ax1.plot(xs, mean, "o-", color=c, lw=2, label=f"{lab} (mean)")
        ax1.fill_between(xs, mn, mean, color=c, alpha=0.15)
    ax1.set_xlabel("activation sparsity s (drop fraction)")
    ax1.set_ylabel("kept-set overlap with fp16  |K_fp ∩ K_q| / |K_fp|")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Quantized model keeps mostly the same neurons\n(band = min layer .. mean)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    for q in by_q:
        key = f"{q}_s{layer_s}"
        if key not in npz.files:
            continue
        ax2.plot(layers, npz[key], "-", color=COLORS.get(q), lw=2, label=LABELS.get(q, q))
    ax2.axhline(1 - layer_s, color="gray", ls="--", lw=1, label=f"random floor {1 - layer_s:.1f}")
    ax2.set_xlabel("decoder layer index")
    ax2.set_ylabel(f"kept-set overlap with fp16 (s={layer_s})")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(f"Per-layer overlap (s={layer_s}) — uniform across depth")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle("expr3 (b): does weight quantization change which neurons matter? (LLaMA-2-7B)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-csv", default="results/oracle_quant.csv")
    p.add_argument("--overlap-csv", default="results/ranking_stability.csv")
    p.add_argument("--overlap-npz", default="results/ranking_stability_perlayer.npz")
    p.add_argument("--headroom-png", default="results/ppl_vs_sparsity_quant.png")
    p.add_argument("--stability-png", default="results/ranking_stability.png")
    args = p.parse_args()

    plot_headroom(_read_oracle(args.oracle_csv), args.headroom_png)
    plot_stability(args.overlap_csv, args.overlap_npz, args.stability_png)


if __name__ == "__main__":
    main()
