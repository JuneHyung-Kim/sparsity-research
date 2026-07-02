"""Overlay several BFCL sparsity tiers into one capability-vs-sparsity figure.

Reads the per-tier `sweep_scores.csv` files that sweep_plot.py produced (any number
of them) and draws ONE line per capability group (count-weighted accuracy), so
single_turn / multi_turn / memory sit on the same axes -- the "which capability
breaks first" picture. Only the oracle_gate method is plotted (dense is its s=0).

Tiers may come from different runs/conditions (e.g. single_turn fp8/THINK=0 on the
0.85 grid vs multi_turn+memory bf16/THINK=1 on the 0.8/0.9 grid); each line is drawn
at its own sparsity points, and the differing conditions belong in --title/caption.

Usage:
  python benchmarks/bfcl/final_plot.py \
    --scores results/bfcl_gemma4/single_turn/sweep_scores.csv \
    --scores results/bfcl_gemma4/mt_mem_bf16/sweep_scores.csv \
    --out results/bfcl_gemma4/final_capability_vs_sparsity.png \
    --md  results/bfcl_gemma4/final_capability_vs_sparsity.md
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_plot import GROUPS   # reuse the exact capability-group definitions


def load(paths):
    """-> weighted[(group, sparsity)] = [correct, total]."""
    cat2group = {c: g for g, cats in GROUPS.items() for c in cats}
    weighted = defaultdict(lambda: [0, 0])
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r["method"] != "oracle_gate":
                    continue
                g = cat2group.get(r["category"])
                if g is None:
                    continue
                try:
                    corr, tot = int(r["correct"]), int(r["total"])
                except (ValueError, TypeError):
                    continue
                s = float(r["sparsity"])
                weighted[(g, s)][0] += corr
                weighted[(g, s)][1] += tot
    return weighted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", required=True,
                    help="a sweep_scores.csv (repeat for each tier)")
    ap.add_argument("--out", default="results/bfcl_gemma4/final_capability_vs_sparsity.png")
    ap.add_argument("--md", default="results/bfcl_gemma4/final_capability_vs_sparsity.md")
    ap.add_argument("--title", default="Gemma-4-12B: BFCL capability vs oracle_gate sparsity")
    a = ap.parse_args()

    w = load(a.scores)
    if not w:
        raise SystemExit(f"no oracle_gate rows found in {a.scores}")
    groups = [g for g in GROUPS if any(k[0] == g for k in w)]
    sps = sorted({s for (_, s) in w})

    os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
    lines = [f"# {a.title}", "",
             "Count-weighted accuracy (sum correct / sum total). s=0 = dense. "
             "Tiers may differ in precision/THINK/grid -- see caption.", "",
             "| group | " + " | ".join(f"s={s:g}" for s in sps) + " |",
             "|" + "---|" * (len(sps) + 1)]
    for g in groups:
        cells = []
        for s in sps:
            c, t = w.get((g, s), [0, 0])
            cells.append(f"{100*c/t:.1f}%" if t else "—")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    with open(a.md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {a.md}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"skip {a.out}: matplotlib not installed (md written; run from a venv with it)")
        return
    plt.figure(figsize=(8, 5))
    for g in groups:
        xs = [s for s in sps if w.get((g, s), [0, 0])[1]]
        ys = [100 * w[(g, s)][0] / w[(g, s)][1] for s in xs]
        plt.plot(xs, ys, marker="o", label=g)
    plt.xlabel("per-token FFN activation sparsity (oracle_gate)")
    plt.ylabel("accuracy (%)")
    plt.title(a.title)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
