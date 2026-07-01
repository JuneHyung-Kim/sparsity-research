"""Aggregate a BFCL sparsity sweep (method x sparsity) into CSV, tables, plots.

Reads the layout sweep.sh writes:
    <base>/dense/bfcl_run_s00/score/**/BFCL_v4_<cat>_score.json   # s=0, shared
    <base>/<method>/bfcl_run_s<NN>/score/**/BFCL_v4_<cat>_score.json

Each *_score.json's first line is {"accuracy","correct_count","total_count"}.
The `dense` subtree is the s=0 point for EVERY method (the masker no-ops at 0).

Outputs (under --out-dir):
    sweep_scores.csv           long form: method,category,sparsity,accuracy,correct,total
    sweep_by_category.md       per-category accuracy, rows=sparsity, one block per method
    sweep_by_group.md          capability-group weighted accuracy (correct/total), method x s
    sweep_groups.png           one panel per group, x=sparsity, y=acc%, line per method

Usage:
    python benchmarks/bfcl/sweep_plot.py --base <SWEEP_BASE> --out-dir results
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

# Capability groups for the single-turn (T1) sweep. Group accuracy is
# count-weighted (sum correct / sum total) across its categories.
GROUPS = {
    "simple (single call)": ["simple_python", "simple_java", "simple_javascript",
                             "live_simple"],
    "compositional (multi/parallel)": ["multiple", "parallel", "parallel_multiple",
                                       "live_multiple", "live_parallel",
                                       "live_parallel_multiple"],
    "abstention (irrelevance)": ["irrelevance", "live_irrelevance"],
    "relevance (should call)": ["live_relevance"],
}


def sparsity_from_dir(path):
    m = re.search(r"_s(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) / 100.0 if m else 0.0


def collect(base):
    """-> raw[(method, sparsity, category)] = (accuracy, correct, total).

    The `dense` subtree is emitted as sparsity 0 for every method found."""
    per_dir = defaultdict(dict)   # dirname -> {(sparsity, cat): (acc, corr, tot)}
    for sub in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(sub):
            continue
        dirname = os.path.basename(sub)
        for rd in sorted(glob.glob(os.path.join(sub, "bfcl_run*"))):
            s = sparsity_from_dir(rd)
            for sf in glob.glob(os.path.join(rd, "score", "**", "BFCL_v4_*_score.json"),
                                recursive=True):
                cat = re.sub(r"^BFCL_v4_|_score\.json$", "", os.path.basename(sf))
                with open(sf) as fh:
                    head = json.loads(fh.readline())
                per_dir[dirname][(s, cat)] = (
                    head.get("accuracy"), head.get("correct_count"),
                    head.get("total_count"))

    methods = sorted(d for d in per_dir if d != "dense")
    dense = per_dir.get("dense", {})
    raw = {}
    for m in methods:
        for (s, cat), v in dense.items():          # dense -> s=0 for this method
            raw[(m, s, cat)] = v
        for (s, cat), v in per_dir[m].items():
            raw[(m, s, cat)] = v
    if not methods and dense:                       # dense-only run
        for (s, cat), v in dense.items():
            raw[("dense", s, cat)] = v
        methods = ["dense"]
    return raw, methods


def write_csv(raw, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "category", "sparsity", "accuracy", "correct", "total"])
        for (m, s, cat), (a, c, t) in sorted(raw.items()):
            w.writerow([m, cat, s, a, c, t])
    print(f"wrote {path}")


def group_weighted(raw, methods):
    """-> table[group][method][sparsity] = weighted accuracy (or None)."""
    sparsities = sorted({s for (_, s, _) in raw})
    out = {}
    for gname, cats in list(GROUPS.items()) + [("OVERALL (all single-turn)", None)]:
        out[gname] = {}
        for m in methods:
            out[gname][m] = {}
            for s in sparsities:
                corr = tot = 0
                for (mm, ss, cat), (a, c, t) in raw.items():
                    if mm != m or ss != s or c is None or t is None:
                        continue
                    if cats is not None and cat not in cats:
                        continue
                    corr += c
                    tot += t
                out[gname][m][s] = (corr / tot) if tot else None
    return out, sparsities


def write_group_md(table, sparsities, methods, path):
    lines = ["# BFCL accuracy vs activation sparsity — capability groups",
             "", "Count-weighted accuracy (sum correct / sum total). s=0 = dense.", ""]
    for gname, per_m in table.items():
        lines += [f"## {gname}", "",
                  "| method | " + " | ".join(f"s={s:g}" for s in sparsities) + " |",
                  "|" + "---|" * (len(sparsities) + 1)]
        for m in methods:
            cells = []
            for s in sparsities:
                v = per_m.get(m, {}).get(s)
                cells.append(f"{v*100:.2f}%" if v is not None else "—")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")


def write_category_md(raw, methods, path):
    cats = sorted({cat for (_, _, cat) in raw})
    sparsities = sorted({s for (_, s, _) in raw})
    lines = ["# BFCL accuracy vs sparsity — per category", ""]
    for m in methods:
        lines += [f"## {m}", "",
                  "| category | " + " | ".join(f"s={s:g}" for s in sparsities) + " |",
                  "|" + "---|" * (len(sparsities) + 1)]
        for cat in cats:
            cells = []
            for s in sparsities:
                v = raw.get((m, s, cat))
                cells.append(f"{v[0]*100:.2f}%" if v and v[0] is not None else "—")
            lines.append(f"| {cat} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")


def plot_groups(table, sparsities, methods, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"skip {path}: matplotlib not installed in this venv "
              "(CSV/markdown were still written; run this script from a venv with "
              "matplotlib to render the plot)")
        return
    groups = list(table.keys())
    n = len(groups)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    for i, gname in enumerate(groups):
        ax = axes[i // cols][i % cols]
        for m in methods:
            xs = [s for s in sparsities if table[gname][m].get(s) is not None]
            ys = [table[gname][m][s] * 100 for s in xs]
            if xs:
                ax.plot(xs, ys, marker="o", label=m)
        ax.set_title(gname)
        ax.set_xlabel("per-token FFN activation sparsity")
        ax.set_ylabel("accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SWEEP_BASE (has dense/ + <method>/)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--title", default="BFCL vs activation sparsity (Gemma-4-12B, fp8)")
    args = ap.parse_args()

    raw, methods = collect(args.base)
    if not raw:
        raise SystemExit(f"no scores under {args.base!r} "
                         "(expected <base>/<method>/bfcl_run_s*/score/**/*_score.json)")
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(raw, os.path.join(args.out_dir, "sweep_scores.csv"))
    write_category_md(raw, methods, os.path.join(args.out_dir, "sweep_by_category.md"))
    table, sparsities = group_weighted(raw, methods)
    write_group_md(table, sparsities, methods, os.path.join(args.out_dir, "sweep_by_group.md"))
    plot_groups(table, sparsities, methods,
                os.path.join(args.out_dir, "sweep_groups.png"), args.title)
    print(f"\nmethods={methods}  sparsities={sparsities}")


if __name__ == "__main__":
    main()
