"""Collect BFCL scores across a sparsity sweep and plot accuracy vs sparsity.

benchmarks/bfcl/run.sh writes one project root per sparsity point, named
bfcl_run_s00 / bfcl_run_s50 / ... Each holds
    score/<served_name>/<group>/BFCL_v4_<category>_score.json
whose FIRST line is a summary like {"accuracy": 0.86, "correct_count": 207,
"total_count": 240}. We glob those, parse the sparsity from the dir name, and
draw one line per category.

Usage:
    python benchmarks/bfcl/plot.py --runs-dir . \
        --out results/bfcl_acc_vs_sparsity.png \
        --csv results/bfcl_acc_vs_sparsity.csv
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sparsity_from_dir(path):
    """bfcl_run_s50 -> 0.50, bfcl_run_s00 -> 0.0, bfcl_run -> 0.0."""
    name = os.path.basename(path.rstrip("/"))
    m = re.search(r"_s(\d+)$", name)
    return int(m.group(1)) / 100.0 if m else 0.0


def collect(runs_dir):
    """-> {category: [(sparsity, accuracy, correct, total), ...sorted]}."""
    rows = {}
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "bfcl_run*")))
    for rd in run_dirs:
        if not os.path.isdir(rd):
            continue
        s = sparsity_from_dir(rd)
        for sf in glob.glob(os.path.join(rd, "score", "**", "BFCL_v4_*_score.json"),
                            recursive=True):
            cat = re.sub(r"^BFCL_v4_|_score\.json$", "", os.path.basename(sf))
            with open(sf) as fh:
                head = json.loads(fh.readline())
            rows.setdefault(cat, []).append(
                (s, head.get("accuracy"), head.get("correct_count"),
                 head.get("total_count")))
    for cat in rows:
        rows[cat].sort(key=lambda r: r[0])
    return rows


def pivot_table(rows):
    """Markdown table: one row per sparsity, one accuracy column per category."""
    cats = sorted(rows)
    sps = sorted({p[0] for pts in rows.values() for p in pts})
    acc = {(cat, s): None for cat in cats for s in sps}
    for cat, pts in rows.items():
        for s, a, c, t in pts:
            acc[(cat, s)] = a
    lines = ["| sparsity | " + " | ".join(cats) + " |",
             "|" + "---|" * (len(cats) + 1)]
    for s in sps:
        cells = [f"{acc[(cat, s)] * 100:.2f}%" if acc[(cat, s)] is not None else "—"
                 for cat in cats]
        lines.append(f"| {s:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=".",
                    help="dir that contains the bfcl_run_s* project roots")
    ap.add_argument("--out", default="results/bfcl_acc_vs_sparsity.png")
    ap.add_argument("--csv", default="results/bfcl_acc_vs_sparsity.csv")
    ap.add_argument("--md", default="results/bfcl_acc_vs_sparsity.md")
    ap.add_argument("--title", default="BFCL accuracy vs activation sparsity (Qwen3-8B)")
    args = ap.parse_args()

    rows = collect(args.runs_dir)
    if not rows:
        raise SystemExit(f"no BFCL scores found under {args.runs_dir!r} "
                         "(expected bfcl_run_s*/score/**/BFCL_v4_*_score.json)")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "sparsity", "accuracy", "correct", "total"])
        for cat, pts in sorted(rows.items()):
            for s, acc, c, t in pts:
                w.writerow([cat, s, acc, c, t])
    print(f"wrote {args.csv}")

    # Pivot table: accuracy (%) by sparsity x category. Printed and saved as md.
    table = pivot_table(rows)
    print(f"\n{args.title}\n{table}\n")
    with open(args.md, "w") as fh:
        fh.write(f"# {args.title}\n\n{table}\n")
    print(f"wrote {args.md}")

    plt.figure(figsize=(7, 5))
    for cat, pts in sorted(rows.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        plt.plot(xs, ys, marker="o", label=cat)
    plt.xlabel("per-token FFN activation sparsity")
    plt.ylabel("BFCL accuracy (%)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
