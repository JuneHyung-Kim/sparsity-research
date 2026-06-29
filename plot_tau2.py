"""Collect tau2-bench metrics across a sparsity sweep and plot them vs sparsity.

The tau2 analogue of plot_bfcl.py. run_tau2.sh writes one dir per sparsity point,
named tau2_run_s00 / tau2_run_s50 / ... Each holds one compact record per domain
    tau2_run_s<NN>/<domain>.json
written by tau2_score.py: {domain, sparsity, method, avg_reward, pass_hat_1, ...}.
We glob those and draw one line per domain (default metric: pass^1, the
task-completion reliability that is tau2's headline number).

Reads only JSON, so it runs in the research .venv (no tau2 import needed).

Usage:
    python plot_tau2.py --runs-dir . \
        --out results/tau2_passk_vs_sparsity.png \
        --csv results/tau2_passk_vs_sparsity.csv
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(runs_dir):
    """-> ({domain: [(sparsity, value), ...sorted]}, all_records)."""
    records = []
    for sf in glob.glob(os.path.join(runs_dir, "tau2_run_s*", "*.json")):
        with open(sf) as fh:
            records.append(json.load(fh))
    return records


def series(records, metric):
    rows = {}
    for r in records:
        if r.get(metric) is None:
            continue
        rows.setdefault(r["domain"], []).append((r["sparsity"], r[metric]))
    for d in rows:
        rows[d].sort(key=lambda p: p[0])
    return rows


def pivot_table(records, metric):
    domains = sorted({r["domain"] for r in records})
    sps = sorted({r["sparsity"] for r in records})
    val = {}
    for r in records:
        val[(r["domain"], r["sparsity"])] = r.get(metric)
    lines = ["| sparsity | " + " | ".join(domains) + " |",
             "|" + "---|" * (len(domains) + 1)]
    for s in sps:
        cells = []
        for d in domains:
            v = val.get((d, s))
            cells.append(f"{v:.3f}" if v is not None else "—")
        lines.append(f"| {s:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=".",
                    help="dir that contains the tau2_run_s* dirs")
    ap.add_argument("--metric", default="pass_hat_1",
                    help="metric to plot (pass_hat_1 | avg_reward | pass_hat_2 ...)")
    ap.add_argument("--out", default="results/tau2_passk_vs_sparsity.png")
    ap.add_argument("--csv", default="results/tau2_passk_vs_sparsity.csv")
    ap.add_argument("--md", default="results/tau2_passk_vs_sparsity.md")
    ap.add_argument("--title", default="tau2-bench vs activation sparsity (Qwen3-8B)")
    args = ap.parse_args()

    records = collect(args.runs_dir)
    if not records:
        raise SystemExit(f"no tau2 metrics found under {args.runs_dir!r} "
                         "(expected tau2_run_s*/<domain>.json)")

    # CSV: every metric we have, long-form.
    metric_keys = sorted({k for r in records for k in r
                          if k.startswith("pass_hat_") or k in ("avg_reward",)})
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "sparsity", "method", "total_simulations"] + metric_keys)
        for r in sorted(records, key=lambda r: (r["domain"], r["sparsity"])):
            w.writerow([r["domain"], r["sparsity"], r.get("method"),
                        r.get("total_simulations")] + [r.get(k) for k in metric_keys])
    print(f"wrote {args.csv}")

    table = pivot_table(records, args.metric)
    title = f"{args.title} [{args.metric}]"
    print(f"\n{title}\n{table}\n")
    with open(args.md, "w") as fh:
        fh.write(f"# {title}\n\n{table}\n")
    print(f"wrote {args.md}")

    rows = series(records, args.metric)
    plt.figure(figsize=(7, 5))
    for d, pts in sorted(rows.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=d)
    plt.xlabel("per-token FFN activation sparsity (agent)")
    plt.ylabel(args.metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
