#!/usr/bin/env python
"""oracle CSV -> PPL-vs-(activation)sparsity figure."""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "random": "random",
    "oracle_gate": "top-k |a|",
    "oracle_contrib": "top-k |a|·‖down‖",
}
ORDER = ["random", "oracle_gate", "oracle_contrib"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="results/ppl_vs_actsparsity.png")
    p.add_argument("--title", default="")
    p.add_argument("--linear-y", action="store_true")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="method names to omit from the plot (e.g. random)")
    p.add_argument("--caption", default=None,
                   help="footnote text rendered under the axes (\\n for lines)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    methods = [m for m in df["method"].unique()
               if m != "dense" and m not in args.exclude]
    methods.sort(key=lambda m: ORDER.index(m) if m in ORDER else len(ORDER))

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        sub = df[df["method"] == m].sort_values("sparsity")
        ax.plot(sub["sparsity"], sub["ppl"], marker="o", label=LABELS.get(m, m))
    if (df["method"] == "dense").any():
        d = df[df["method"] == "dense"]["ppl"].iloc[0]
        ax.axhline(d, ls="--", color="gray", lw=1, label=f"dense ({d:.2f})")

    if not args.linear_y:
        ax.set_yscale("log")
    ax.set_xlabel("sparsity")
    ax.set_ylabel("PPL")
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if args.caption:
        cap = args.caption.replace("\\n", "\n")
        fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=7.5,
                 color="0.25", wrap=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
