#!/usr/bin/env python
"""expr1: compare the three per-token neuron-ranking signals on one figure.

  Gate magnitude     |silu(gate)|            oracle_gateonly  (realizable: chosen
                                             before up_proj -> skips up/down matmuls)
  Activation magnitude |silu(gate)*up|       oracle_gate      (oracle, no speedup)
  Output-aware score |silu(gate)*up|*||down|| oracle_contrib  (true output norm)

gate-only PPL blows up at high sparsity (~141 @0.9) while the other two stay <9.
We plot a linear axis zoomed to the operating region (--zoom-top); the gate-only
curve simply runs off the top once it diverges, which is the point.

  .venv/bin/python plot_signals.py
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

# method -> (legend label, color)
SIGNALS = {
    "oracle_gateonly": ("Gate magnitude  |silu(gate)|", "#1f77b4"),
    "oracle_gate":     ("Activation magnitude  |silu(gate)·up|", "#ff7f0e"),
    "oracle_contrib":  ("Output-aware score  |silu(gate)·up|·‖down‖", "#2ca02c"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+",
                   default=["results/oracle_llama2-7b.csv",
                            "results/oracle_gateonly.csv"])
    p.add_argument("--out", default="results/ppl_vs_sparsity_signals.png")
    p.add_argument("--zoom-top", type=float, default=8.5,
                   help="upper PPL limit for the zoomed (left) panel")
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)
    df = df.drop_duplicates(subset=["method", "sparsity"])
    dense = df.loc[df["method"] == "dense", "ppl"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 5))

    for method, (label, color) in SIGNALS.items():
        sub = (df[df["method"] == method]
               .dropna(subset=["ppl"])          # 0.8 gate-only is a float16 NaN
               .sort_values("sparsity"))
        if sub.empty:
            continue
        ax.plot(sub["sparsity"], sub["ppl"], marker="o", color=color,
                label=label, lw=1.8, ms=5)

    ax.axhline(dense, ls="--", color="gray", lw=1, label=f"dense ({dense:.2f})")
    ax.set_ylim(dense - 0.05, args.zoom_top)
    ax.set_xlabel("activation sparsity (fraction of neurons dropped)")
    ax.set_ylabel("WikiText-2 PPL")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("LLaMA-2-7B oracle: which neuron-ranking signal holds up?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
