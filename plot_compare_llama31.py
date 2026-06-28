#!/usr/bin/env python
"""Compare contextual activation sparsity: Llama-3.1-8B-Instruct vs LLaMA-2-7B.

Left  : oracle PPL vs sparsity (ceiling = top-k by importance, floor = random),
        PPL expressed as a ratio to each model's own dense baseline so the two
        models are comparable despite different absolute PPL.
Right : intrinsic headroom — mean captured contribution mass vs neurons kept.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = [
    ("Llama-3.1-8B", "results/oracle_llama31-8b.csv",
     "results/sparsity_headroom_llama31-8b.npz", "tab:red"),
    ("LLaMA-2-7B", "results/oracle_llama2-7b.csv",
     "results/sparsity_headroom.npz", "tab:blue"),
]
# deployable gate-magnitude predictor ceiling, measured on LLaMA-2 only (expr1)
GATEONLY = ("LLaMA-2-7B", "results/oracle_gateonly.csv", "tab:blue")

fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))

# --- left: oracle PPL ratio vs sparsity --------------------------------------
STYLE = {"oracle_gate": "-", "oracle_contrib": "--", "random": ":"}
for name, csv, _npz, color in MODELS:
    df = pd.read_csv(csv)
    dense = df[df.method == "dense"].ppl.iloc[0]
    for method, ls in STYLE.items():
        sub = df[df.method == method].sort_values("sparsity")
        if sub.empty:
            continue
        axl.plot(sub.sparsity, sub.ppl / dense, ls=ls, marker="o", ms=4,
                 color=color,
                 label=f"{name} · {method.replace('oracle_','').replace('_',' ')}")
axl.axhline(1.0, color="gray", lw=1, ls="-", alpha=0.6)
axl.set_yscale("log")
axl.set_xlabel("activation sparsity (fraction of neurons skipped)")
axl.set_ylabel("PPL / dense PPL")
axl.set_title("Oracle ceiling vs random floor")
axl.grid(True, which="both", alpha=0.3)
axl.legend(fontsize=7.5, ncol=1)

# --- right: intrinsic headroom -----------------------------------------------
for name, _csv, npz, color in MODELS:
    d = np.load(npz)
    mean = d["curves"].mean(0)
    fk = d["frac_kept"]
    axr.plot(fk, mean, color=color, lw=2, label=name)
axr.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="no sparsity")
axr.axhline(0.9, color="gray", lw=0.8, ls=":")
axr.set_xlabel("fraction of neurons kept (top-k by contribution)")
axr.set_ylabel("mean captured contribution mass")
axr.set_title("Intrinsic sparsity headroom (mean over layers)")
axr.grid(alpha=0.3)
axr.legend(fontsize=9)

fig.suptitle("Contextual activation sparsity: Llama-3.1-8B vs LLaMA-2-7B (WikiText-2)",
             fontsize=12)
fig.tight_layout()
out = "results/compare_llama31_vs_llama2.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
