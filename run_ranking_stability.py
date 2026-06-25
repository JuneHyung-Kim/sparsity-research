#!/usr/bin/env python
"""expr3 / Q2 (b): does weight quantization change *which* neurons matter?

The oracle keeps each token's top-(1-s) FFN neurons ranked by |a|, where
a = act(gate(x))*up(x). Under weight-only quantization gate/up are quantized, so
`a` — and hence the ranking — shifts. This asks: how much does the kept set move?

A real detector (Deja Vu style) predicts the kept set from cheap features. If a
quantized model wants the *same* neurons as fp16 (high overlap), a detector built
on fp16 transfers to the quantized deployment for free; if the overlap drops, the
detector must be rebuilt against the quantized weights.

We load fp16 + a quantized copy together, push identical batches through both,
and measure the per-token kept-set overlap |K_fp ∩ K_q| / |K_fp| (= Jaccard for
equal-size sets), per layer and sparsity. Random-choice overlap at sparsity s is
(1-s), the floor; 1.0 means the quantized model ranks neurons identically.
"""
import argparse
import csv
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from run_oracle_quant import load_model
from src.actsparse import install_sparse_mlps
from src.data import get_wikitext2_testenc


def keep_mask(absa, s):
    """Top-(1-s) by magnitude -> bool kept mask. absa: [tokens, N]."""
    n = absa.shape[-1]
    keep = n - int(round(s * n))
    idx = torch.topk(absa, keep, dim=-1).indices
    m = torch.zeros_like(absa, dtype=torch.bool)
    m.scatter_(-1, idx, True)
    return m


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--quants", nargs="+", default=["int8", "nf4"],
                   choices=["int8", "nf4"])
    p.add_argument("--sparsities", nargs="+", type=float,
                   default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--segments", type=int, default=4,
                   help="WikiText-2 segments to average over (overlap converges fast)")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--out", default="results/ranking_stability.csv")
    p.add_argument("--png", default="results/ranking_stability.png")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    testenc = get_wikitext2_testenc(tok)
    nseg = min(args.segments, testenc.shape[1] // args.seqlen)
    sps = sorted(set(args.sparsities))

    print(f"[load] fp16 reference", flush=True)
    fp = load_model(args.model, "none", args.device, dtype)
    ctrl_fp, _ = install_sparse_mlps(fp)
    nlayers = len(list(fp.model.layers))

    rows = []
    per_layer_out = {}  # f"{q}_s{ s}" -> [nlayers] overlap, saved to npz
    for q in args.quants:
        print(f"[load] quant={q} (compared against fp16)", flush=True)
        qm = load_model(args.model, q, args.device, dtype)
        ctrl_q, _ = install_sparse_mlps(qm)

        # inter[s][layer], denom[s][layer] accumulate intersection / kept-count
        inter = {s: np.zeros(nlayers) for s in sps}
        denom = {s: np.zeros(nlayers) for s in sps}

        for i in range(nseg):
            batch = testenc[:, i * args.seqlen:(i + 1) * args.seqlen].to(args.device)

            store = {}  # layer -> |a| from the fp16 pass, on CPU (fp16)
            ctrl_fp["recorder"] = lambda w, a: store.__setitem__(
                w.idx, a.detach().abs().half().reshape(-1, a.shape[-1]).cpu())
            fp(batch)
            ctrl_fp["recorder"] = None

            def qrec(w, a):
                af = store[w.idx].to(a.device).float()
                aq = a.detach().abs().float().reshape(-1, a.shape[-1])
                for s in sps:
                    mf, mq = keep_mask(af, s), keep_mask(aq, s)
                    inter[s][w.idx] += (mf & mq).sum().item()
                    denom[s][w.idx] += mf.sum().item()
            ctrl_q["recorder"] = qrec
            qm(batch)
            ctrl_q["recorder"] = None
            store.clear()
            print(f"    quant={q} segment {i + 1}/{nseg}", flush=True)

        for s in sps:
            per_layer = inter[s] / np.maximum(denom[s], 1)
            per_layer_out[f"{q}_s{s}"] = per_layer
            rows.append({"quant": q, "sparsity": round(s, 4),
                         "overlap_mean": float(per_layer.mean()),
                         "overlap_min": float(per_layer.min()),
                         "random_floor": round(1 - s, 4)})
            _write_csv(args.out, rows)
            print(f"  quant={q} s={s:.2f}  overlap mean={per_layer.mean():.4f} "
                  f"min={per_layer.min():.4f}  (floor {1 - s:.2f})", flush=True)

        del qm
        gc.collect()
        torch.cuda.empty_cache()

    npz = args.out.rsplit(".", 1)[0] + "_perlayer.npz"
    np.savez(npz, layers=np.arange(nlayers), sparsities=np.array(sps),
             **per_layer_out)
    print(f"[npz] wrote {npz}", flush=True)

    _plot(rows, args.png, args.model)
    print(f"[done] wrote {args.out} and {args.png}")


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quant", "sparsity", "overlap_mean",
                                          "overlap_min", "random_floor"])
        w.writeheader()
        w.writerows(rows)


def _plot(rows, png, model_id):
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quant"], []).append(r)
    colors = {"int8": "#ff7f0e", "nf4": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 5))
    floor = sorted({(r["sparsity"], r["random_floor"]) for r in rows})
    ax.plot([s for s, _ in floor], [f for _, f in floor], "k--", lw=1,
            label="random floor (1-s)")
    for q, rs in by_q.items():
        rs = sorted(rs, key=lambda r: r["sparsity"])
        xs = [r["sparsity"] for r in rs]
        ax.plot(xs, [r["overlap_mean"] for r in rs], "o-", color=colors.get(q),
                lw=2, label=f"{q} (mean over layers)")
    ax.set_xlabel("activation sparsity s (drop fraction)")
    ax.set_ylabel("kept-set overlap with fp16  |K_fp ∩ K_q| / |K_fp|")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Does quantization keep the same neurons? ({model_id.split('/')[-1]})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
