#!/usr/bin/env python
"""Does a FIXED per-layer neuron permutation make active sets contiguous?

A permutation reorders the (gate row, up row, down column) of each neuron once,
offline, identically for every token (it leaves the model output exact). The
question: is there ONE ordering that lands per-token active sets into few
contiguous blocks, so a tile-skip kernel can drop whole blocks?

Headline metric: fraction of neuron-blocks a token must TOUCH (>=1 active neuron
in the block => the tile is loaded/computed).
  scattered  -> ~1.0  (every tile touched, no saving)
  ideal pack -> keep_frac  (only the needed tiles)
We derive each permutation on TRAIN tokens and score it on held-out TEST tokens.

Permutations tried:
  identity      natural order (baseline)
  freq          sort neurons by activation frequency (captures static hot/cold)
  kmeans        cluster neurons by co-activation pattern into n_blocks groups,
                lay groups out contiguously (captures dynamic groups)

  .venv/bin/python reorder_test.py --sparsities 0.9 0.5 --block 128
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.actsparse import install_sparse_mlps
from src.data import get_wikitext2_testenc


class MultiMaskRecorder:
    """Per layer, per-token binary active masks for several sparsities at once."""

    def __init__(self, offset, n_tokens, sparsities):
        self.offset, self.n_tokens = offset, n_tokens
        self.sparsities = sparsities
        self.masks = {s: {} for s in sparsities}

    def __call__(self, wrapper, a):
        if wrapper.idx in self.masks[self.sparsities[0]]:
            return
        win = a[0, self.offset:self.offset + self.n_tokens].abs().float()
        inter = win.shape[-1]
        for s in self.sparsities:
            k_keep = inter - int(round(s * inter))
            keep = torch.topk(win, k_keep, dim=-1).indices
            m = torch.zeros_like(win, dtype=torch.bool)
            m.scatter_(-1, keep, True)
            self.masks[s][wrapper.idx] = m.cpu().numpy()


def frac_touched(mask, block):
    T, inter = mask.shape
    nb = inter // block
    m = mask[:, :nb * block].reshape(T, nb, block)
    return float(m.any(-1).mean())                       # mean over (token, block)


def block_fraction(mask, block):
    T, inter = mask.shape
    nb = inter // block
    return mask[:, :nb * block].reshape(T, nb, block).mean(-1)


def kmeans(X, K, iters=25):
    """X [N, D] on device -> cluster label per row. Deterministic init."""
    N = X.shape[0]
    C = X[torch.linspace(0, N - 1, K).long()].clone()
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        for k in range(K):
            sel = X[lab == k]
            if len(sel):
                C[k] = sel.mean(0)
    return lab


def make_perm(method, train_mask, block, device):
    inter = train_mask.shape[1]
    freq = train_mask.mean(0)                            # [inter]
    if method == "identity":
        return np.arange(inter)
    if method == "freq":
        return np.argsort(-freq)
    if method == "kmeans":
        X = torch.from_numpy(train_mask.T.astype(np.float32)).to(device)  # [inter, T]
        lab = kmeans(X, inter // block).cpu().numpy()
        key = lab.astype(np.float64) * 2.0 - freq        # group by cluster, freq desc
        return np.argsort(key)
    raise ValueError(method)


def coactivation_summary(mask, n_pairs=200_000, seed=0):
    """Mean |corr| of activation indicators over random neuron pairs, vs the
    independence null (~1/sqrt(T)). >> null => exploitable dynamic structure."""
    T, inter = mask.shape
    rng = np.random.default_rng(seed)
    i = rng.integers(0, inter, n_pairs)
    j = rng.integers(0, inter, n_pairs)
    ok = i != j
    i, j = i[ok], j[ok]
    X = mask.astype(np.float32)
    Xi, Xj = X[:, i], X[:, j]                            # [T, P]
    mi, mj = Xi.mean(0), Xj.mean(0)
    si = Xi.std(0) + 1e-6
    sj = Xj.std(0) + 1e-6
    corr = ((Xi * Xj).mean(0) - mi * mj) / (si * sj)
    return float(np.abs(corr).mean()), 1.0 / np.sqrt(T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--offset", type=int, default=256)
    p.add_argument("--n-tokens", type=int, default=1024, help="split half train/half test")
    p.add_argument("--sparsities", nargs="+", type=float, default=[0.9, 0.5])
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--methods", nargs="+", default=["identity", "freq", "kmeans"])
    p.add_argument("--viz-layer", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out-prefix", default="results/reorder")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    model.config.use_cache = False

    testenc = get_wikitext2_testenc(tok)
    seg = testenc[:, :args.seqlen]
    ctrl, _ = install_sparse_mlps(model)
    rec = MultiMaskRecorder(args.offset, args.n_tokens, args.sparsities)
    ctrl["recorder"] = rec
    with torch.no_grad():
        model(seg.to(args.device))
    ctrl["recorder"] = None

    half = args.n_tokens // 2
    layers = sorted(rec.masks[args.sparsities[0]])
    block = args.block

    for s in args.sparsities:
        keep = 1 - s
        per_method = {m: [] for m in args.methods}
        natural_touch, coact = [], []
        for L in layers:
            full = rec.masks[s][L]                       # [n_tokens, inter]
            tr, te = full[:half], full[half:]
            natural_touch.append(frac_touched(te, block))
            ca, null = coactivation_summary(full)
            coact.append(ca)
            for method in args.methods:
                perm = make_perm(method, tr, block, args.device)
                per_method[method].append(frac_touched(te[:, perm], block))

        print(f"\n=== keep top {keep:.0%} (sparsity {s:.0%})  "
              f"block={block}  ideal touch={keep:.3f} ===")
        print(f"  mean co-activation |corr| = {np.mean(coact):.4f}  "
              f"(independence null ~{null:.4f})")
        print(f"  {'method':10s}  mean frac blocks touched (held-out)")
        for method in args.methods:
            print(f"  {method:10s}  {np.mean(per_method[method]):.3f}")

        # before/after visual for one layer
        full = rec.masks[s][args.viz_layer]
        tr, te = full[:half], full[half:]
        best = min(args.methods, key=lambda m: np.mean(per_method[m]))
        perm = make_perm(best, tr, block, args.device)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        for a_, mask_, ttl in [(ax[0], te, "identity"),
                               (ax[1], te[:, perm], f"{best} (best)")]:
            im = a_.imshow(block_fraction(mask_, block), aspect="auto",
                           cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            a_.set_title(f"layer {args.viz_layer}, keep {keep:.0%}, {ttl}\n"
                         f"frac touched = {frac_touched(mask_, block):.3f}", fontsize=9)
            a_.set_xlabel("neuron block"); a_.set_ylabel("token")
        fig.colorbar(im, ax=ax, label="block active fraction", fraction=0.02)
        out = f"{args.out_prefix}_s{int(s*100)}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
