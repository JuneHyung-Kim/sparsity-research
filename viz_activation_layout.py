#!/usr/bin/env python
"""Visualize *where* the active FFN neurons sit, per token, per layer.

Scoring criterion is fixed to |silu(gate)*up| (= |a_i|). For each token we keep
the top-(1-sparsity) neurons by |a_i| and call those "active" — i.e. the neurons
whose gate/up rows and down column are actually loaded/computed for that token.

The question is whether the active neurons CLUSTER into contiguous index blocks.
That matters because a GPU skips work at tile (block) granularity: a tile of B
contiguous neurons is skippable only if *all* B are inactive. So we don't plot
11008 neurons individually; we aggregate to blocks of B and show each block's
active fraction (0 = whole tile skippable, 1 = full tile, ~p = scattered/worst).

Outputs (under --out-prefix):
  *_block_per_token.png  rows=tokens, cols=blocks; one panel per representative layer
  *_block_per_layer.png  one token: rows=all layers, cols=blocks
  *_raw_zoom.png         one (token, layer): raw 11008 mask reshaped to a grid
Plus a per-layer clustering ratio = std(block fraction) / std-if-random. >1 means
more clustered than a uniform-random placement of the same number of actives.

  .venv/bin/python viz_activation_layout.py \
      --model NousResearch/Llama-2-7b-hf --sparsity 0.5 --block 128 \
      --tok-offset 512 --n-tokens 64 --layers 0 8 16 24 31
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.actsparse import install_sparse_mlps
from src.data import get_wikitext2_testenc


class MaskRecorder:
    """Capture, per layer, a per-token binary active mask for a token window.
    active_i = neuron i is in the per-token top-(1-sparsity) by |a_i|."""

    def __init__(self, offset, n_tokens, sparsity):
        self.offset = offset
        self.n_tokens = n_tokens
        self.sparsity = sparsity
        self.masks = {}                                  # layer idx -> bool [T, inter]

    def __call__(self, wrapper, a):
        if wrapper.idx in self.masks:                    # first forward only
            return
        win = a[0, self.offset:self.offset + self.n_tokens].abs().float()  # [T, inter]
        inter = win.shape[-1]
        k_keep = inter - int(round(self.sparsity * inter))
        keep = torch.topk(win, k_keep, dim=-1).indices
        mask = torch.zeros_like(win, dtype=torch.bool)
        mask.scatter_(-1, keep, True)
        self.masks[wrapper.idx] = mask.cpu()


def block_fraction(mask, block):
    """[T, inter] bool -> [T, n_blocks] float: active fraction per contiguous block."""
    T, inter = mask.shape
    assert inter % block == 0, f"inter {inter} not divisible by block {block}"
    return mask.reshape(T, inter // block, block).float().mean(-1).numpy()


def clustering_ratio(bf, keep_frac, block):
    """std of block fractions vs std under uniform-random placement of the same
    count. ~1 => scattered like random; >1 => clustered into blocks."""
    rand_std = np.sqrt(keep_frac * (1 - keep_frac) / block)
    return float(bf.std() / max(rand_std, 1e-9))


def record_masks(args):
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    model.config.use_cache = False

    testenc = get_wikitext2_testenc(tok)
    seg = testenc[:, args.segment * args.seqlen:(args.segment + 1) * args.seqlen]
    assert seg.shape[1] >= args.offset + args.n_tokens, "window exceeds segment"

    ctrl, _ = install_sparse_mlps(model)
    rec = MaskRecorder(args.offset, args.n_tokens, args.sparsity)
    ctrl["recorder"] = rec
    with torch.no_grad():
        model(seg.to(args.device))
    ctrl["recorder"] = None

    layers = sorted(rec.masks)
    masks = np.stack([rec.masks[i].numpy() for i in layers])      # [L, T, inter]
    token_ids = seg[0, args.offset:args.offset + args.n_tokens].numpy()
    np.savez(args.npz, masks=masks, layers=np.array(layers),
             token_ids=token_ids, sparsity=args.sparsity)
    print(f"wrote {args.npz}  masks={masks.shape}")
    return masks, np.array(layers), token_ids


def plot_block_per_token(masks, layers, sel, block, keep_frac, out):
    sel = [l for l in sel if l in layers.tolist()]
    li = {l: i for i, l in enumerate(layers.tolist())}
    n = len(sel)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.6 * n + 1.2), squeeze=False)
    for ax, L in zip(axes[:, 0], sel):
        bf = block_fraction(torch.from_numpy(masks[li[L]]), block)   # [T, n_blocks]
        im = ax.imshow(bf, aspect="auto", cmap="magma", vmin=0, vmax=1,
                       interpolation="nearest")
        r = clustering_ratio(bf, keep_frac, block)
        ax.set_ylabel(f"layer {L}\ntoken", fontsize=8)
        ax.set_title(f"layer {L}   clustering ratio = {r:.2f}  "
                     f"(1=random, >1=clustered)", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel(f"neuron block (size {block};  {masks.shape[2]//block} blocks)")
    fig.colorbar(im, ax=axes[:, 0].tolist(), label="block active fraction",
                 fraction=0.025, pad=0.01)
    fig.suptitle(f"Active-neuron block layout per token  "
                 f"(keep top {keep_frac:.0%} by |silu(gate)*up|)", fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def plot_block_per_layer(masks, layers, token, block, keep_frac, out):
    bf = np.stack([block_fraction(torch.from_numpy(masks[i]), block)[token]
                   for i in range(masks.shape[0])])               # [L, n_blocks]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(bf, aspect="auto", cmap="magma", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers.tolist(), fontsize=6)
    ax.set_xlabel(f"neuron block (size {block})")
    ax.set_ylabel("layer")
    ax.set_title(f"Active-neuron block layout, single token #{token}, all layers "
                 f"(keep top {keep_frac:.0%} by |silu(gate)*up|)", fontsize=10)
    fig.colorbar(im, ax=ax, label="block active fraction", fraction=0.03, pad=0.01)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def plot_raw_zoom(masks, layers, token, layer, out):
    li = layers.tolist().index(layer)
    m = masks[li][token]                                          # [inter] bool
    inter = m.shape[0]
    # reshape to a near-square grid in index order (rows wrap every `cols`)
    cols = 128
    rows = inter // cols
    grid = m[:rows * cols].reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(grid, aspect="auto", cmap="gray_r", interpolation="nearest")
    ax.set_title(f"Raw active mask, token #{token}, layer {layer}  "
                 f"({int(m.sum())}/{inter} active; index order, {cols}/row)",
                 fontsize=10)
    ax.set_xlabel(f"neuron index mod {cols}")
    ax.set_ylabel(f"neuron index // {cols}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--segment", type=int, default=0)
    p.add_argument("--offset", "--tok-offset", dest="offset", type=int, default=512)
    p.add_argument("--n-tokens", type=int, default=64)
    p.add_argument("--sparsity", type=float, default=0.5,
                   help="fraction of neurons dropped; keep = 1 - sparsity")
    p.add_argument("--block", type=int, default=128, help="tile/group size")
    p.add_argument("--layers", nargs="+", type=int, default=[0, 8, 16, 24, 31])
    p.add_argument("--zoom-token", type=int, default=0)
    p.add_argument("--zoom-layer", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out-prefix", default="results/viz_actlayout")
    p.add_argument("--npz", default="results/viz_actlayout.npz")
    p.add_argument("--from-npz", action="store_true",
                   help="skip the model, replot from a saved npz")
    args = p.parse_args()

    if args.from_npz:
        d = np.load(args.npz)
        masks, layers = d["masks"], d["layers"]
        args.sparsity = float(d["sparsity"])
    else:
        masks, layers, _ = record_masks(args)

    keep_frac = 1 - args.sparsity
    inter = masks.shape[2]
    print(f"intermediate={inter}  block={args.block}  "
          f"n_blocks={inter // args.block}  keep={keep_frac:.0%}")
    for i, L in enumerate(layers.tolist()):
        bf = block_fraction(torch.from_numpy(masks[i]), args.block)
        print(f"  layer {L:2d}  clustering ratio = "
              f"{clustering_ratio(bf, keep_frac, args.block):.2f}")

    plot_block_per_token(masks, layers, args.layers, args.block, keep_frac,
                         f"{args.out_prefix}_block_per_token.png")
    plot_block_per_layer(masks, layers, args.zoom_token, args.block, keep_frac,
                         f"{args.out_prefix}_block_per_layer.png")
    plot_raw_zoom(masks, layers, args.zoom_token, args.zoom_layer,
                  f"{args.out_prefix}_raw_zoom.png")


if __name__ == "__main__":
    main()
