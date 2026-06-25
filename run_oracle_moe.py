#!/usr/bin/env python
"""expr2: oracle PPL vs intra-expert activation sparsity (Qwen3-MoE).

For each active expert, oracle-drops the bottom-s fraction of its neurons (by
|a|*||down_i||) before down_proj, then measures WikiText-2 PPL. sparsity=0 is the
dense baseline. This is the *ceiling* (uses true contributions) — it answers
"how much can intra-expert activation sparsity buy at no/low PPL cost?".
"""
import argparse, csv, math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

from src.data import get_wikitext2_testenc

_CTRL = {"sparsity": 0.0}


def _oracle_forward(self, hidden_states, top_k_index, top_k_weights):
    s = _CTRL["sparsity"]
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        em = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
    for ei in hit:
        ei = ei[0]
        if ei == self.num_experts:
            continue
        pos, tok = torch.where(em[ei])
        x = hidden_states[tok]
        gate, up = F.linear(x, self.gate_up_proj[ei]).chunk(2, dim=-1)
        a = self.act_fn(gate) * up
        if s > 0:
            col = self.down_proj[ei].detach().float().norm(dim=0)
            score = a.detach().abs().float() * col
            ndrop = int(round(s * a.shape[-1]))
            idx = torch.topk(score, ndrop, dim=-1, largest=False).indices
            a = a.scatter(-1, idx, 0.0)
        out = F.linear(a, self.down_proj[ei]) * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, out.to(final.dtype))
    return final


ALL_EXPERTS_FUNCTIONS["oracle"] = _oracle_forward


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--segments", type=int, default=10)
    p.add_argument("--sparsities", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--offload-folder", default="/home/jhkim/workdir/offload")
    p.add_argument("--out", default="results/oracle_moe.csv")
    p.add_argument("--png", default="results/ppl_vs_intraexpert_sparsity.png")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        offload_folder=args.offload_folder, low_cpu_mem_usage=True)
    model.eval(); model.config.use_cache = False
    for blk in model.model.layers:
        if getattr(blk.mlp, "experts", None) is not None:
            blk.mlp.experts.config._experts_implementation = "oracle"
    model.config._experts_implementation = "oracle"

    testenc = get_wikitext2_testenc(tok)
    nseg = min(args.segments, testenc.shape[1] // args.seqlen)
    loss_fct = nn.CrossEntropyLoss()

    rows = []
    for s in args.sparsities:
        _CTRL["sparsity"] = s
        nlls = []
        for i in range(nseg):
            batch = testenc[:, i * args.seqlen:(i + 1) * args.seqlen].to(0)
            logits = model(batch).logits[0]            # [seqlen, V] bf16
            sl, lab = logits[:-1], batch[0, 1:]
            sum_nll = 0.0                              # chunk the float upcast to fit VRAM
            for c in range(0, sl.size(0), 256):
                lg = sl[c:c + 256].float()
                sum_nll += loss_fct(lg, lab[c:c + 256]).item() * lg.size(0)
            nlls.append(sum_nll / sl.size(0) * args.seqlen)
            del logits, sl
        ppl = math.exp(sum(nlls) / (nseg * args.seqlen))
        rows.append((s, ppl))
        print(f"  sparsity {s:.2f} -> PPL {ppl:.3f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sparsity", "ppl"]); w.writerows(rows)

    xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", color="#d62728", lw=2)
    ax.axhline(ys[0], color="gray", ls="--", lw=1, label=f"dense PPL {ys[0]:.2f}")
    ax.set_xlabel("intra-expert activation sparsity (oracle drop fraction)")
    ax.set_ylabel("WikiText-2 PPL")
    ax.set_title(f"Oracle PPL vs intra-expert sparsity ({args.model.split('/')[-1]})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(args.png, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} and {args.png}")


if __name__ == "__main__":
    main()
