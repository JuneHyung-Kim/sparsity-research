#!/usr/bin/env python3
"""Probe peak GPU memory for the served model at a grid of (batch, sequence length).

The BFCL server (benchmarks/bfcl/server.py) micro-batches up to --batch concurrent
requests, each of which can grow to ~input + max_new tokens. Peak VRAM is dominated
by the KV cache (batch x seqlen), so a batch that fits the short single-turn
categories can still OOM on multi_turn_long_context. Run this BEFORE committing a
multi-hour sweep to pick the largest batch that fits.

For each (batch, seqlen) it prefills a seqlen-long random batch via
generate(max_new_tokens=1): that allocates the full KV cache the real decode path
would, WITHOUT materialising the [batch x seqlen x vocab] logits that a plain
forward() builds -- that logits tensor (not the KV cache) is what makes a naive
forward()-based probe OOM spuriously even at small batch.

Usage (on a GPU node, model already in the HF cache):
    HF_HUB_OFFLINE=1 .venv/bin/python vulcan/probe_vram.py
    HF_HUB_OFFLINE=1 .venv/bin/python vulcan/probe_vram.py --batches 16,12,8 --seqlens 12000,16000,20000

Pick the largest batch whose worst-case seqlen stays a few GB under the card
(reported "OK"), then pass it as BATCH=/NUM_THREADS= to vulcan/bfcl_sweep.slurm.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batches", default="16,12,8",
                    help="comma-separated server --batch values to test")
    ap.add_argument("--seqlens", default="12000,16000,20000",
                    help="comma-separated FINAL sequence lengths (input + generated)")
    ap.add_argument("--headroom-gb", type=float, default=4.0,
                    help="GB to keep free for fragmentation/CUDA context -> 'OK' cutoff")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    seqlens = [int(s) for s in args.seqlens.split(",")]

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device: {torch.cuda.get_device_name(0)} ({total_gb:.0f} GB), "
          f"OK cutoff = {total_gb - args.headroom_gb:.0f} GB")
    print(f"loading {args.model} (bf16) ...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    cfg = m.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    kv_kb = 2 * cfg.num_key_value_heads * head_dim * cfg.num_hidden_layers * 2 / 1024
    print(f"KV cache: {kv_kb:.0f} KB/token "
          f"({cfg.num_hidden_layers} layers x {cfg.num_key_value_heads} KV heads x {head_dim} head_dim)")
    print(f"weights resident: {torch.cuda.memory_reserved() / 1e9:.1f} GB\n")

    print(f"{'batch':>6} {'seqlen':>8} {'peak GB':>9}   verdict")
    for b in batches:
        for s in seqlens:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                ids = torch.randint(0, 1000, (b, s), device="cuda")
                with torch.no_grad():
                    m.generate(ids, max_new_tokens=1, do_sample=False, pad_token_id=0)
                peak = torch.cuda.max_memory_reserved() / 1e9
                verdict = "OK" if peak < total_gb - args.headroom_gb else "TIGHT"
                print(f"{b:>6} {s:>8} {peak:>8.1f}   {verdict}", flush=True)
                del ids
            except RuntimeError as e:
                tag = "OOM" if "out of memory" in str(e).lower() else f"ERR: {str(e)[:40]}"
                print(f"{b:>6} {s:>8} {'--':>8}   {tag}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
