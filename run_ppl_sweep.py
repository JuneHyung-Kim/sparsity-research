#!/usr/bin/env python
"""expr4: PPL vs activation sparsity on Gemma-4-12B and Qwen3-8B, bf16 vs int4.

Same protocol as expr1/expr3 (WikiText-2 test, seqlen 2048, SparseGPT averaging,
oracle_gate masker) on the two model families used in the agentic benchmarks,
each at bf16 and int4:

  * qwen3-8b:   Qwen/Qwen3-8B; int4 = Qwen's official AWQ (W4A16, rotation-free
    so per-neuron identity survives — QuaRot/SpinQuant would mix the FFN
    intermediate and erase the thing we measure).
  * gemma4-12b: google/gemma-4-12B (BASE, not -it). The -it model is turn-format
    bound: outside its chat DSL it emits garbage (raw-window WikiText PPL ~1000
    even through the vLLM engine that scores ~55% on tau2; the best turn-embedded
    variant was still ~117), so raw-LM PPL is only meaningful on the base model.
    Google ships no base QAT int4 for Gemma-4 (all qat variants are -it-), so
    int4 = bitsandbytes NF4 — the expr3 methodology, also rotation-free.

Each config writes its own CSV under results/ppl_sweep/ so configs can run (and
resume) independently; plot_ppl_sweep.py combines them.

  .venv/bin/python run_ppl_sweep.py --config qwen3-8b-bf16
  .venv/bin/python run_ppl_sweep.py --all
"""
import argparse
import csv
import gc
import os
import time

import torch
import transformers
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          AwqConfig, BitsAndBytesConfig)

from src.actsparse import build_masker, install_sparse_mlps
from src.data import get_wikitext2_testenc
from src.eval_ppl import eval_ppl

# bf16 Gemma-4-12B (~24GB weights) does not fit a 24GB card: device_map="auto"
# offloads the tail layers to CPU, and gpu_gib caps VRAM so the logits (262k
# vocab) still fit. NF4 packs the whole model onto the GPU (~8GB), so no cap.
# AWQ resolves dtype/backend (gptqmodel) from the checkpoint's
# quantization_config, so dtype=None there.
CONFIGS = {
    "gemma4-12b-bf16": dict(model="google/gemma-4-12B", dtype=torch.bfloat16,
                            gpu_gib=18),
    "gemma4-12b-int4": dict(model="google/gemma-4-12B", dtype=torch.bfloat16,
                            quant="nf4"),
    "qwen3-8b-bf16":   dict(model="Qwen/Qwen3-8B", dtype=torch.bfloat16,
                            gpu_gib=18),
    # backend: the default (marlin) JIT-builds a CUDA extension, which needs
    # nvcc — absent here. triton also JIT-compiles (needs Python.h, absent);
    # torch_awq is pure PyTorch.
    "qwen3-8b-int4":   dict(model="Qwen/Qwen3-8B-AWQ", dtype=None,
                            awq_backend="torch_awq"),
}
SPARSITIES = [0, 0.5, 0.6, 0.7, 0.8, 0.9]


def resolve_model_class(model_id):
    """Gemma-4-12B-it is a multimodal wrapper (Gemma4UnifiedForConditionalGeneration)
    that AutoModelForCausalLM rejects; instantiate whatever the checkpoint says."""
    cfg = AutoConfig.from_pretrained(model_id)
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    return getattr(transformers, arch, AutoModelForCausalLM)


def load_model(model_id, dtype, gpu_gib, awq_backend=None, quant=None):
    kwargs = dict(device_map="auto", low_cpu_mem_usage=True)
    if dtype is not None:
        kwargs["dtype"] = dtype
    if gpu_gib:
        kwargs["max_memory"] = {0: f"{gpu_gib}GiB", "cpu": "26GiB"}
    if awq_backend:
        kwargs["quantization_config"] = AwqConfig(backend=awq_backend)
    if quant == "nf4":
        kwargs["device_map"] = {"": 0}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = resolve_model_class(model_id).from_pretrained(model_id, **kwargs)
    model.eval()
    return model


@torch.no_grad()
def sweep(model, testenc, sparsities, seqlen, device, on_row):
    ctrl, _ = install_sparse_mlps(model)
    for sp in sparsities:
        ctrl["masker"] = None if sp == 0 else build_masker("oracle_gate", sp, device)
        t0 = time.time()
        ppl = eval_ppl(model, testenc, seqlen, device)
        on_row(sp, ppl, time.time() - t0)
    ctrl["masker"] = None


def run_config(name, args):
    spec = CONFIGS[name]
    out = os.path.join(args.outdir, f"{name}.csv")
    done = set()
    rows = []
    if os.path.exists(out) and not args.overwrite:
        with open(out) as f:
            rows = list(csv.DictReader(f))
        done = {float(r["sparsity"]) for r in rows}
    todo = [sp for sp in sorted(set(args.sparsities)) if sp not in done]
    if not todo:
        print(f"[skip] {name}: all {len(done)} points present in {out}", flush=True)
        return

    print(f"[load] {name}: {spec['model']}", flush=True)
    tok = AutoTokenizer.from_pretrained(spec["model"])
    testenc = get_wikitext2_testenc(tok)
    if args.max_segments:
        testenc = testenc[:, :args.max_segments * args.seqlen]
    model = load_model(spec["model"], spec["dtype"],
                       args.max_gpu_mem or spec.get("gpu_gib", 0),
                       awq_backend=spec.get("awq_backend"),
                       quant=spec.get("quant"))

    def on_row(sp, ppl, secs):
        rows.append({"config": name, "model": spec["model"],
                     "sparsity": round(sp, 4), "ppl": round(ppl, 4),
                     "seconds": round(secs, 1)})
        rows.sort(key=lambda r: float(r["sparsity"]))
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["config", "model", "sparsity",
                                              "ppl", "seconds"])
            w.writeheader()
            w.writerows(rows)
        print(f"  {name}  sparsity={sp:.2f}  ppl={ppl:.4f}  [{secs:.0f}s]", flush=True)

    sweep(model, testenc, todo, args.seqlen, args.device, on_row)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[done] {name} -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=sorted(CONFIGS), action="append",
                   help="run one config (repeatable); default with --all: all four")
    p.add_argument("--all", action="store_true")
    p.add_argument("--sparsities", nargs="+", type=float, default=SPARSITIES)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-segments", type=int, default=0,
                   help="cap WikiText-2 windows (0 = full test set; use 2 for smoke)")
    p.add_argument("--max-gpu-mem", type=int, default=0,
                   help="override the per-config VRAM cap (GiB, 0 = use config)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default="results/ppl_sweep")
    p.add_argument("--overwrite", action="store_true",
                   help="redo points already present in the CSV")
    args = p.parse_args()

    names = args.config or (sorted(CONFIGS) if args.all else None)
    if not names:
        raise SystemExit("pass --config NAME (repeatable) or --all")
    os.makedirs(args.outdir, exist_ok=True)
    for name in names:
        run_config(name, args)


if __name__ == "__main__":
    main()
