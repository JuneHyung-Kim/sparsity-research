#!/usr/bin/env python
"""Dequantize a compressed-tensors pack-quantized checkpoint to plain bf16.

transformers' compressed-tensors path has no int4 inference kernels: the model
is always decompressed to bf16 in memory (CompressedLinear was removed
upstream), and with device_map offload the decompress step crashes on meta
tensors. So we dequantize once, offline and streaming (tensor at a time, <6GB
RAM), producing a normal bf16 checkpoint with the QAT int4 numerics baked in.
The result runs through the exact same loading path as the bf16 baseline.

  .venv/bin/python convert_ct_to_bf16.py \
      --src google/gemma-4-12B-it-qat-w4a16-ct \
      --dst /home/jhkim/workdir/models/gemma4-12b-it-qat-w4a16-bf16
"""
import argparse
import glob
import json
import os
import shutil

import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 4 * 2**30
# consumed alongside weight_packed, never copied through
QUANT_AUX = ("weight_scale", "weight_shape", "weight_zero_point", "weight_g_idx")


def dequant(f, base, num_bits=4, group_size=32):
    packed = f.get_tensor(base + "weight_packed")
    scale = f.get_tensor(base + "weight_scale")
    shape = torch.Size(f.get_tensor(base + "weight_shape").tolist())
    q = unpack_from_int32(packed, num_bits, shape)          # int8, [out, in]
    w = q.reshape(shape[0], -1, group_size).float() * scale.unsqueeze(-1).float()
    return w.reshape(shape).to(torch.bfloat16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()
    src = args.src if os.path.isdir(args.src) else snapshot_download(args.src)
    os.makedirs(args.dst, exist_ok=True)

    shard, shard_bytes, n_file, weight_map = {}, 0, 0, {}

    def flush():
        nonlocal shard, shard_bytes, n_file
        if not shard:
            return
        n_file += 1
        name = f"model-{n_file:05d}.safetensors"          # renamed to -of-N below
        save_file(shard, os.path.join(args.dst, name),
                  metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        shard, shard_bytes = {}, 0

    for path in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for name in sorted(f.keys()):
                if name.endswith(QUANT_AUX):
                    continue
                if name.endswith("weight_packed"):
                    out_name = name[:-len("_packed")]
                    t = dequant(f, name[:-len("weight_packed")])
                else:
                    out_name, t = name, f.get_tensor(name)
                shard[out_name] = t
                shard_bytes += t.numel() * t.element_size()
                if shard_bytes >= SHARD_BYTES:
                    flush()
    flush()

    for old, new in [(f"model-{i:05d}.safetensors",
                      f"model-{i:05d}-of-{n_file:05d}.safetensors")
                     for i in range(1, n_file + 1)]:
        os.rename(os.path.join(args.dst, old), os.path.join(args.dst, new))
        weight_map = {k: (new if v == old else v) for k, v in weight_map.items()}
    total = sum(os.path.getsize(os.path.join(args.dst, f))
                for f in os.listdir(args.dst) if f.endswith(".safetensors"))
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
                  f, indent=2, sort_keys=True)

    for path in glob.glob(os.path.join(src, "*")):
        base = os.path.basename(path)
        if base.endswith(".safetensors") or base.endswith(".index.json"):
            continue
        if base == "config.json":
            cfg = json.load(open(path))
            cfg.pop("quantization_config", None)
            json.dump(cfg, open(os.path.join(args.dst, base), "w"), indent=2)
        elif os.path.isfile(path):
            shutil.copy(path, os.path.join(args.dst, base))
    print(f"wrote {n_file} shards ({total/2**30:.1f} GiB) to {args.dst}")


if __name__ == "__main__":
    main()
