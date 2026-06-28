# Contextual Activation Sparsity on SwiGLU LLMs

**Research question:** *Is it possible to improve sparsity detection?*

This is **token-dependent activation sparsity** (à la Deja Vu / PowerInfer), **not**
weight pruning. For each token, only a subset of FFN neurons meaningfully
contributes; if we can cheaply *detect* that subset we skip the rest →
speedup at ~no quality loss. A *better detector* pushes more sparsity at the
same perplexity (PPL).

The twist: the base model is **vanilla LLaMA-2-7B (SwiGLU/SiLU)**, which has
**no exact zeros** (unlike the ReLU models Deja Vu/PowerInfer rely on). So
"sparsity" is *soft* — skipping low-contribution neurons — and detection is
genuinely open. We are not reproducing prior work; we target the harder
SwiGLU regime.

## The neuron model

LLaMA FFN, per token `x`:

```
a_i  = SiLU(gate_proj(x)_i) * up_proj(x)_i      # neuron i's scalar activation
out  = Σ_i  a_i * down_proj[:, i]               # neuron i contributes a_i · column_i
```

"Skip neuron i for this token" = drop `a_i · down[:,i]`. Sparsity = fraction of
neurons skipped, chosen **per token** (contextual, not static).

## What it measures

- **oracle** — keep each token's top-(1−s) neurons by *true* importance, zero
  the rest. Needs every activation (no speedup), so it's the **ceiling** any
  real detector chases. Two rankings, to see if detection must be down_proj-aware:
  - `oracle_gate`   : rank by `|a_i|`
  - `oracle_contrib`: rank by `|a_i| · ‖down[:,i]‖`  (true output-contribution norm)
- **random** — drop a random per-token subset. The **floor**.
- The **gap between floor and ceiling is the room a good detector can win.**

`characterize.py` separately measures *intrinsic* headroom: how few neurons
carry most of each token's contribution mass, per layer.

### Outputs
- `results/ppl_vs_actsparsity.png` — PPL vs activation-sparsity (oracle ceiling,
  random floor, dense line)
- `results/sparsity_headroom.png` — captured-mass vs neurons-kept, per layer

## Setup

`.venv` already holds the deps (created with [uv](https://docs.astral.sh/uv/);
no system pip on this box):

```bash
uv venv --python 3.12 .venv && uv pip install torch transformers datasets accelerate sentencepiece protobuf matplotlib pandas tqdm numpy
```

## Run

```bash
# PPL vs activation-sparsity (oracle + random) -> CSV, then the figure
.venv/bin/python run_oracle.py \
    --model NousResearch/Llama-2-7b-hf \
    --methods random oracle_gate oracle_contrib \
    --sparsities 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --out results/oracle_llama2-7b.csv
.venv/bin/python plot.py --csv results/oracle_llama2-7b.csv --out results/ppl_vs_actsparsity.png

# Intrinsic sparsity headroom per layer
.venv/bin/python characterize.py --model NousResearch/Llama-2-7b-hf --out results/sparsity_headroom.png
```

`NousResearch/Llama-2-7b-hf` is an ungated mirror; use `meta-llama/Llama-2-7b-hf`
(after `huggingface-cli login`) for the official weights. Weights are never
modified — only the per-token FFN masker is swapped — so the model loads once.

## BFCL downstream metric (function-calling vs sparsity)

Instead of (or alongside) PPL, sweep BFCL function-calling accuracy vs activation
sparsity. The same per-token FFN masker is applied to a served HF model
(`bfcl_server.py`, an OpenAI-compatible `/v1/completions` endpoint); BFCL points
at it with `--skip-server-setup` so the masker is genuinely in the generation
path (vLLM/sglang would bypass it). See `results/SUMMARY_bfcl.md`.

Two venvs (the BFCL `[oss]` extra pins `vllm==0.8.5`, which would clobber the
research torch — so `bfcl-eval` core lives in its own venv):

```bash
uv venv --python 3.12 .venv      && uv pip install -r requirements-research.txt   # masker + server + plot
uv venv --python 3.12 .venv-bfcl && uv pip install -r requirements-bfcl.txt        # bfcl CLI only
```

One sparsity point at a time, then plot the sweep:

```bash
./run_bfcl.sh 0.0 oracle_gate simple_python,irrelevance   # dense baseline -> bfcl_run_s00/
./run_bfcl.sh 0.5 oracle_gate simple_python,irrelevance   # -> bfcl_run_s50/
./run_bfcl.sh 0.8 oracle_gate simple_python,irrelevance   # -> bfcl_run_s80/
.venv/bin/python plot_bfcl.py --runs-dir . --out results/bfcl_acc_vs_sparsity.png
```

### On a SLURM cluster (Vulcan / DRAC, L40S 46 GB)

Login node has internet but no GPU; compute nodes have GPUs but no internet, so
the venvs and the model are prepared on the login node and the sweep runs
offline under SLURM. Clone into your own dir under the project space
(`~/projects/aip-nanditav/sankeert/<you>/`):

```bash
source vulcan/env.sh        # repo venv paths + scratch HF cache, uv, SLURM account
bash vulcan/setup.sh        # LOGIN node: build both venvs + pre-download Qwen3-8B
sbatch vulcan/bfcl_sweep.slurm    # one L40S, offline; sweeps + writes the figure
```

`vulcan/env.sh` keeps the venvs (`.venv`, `.venv-bfcl`), the `bfcl_run_s*`
scores, and the `results/` figure/table/CSV **inside the repo** (project space,
persistent). Only the re-downloadable HF model cache and the uv toolchain live
on scratch (`$SCRATCH/jhkim`). Everything is overridable. Sweep knobs:

```bash
sbatch --export=ALL,SPARSITIES="0 0.3 0.5 0.7 0.8",CATS=simple_python,irrelevance,METHOD=oracle_gate \
       vulcan/bfcl_sweep.slurm
```

`--time` defaults to fit the 3h partition; for a longer sweep raise both the
partition (`#SBATCH --partition=gpubase_bygpu_b2`) and `--time`. The figure
lands at `results/bfcl_acc_vs_sparsity.png` (+ `.csv`).

## How the masking works (`src/actsparse.py`)

`install_sparse_mlps(model)` replaces every block's `.mlp` with a `SparseMLP`
that recomputes `a = act(gate(x))*up(x)`, optionally records stats, applies the
shared `ctrl["masker"]` (per-token top-k / random), then `down_proj`. Mutate
`ctrl["masker"]` to change method/sparsity with no reload.

## Layout

```
# expr1 — dense LLaMA-2-7B (results/SUMMARY.md)
run_oracle.py            # (method × sparsity) -> PPL CSV
characterize.py          # intrinsic sparsity headroom per layer
plot.py                  # CSV -> PPL-vs-sparsity figure
src/actsparse.py         # SparseMLP wrapper, per-token maskers, MassRecorder
src/eval_ppl.py          # WikiText-2 perplexity
src/data.py              # WikiText-2 loader

# expr2 / Q1 — MoE active-expert intra-sparsity (Qwen3-MoE)
characterize_moe.py      # intra-expert headroom per layer
run_oracle_moe.py        # oracle PPL vs intra-expert sparsity
src/moe.py               # recording/oracle experts-forward hook

# expr3 / Q2 — activation sparsity under weight quantization (results/SUMMARY_expr3.md)
run_oracle_quant.py      # oracle PPL vs sparsity for fp16 / int8 / NF4
run_ranking_stability.py # kept-set overlap fp16 vs quantized
plot_expr3.py            # CSV/npz -> the two expr3 figures (no GPU)
```
