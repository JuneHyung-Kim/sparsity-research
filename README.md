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
