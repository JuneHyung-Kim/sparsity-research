# BFCL under contextual activation sparsity

Measure how **token-dependent FFN activation sparsity** affects a model's
**function-calling** ability, using the Berkeley Function Calling Leaderboard
(BFCL V4) as the downstream metric instead of perplexity.

This is *contextual activation sparsity* (à la Deja Vu / PowerInfer), **not**
weight pruning: for each token only a subset of FFN neurons meaningfully
contributes, so we skip the rest. PPL is a weak proxy for "did the model get
worse"; BFCL reads out *which capability* a given sparsity level costs
(differential degradation across categories).

## The neuron model (`src/actsparse.py`)

SwiGLU FFN, per token `x`:

```
a_i  = SiLU(gate_proj(x)_i) * up_proj(x)_i      # neuron i's scalar activation
out  = Σ_i  a_i · down_proj[:, i]               # neuron i contributes a_i · column_i
```

"Skip neuron i for this token" = drop `a_i · down[:,i]`. Sparsity = fraction of
neurons skipped, chosen **per token**. `install_sparse_mlps(model)` wraps every
block's `.mlp` with a `SparseMLP` that recomputes `a`, applies the shared
`ctrl["masker"]`, then `down_proj`; `build_masker(method, sparsity, device)`
makes the masker. The default `oracle_gate` keeps each token's top-(1−s)
neurons by `|a_i|` — the **ceiling** any real (cheap) detector would chase.

## How it fits together

- **Model:** `Qwen/Qwen3-8B` (dense SwiGLU; same gate/up/down_proj + SiLU as
  LLaMA, so the masker is drop-in). bf16.
- **Serving:** `bfcl_server.py` — a minimal OpenAI `/v1/completions` server in
  the research venv. Weights load once; the masker is fixed at startup
  (`--method/--sparsity`). Qwen3 "thinking" is disabled for parse-stability.
- **Harness:** `bfcl-eval` in a separate venv (its `[oss]`/vllm extra would
  clobber the research torch). Run with `--skip-server-setup` so BFCL renders
  the FC prompt and parses `<tool_call>` itself and our masker is actually in
  the generation path (vLLM/sglang would bypass it).

## Setup (two venvs)

```bash
uv venv --python 3.12 .venv      && uv pip install -r requirements-research.txt   # masker + server + plot
uv venv --python 3.12 .venv-bfcl && uv pip install -r requirements-bfcl.txt        # bfcl CLI only
```

## Run

One sparsity point (serves the model, scores the categories, tears down):

```bash
./run_bfcl.sh <sparsity> [method] [categories]
# e.g. dense baseline over all key-free function-calling categories:
./run_bfcl.sh 0.0 oracle_gate single_turn,multi_turn
```

A point writes `bfcl_run_s<NN>/score/.../BFCL_v4_<category>_score.json`. After a
sweep, collect them into a figure + table + CSV:

```bash
.venv/bin/python plot_bfcl.py --runs-dir .
# -> results/bfcl_acc_vs_sparsity.{png,md,csv}   (accuracy vs sparsity, per category)
```

### Categories

`single_turn` (= `non_live` AST + `live`) and `multi_turn` are key-free and
scored. `web_search` needs a paid SerpAPI key; `memory` needs agentic backends;
`format_sensitivity` is non-scoring — all three are excluded by default.

## On a SLURM cluster (Vulcan / DRAC, L40S 46 GB)

Login node has internet but no GPU; compute nodes have GPUs but no internet, so
the venvs and the model are prepared on the login node and the sweep runs
offline under SLURM. Clone into your own dir under the project space
(`~/projects/aip-nanditav/sankeert/<you>/`):

```bash
source vulcan/env.sh        # repo venv paths + scratch HF cache, uv, SLURM account
bash vulcan/setup.sh        # LOGIN node: build both venvs + pre-download Qwen3-8B
sbatch vulcan/bfcl_sweep.slurm   # one L40S, offline; full sweep -> figure/table/CSV
```

`vulcan/env.sh` keeps the venvs (`.venv`, `.venv-bfcl`), the `bfcl_run_s*`
scores, and `results/` **inside the repo** (project space, persistent). Only the
re-downloadable HF model cache and the uv toolchain live on scratch
(`~/scratch/jhkim`). Everything is overridable.

`bfcl_sweep.slurm` defaults to the full sweep (`SPARSITIES="0 0.5 0.6 0.7 0.8"`,
`CATS=single_turn,multi_turn`) — a long multi-hour job on the 7-day partition.
Trim for a quick run:

```bash
sbatch --export=ALL,SPARSITIES="0 0.5 0.8",CATS=single_turn vulcan/bfcl_sweep.slurm
```

## Layout

```
bfcl_server.py           # OpenAI /v1/completions server with the masker installed
run_bfcl.sh              # serve at one sparsity, then bfcl generate + evaluate
plot_bfcl.py             # bfcl_run_s*/score -> accuracy-vs-sparsity figure/table/CSV
src/actsparse.py         # SparseMLP wrapper + per-token maskers
requirements-*.txt       # pinned deps for the two venvs
vulcan/                  # env.sh, setup.sh, bfcl_sweep.slurm (SLURM cluster)
results/SUMMARY_bfcl.md  # method notes + result writeup
```
