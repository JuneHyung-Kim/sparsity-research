# Downstream benchmarks under contextual activation sparsity

Measure how **token-dependent FFN activation sparsity** affects a model's
downstream ability, instead of perplexity. Two downstream metrics share the same
masker:

- **BFCL V4** (Berkeley Function Calling Leaderboard) — single-shot
  **function-calling** accuracy. `benchmarks/bfcl/`.
- **tau2-bench** — **agentic** tool-use where the model converses with a
  simulated user and acts against a domain DB over many turns (retail / airline /
  telecom), scored by `pass^k` task completion. `benchmarks/tau2/`.

Each benchmark is self-contained under `benchmarks/<name>/` (server + run + plot)
and shares the one masker in `src/actsparse.py`.

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
- **Serving:** `benchmarks/bfcl/server.py` — a minimal OpenAI `/v1/completions`
  server in the research venv. Weights load once; the masker is fixed at startup
  (`--method/--sparsity`). Qwen3 "thinking" is disabled for parse-stability.
- **Harness:** `bfcl-eval` in a separate venv (its `[oss]`/vllm extra would
  clobber the research torch). Run with `--skip-server-setup` so BFCL renders
  the FC prompt and parses `<tool_call>` itself and our masker is actually in
  the generation path (vLLM/sglang would bypass it).

## Setup (two venvs)

```bash
uv venv --python 3.12 .venv      && uv pip install -r requirements-research.txt          # masker + server + plot
uv venv --python 3.12 .venv-bfcl && uv pip install -r benchmarks/bfcl/requirements.txt    # bfcl CLI only
```

## Run

One sparsity point (serves the model, scores the categories, tears down). Run from
the repo root:

```bash
./benchmarks/bfcl/run.sh <sparsity> [method] [categories]
# e.g. dense baseline over all key-free function-calling categories:
./benchmarks/bfcl/run.sh 0.0 oracle_gate single_turn,multi_turn
```

A point writes `bfcl_run_s<NN>/score/.../BFCL_v4_<category>_score.json`. After a
sweep, collect them into a figure + table + CSV:

```bash
.venv/bin/python benchmarks/bfcl/plot.py --runs-dir .
# -> results/bfcl_acc_vs_sparsity.{png,md,csv}   (accuracy vs sparsity, per category)
```

### Categories

`single_turn` (= `non_live` AST + `live`) and `multi_turn` are key-free and
scored. `web_search` needs a paid SerpAPI key; `memory` needs agentic backends;
`format_sensitivity` is non-scoring — all three are excluded by default.

## tau2-bench (agentic tool + user)

tau2-bench talks the OpenAI **chat + tools** contract (via LiteLLM), so the
masker has to be served behind a chat endpoint that renders the prompt and parses
tool calls itself — `benchmarks/tau2/server.py` does this (Qwen3 chat template with `tools=`,
`<tool_call>` → OpenAI `tool_calls`, `<think>` stripped). It also serves **two
roles from one model**: the **agent** (the policy under test, masker applied) and
the **user simulator** (part of the environment, always dense), routed by the
request's model name. The user-sim is the same dense Qwen3-8B held fixed across a
sweep, so the only thing that varies between sparsity points is the agent's
masker. The benchmark environment (domain tools + DB) runs locally — no API keys.

Setup adds a third venv with the tau2 CLI; tau2's upstream repo is cloned because
it holds both the package and the domain data (`TAU2_DATA_DIR`):

```bash
git clone https://github.com/sierra-research/tau2-bench    # holds pkg + data/
uv venv --python 3.12 .venv-tau2 && uv pip install -e ./tau2-bench
```

Run one sparsity point on a domain (serves the model for both roles, runs the
tasks, scores `pass^k`/`avg_reward`, tears down):

```bash
./benchmarks/tau2/run.sh <sparsity> [method] [domain]
# e.g. dense retail baseline:
./benchmarks/tau2/run.sh 0.0 oracle_gate retail
```

A point writes `tau2_run_s<NN>/<domain>.json` (the compact metrics record). Knobs
mirror the bfcl run: `TRIALS` (raise for `pass^k`), `NTASKS` (subset),
`CONC` (concurrency / server batch), `THINK`, `FRESH=1` (wipe stale sims). After
a sweep, collect into a figure + table + CSV:

```bash
.venv/bin/python benchmarks/tau2/plot.py --runs-dir .   # -> results/tau2_passk_vs_sparsity.{png,md,csv}
```

`retail` (114 tasks) has the most headroom for an 8B agent; `airline` (50) and
`telecom` are harder. tau2 logs a cosmetic `model isn't mapped` ERROR per call
(LiteLLM cost lookup for our local model id) — harmless; the run still scores.

## On a SLURM cluster (Vulcan / DRAC, L40S 46 GB)

Login node has internet but no GPU; compute nodes have GPUs but no internet, so
the venvs and the model are prepared on the login node and the sweep runs
offline under SLURM. Clone into your own dir under the project space
(`~/projects/aip-nanditav/sankeert/<you>/`):

```bash
source vulcan/env.sh        # repo venv paths + scratch HF cache, uv, SLURM account
bash vulcan/setup.sh        # LOGIN node: build the venvs (+ clone tau2) + Qwen3-8B
sbatch vulcan/bfcl_sweep.slurm   # one L40S, offline; function-calling sweep
sbatch vulcan/tau2_sweep.slurm   # one L40S, offline; agentic tool+user sweep
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
src/actsparse.py             # SparseMLP wrapper + per-token maskers (shared masker)
requirements-research.txt    # pinned deps for the shared research venv (.venv)

benchmarks/bfcl/
  server.py                  # OpenAI /v1/completions server with the masker installed
  run.sh                     # serve at one sparsity, then bfcl generate + evaluate
  plot.py                    # bfcl_run_s*/score -> accuracy-vs-sparsity figure/table/CSV
  requirements.txt           # pinned deps for the bfcl CLI venv (.venv-bfcl)

benchmarks/tau2/
  server.py                  # OpenAI /v1/chat/completions (tools) server; agent+user roles
  run.sh                     # serve at one sparsity, then tau2 run + score (pass^k)
  score.py                   # tau2 results.json -> compact metrics record (runs in tau2 venv)
  plot.py                    # tau2_run_s*/<domain>.json -> pass^k-vs-sparsity figure/table/CSV
                             # (tau2 CLI deps come from the cloned tau2-bench repo)

vulcan/                      # env.sh, setup.sh, bfcl_sweep.slurm, tau2_sweep.slurm (SLURM)
results/SUMMARY_bfcl.md      # method notes + result writeup
```
