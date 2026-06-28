# BFCL (function-calling) as a downstream activation-sparsity metric

Replaces WikiText PPL with a downstream agentic-capability score. The same
per-token FFN masker (`src/actsparse.py`) is applied to a served HF model;
BFCL V4 scores its function-calling, so sparsity can be swept exactly like the
PPL experiments but read out as task accuracy.

## Setup

- **Model:** `Qwen/Qwen3-8B` (dense SwiGLU; masker is drop-in — same
  gate/up/down_proj + SiLU as LLaMA). bf16 on the 4090.
- **Serving:** `bfcl_server.py` — a minimal OpenAI `/v1/completions` server in
  the research `.venv`. `install_sparse_mlps` wraps every block's `.mlp`; the
  masker is fixed at startup (`--method/--sparsity`). Qwen3 thinking is disabled
  (empty `<think></think>`) for speed/parse-stability.
- **Harness:** `bfcl-eval` in a separate `.venv-bfcl` (its `vllm==0.8.5` pin
  would clobber the research torch 2.12). Run with `--skip-server-setup` +
  `--local-model-path <snapshot>` so BFCL renders the FC prompt and parses
  `<tool_call>` itself and our masker is actually in the generation path
  (vLLM/sglang would bypass it). Model id `Qwen/Qwen3-8B-FC`.
- **Why these two categories:** `simple_python` = a single straightforward
  AST call (robust capability); `irrelevance` = correctly *declining* to call
  any function (nuanced capability). No external API keys needed.
- Reproduce: `./run_bfcl.sh <sparsity> oracle_gate simple_python,irrelevance`.

## Result — oracle_gate per-token FFN sparsity

| Category | dense (s=0) | s=0.5 | s=0.8 |
|---|---|---|---|
| simple_python (Python Simple AST) | 95.50% | 96.25% | 95.25% |
| irrelevance detection | 86.25% | 85.00% | 76.25% |

(oracle_gate = keep each token's top-(1−s) FFN neurons by \|a_i\|, zero the rest.)

## Read

- The pipeline works end-to-end and the masker is genuinely applied: scores
  move with sparsity.
- **Differential degradation.** A single easy function call survives even 80%
  oracle FFN sparsity (~95% throughout), while abstention (irrelevance) erodes
  86 → 85 → 76 as sparsity rises. The harder/more nuanced behavior breaks first.
- This is the payoff over PPL: a single scalar can't say *which* capability a
  given sparsity level costs. At s=0.5 the oracle headroom is essentially free
  here (consistent with the PPL curves); the cost only shows up at high sparsity
  and lands on the nuanced category.

## Caveats / next

- Two categories only; `oracle_gate` is the *ceiling* (needs all activations,
  no speedup) — a deployable predictor would sit below it.
- Natural extensions: the rest of `non_live`, more sparsity points, and the
  deployable `oracle_gateonly` masker to see the realizable curve. `multi_turn`
  (longer-horizon) likely degrades earlier than single-call AST.
