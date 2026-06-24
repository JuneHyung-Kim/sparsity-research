# Contextual activation sparsity — oracle ceiling

**Figure:** `ppl_vs_actsparsity.png`  ·  **Data:** `oracle_llama2-7b.csv`  ·  **Date:** 2026-06-24

**One line:** How much does WikiText-2 perplexity rise as we skip more FFN
neurons *per token*, if we always skip the right ones? Answer: a *perfectly
informed* selector can drop ~70% of FFN neurons per token for only +5% PPL.

---

## Setup (everything needed to reproduce)

| item | value |
|---|---|
| Model | **LLaMA-2-7B** (`NousResearch/Llama-2-7b-hf`, ungated mirror of `meta-llama/Llama-2-7b-hf`) |
| Precision | fp16 |
| Eval dataset | **WikiText-2 raw**, test split (`Salesforce/wikitext`, `wikitext-2-raw-v1`) |
| Eval size | 341,469 tokens → **166 segments × 2048** (seqlen 2048) |
| Metric | perplexity = exp(mean token NLL), SparseGPT/Wanda averaging convention |
| Hardware / time | single RTX 4090 (24 GB); full sweep ≈ 20 min |
| Sparsity type | **contextual ACTIVATION sparsity** (dynamic, per-token) — **NOT** weight pruning |

## What "sparsity" means here

LLaMA FFN (SwiGLU), per token `x`, intermediate dim 11008:
`a_i = SiLU(gate(x)_i)·up(x)_i`, output `= Σ_i a_i·down[:,i]`.
**Sparsity s = fraction of the 11008 neurons whose `a_i` is zeroed before
`down_proj`, chosen independently for every token.**

- Applied to **all 32 decoder layers**, **uniform fraction** per layer.
- **FFN only — attention is left fully dense.** "s = 0.7" = 70% of FFN neurons skipped, 0% of attention.
- The *set* skipped differs per token & per layer; only the *fraction* is uniform.

## The two curves (both are "oracle")

| curve | neuron ranking | meaning |
|---|---|---|
| `top-k \|a\|` (magnitude) | by `\|a_i\|` | keep each token's largest-activation neurons |
| `top-k \|a\|·‖down‖` (contribution) | by `\|a_i\|·‖down[:,i]‖` | keep largest true output-contribution |

"Oracle" = uses the **true, post-computation activations (hindsight)** to choose
— information a real predictor would not have. It bounds what *perfect detection*
could achieve.

## Key results

| sparsity | top-k `\|a\|` | top-k `\|a\|·‖down‖` |
|---|---|---|
| dense | 5.47 | 5.47 |
| 0.5 | 5.52 | 5.52 |
| 0.7 | **5.73** | 5.76 |
| 0.8 | **6.16** | 6.24 |
| 0.9 | **8.13** | 8.27 |

- **~70% of FFN neurons are skippable per token at +5% PPL** (5.47 → 5.73) — *if perfectly selected*.
- For context, **random** per-token skipping (not shown) already collapses by 20–30% (PPL 13 → 43 → diverges) — a 2–3 orders-of-magnitude gap to the oracle. **That gap is the room a good detector could win.**
- The two rankings nearly coincide; the simpler `|a|` is *as good or slightly better* at high sparsity → on this model, weighting by `‖down‖` does not help.
- Dense PPL 5.47 matches the published LLaMA-2-7B WikiText-2 number → pipeline validated.
- Companion figure `sparsity_headroom.png`: 90% of a token's contribution mass sits in ~53% of neurons (per-layer concentration varies).

## ⚠️ How to read it correctly (don't over-claim)

1. **This is a ceiling, not an achieved result.** The oracle cheats with
   hindsight; a real (cheap, predict-ahead) detector would land *below* these
   curves. The research question is how close it can get.
2. **No speedup is implemented or measured.** We zero `a_i` but still run the
   full FFN — this measures *accuracy headroom* only. Real savings require
   skipping `gate/up/down` compute & weight-loading via prediction.
3. **Not provably optimal.** Top-k-by-magnitude is a greedy heuristic; the true
   minimal-PPL subset per token is combinatorial/unknown. (Evidence: neither
   ranking dominates.)
4. **Soft sparsity by design.** SwiGLU has no exact zeros (unlike the ReLU
   models Deja Vu / PowerInfer rely on). Targeting SwiGLU is the intended
   novelty — we are *not* reproducing prior ReLU-based work.
5. FFN-only, attention dense; sparsity fraction uniform across layers (per-layer
   adaptive budgets are an untested lever).

## Reproduce

```bash
.venv/bin/python run_oracle.py --model NousResearch/Llama-2-7b-hf \
    --methods random oracle_gate oracle_contrib \
    --sparsities 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --out results/oracle_llama2-7b.csv
.venv/bin/python plot.py --csv results/oracle_llama2-7b.csv --exclude random --linear-y
```
