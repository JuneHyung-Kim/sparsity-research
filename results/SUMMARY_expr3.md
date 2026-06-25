# expr3 / Q2 — activation sparsity under weight quantization

**Figures:** `ppl_vs_sparsity_quant.png` (a) · `ranking_stability.png` (b)
**Data:** `oracle_quant.csv` · `ranking_stability.csv` · `ranking_stability_perlayer.npz` · **Date:** 2026-06-25

**One line:** The oracle activation-sparsity ceiling from expr1 was measured in
fp16. Re-measured on the same model weight-quantized to int8 and NF4 (4-bit), the
headroom is unchanged — the *relative* PPL-vs-sparsity curve is precision-invariant
— and the quantized model keeps mostly the same per-token neurons. Weight
quantization and contextual activation sparsity are orthogonal; they compose.

---

## Setup (everything needed to reproduce)

| item | value |
|---|---|
| Model | **LLaMA-2-7B** (`NousResearch/Llama-2-7b-hf`) |
| Precisions | **fp16**, **int8** (`bitsandbytes` `load_in_8bit`), **NF4** (`load_in_4bit`, nf4, double-quant, fp16 compute) |
| Quantization type | **weight-only** — activations `a` stay fp16, so the ranking signal is full precision |
| Eval dataset | **WikiText-2 raw**, test split (`Salesforce/wikitext`, `wikitext-2-raw-v1`) |
| (a) eval size | full test set, 166 segments × 2048 |
| (b) eval size | 4 segments × 2048 (kept-set overlap converges fast) |
| Metric (a) | perplexity = exp(mean token NLL) |
| Metric (b) | per-token kept-set overlap `\|K_fp ∩ K_q\| / \|K_fp\|` (Jaccard for equal-size sets), per layer & sparsity |
| Sparsity ranking | **oracle_gate** (top-(1−s) by `\|a\|`); expr1 showed `\|a\| ≈ \|a\|·‖down‖`, and bnb's packed int down_proj has no usable float column norm |
| Hardware | single RTX 4090 (24 GB); bitsandbytes 0.49.2 |

## What is measured

- **(a) Headroom preservation** — for each precision, oracle PPL vs intra-FFN
  activation sparsity (same sweep as expr1). Compared as **absolute PPL** and as
  **PPL / that precision's own dense PPL** (the curve *shape*). Coinciding relative
  curves ⇒ sparsity's cost is independent of precision ⇒ orthogonal.
- **(b) Ranking stability** — fp16 and a quantized copy are loaded together; the
  same batches pass through both; we compare *which* top-(1−s) neurons each keeps,
  per token, layer and sparsity. High overlap ⇒ a detector built on fp16 stats
  transfers to the quantized deployment. Random-choice overlap is `1−s` (floor).

## Key results

### (a) PPL vs sparsity, per precision

| sparsity | fp16 PPL | int8 PPL | NF4 PPL | rel. (fp16 / int8 / NF4) |
|---|---|---|---|---|
| dense | 5.472 | 5.505 | 5.644 | 1.000 / 1.000 / 1.000 |
| 0.5 | 5.520 | 5.545 | 5.690 | 1.009 / 1.007 / 1.008 |
| **0.7** | **5.730** | **5.759** | **5.899** | **1.047 / 1.046 / 1.045** |
| 0.8 | 6.157 | 6.184 | 6.336 | 1.125 / 1.123 / 1.123 |
| 0.9 | 8.128 | 8.132 | 8.333 | 1.485 / 1.477 / 1.476 |
| 0.95 | 15.94 | 15.89 | 16.26 | 2.913 / 2.887 / 2.881 |

- Dense quant cost: int8 **+0.6%**, NF4 **+3.1%** over fp16 — as expected.
- The **relative** columns coincide within ≈0.3 pp at every sparsity: the multiplicative
  PPL cost of dropping the bottom-s neurons is the **same at 16-, 8- and 4-bit**.
- expr1's headline survives quantization: **~70% of FFN neurons skippable per token
  at ~+4.5% PPL**, on top of whatever quantization already paid.
- No compounding — at s≥0.9 the quantized *relative* curves sit marginally **below**
  fp16 (NF4 1.476 vs fp16 1.485 at 0.9): the large sparsity error dominates, so the
  added quant error is relatively smaller, not larger.

### (b) Kept-set overlap with fp16 (mean over 32 layers)

| sparsity | int8 | NF4 | random floor (1−s) |
|---|---|---|---|
| 0.3 | 0.945 | 0.892 | 0.70 |
| 0.5 | 0.939 | 0.873 | 0.50 |
| 0.7 | 0.930 | 0.855 | 0.30 |
| 0.9 | 0.918 | 0.831 | 0.10 |

- int8 keeps **92–95%** and NF4 **83–89%** of the exact same neurons fp16 keeps —
  far above the random floor, and **uniform across all 32 layers** (early layers
  0–3 drift a little more under NF4; see the per-layer panel).
- NF4 reshuffles ~12–17% of the selected set, yet **(a) shows this costs no extra
  PPL** — the disagreeing neurons sit at the selection boundary (low contribution
  either way); the high-contribution neurons are ranked stably. A detector trained
  on fp16 activations therefore transfers to the quantized model.

## How to read it correctly (don't over-claim)

1. **Weight-only quantization.** Activations (hence the `|a|` ranking signal) stay
   fp16. The clean orthogonality here is *specific to weight-only* quant. **W8A8**
   (activation quantization) would quantize `a` itself and is the untested case
   most likely to fight sparsity — flagged as a follow-up, not answered here.
2. Still a **ceiling**, not an achieved result — oracle uses hindsight (expr1
   caveats carry over). (b) bounds *detector transfer*, it does not build a detector.
3. (b) compares kept *sets*; it does not weight by contribution. The fact that the
   set drifts ~15% but PPL doesn't move is the real point — boundary neurons.
4. FFN-only, attention dense; uniform sparsity fraction per layer.

## Reproduce

```bash
# (a) headroom across precisions  -> results/oracle_quant.csv
.venv/bin/python run_oracle_quant.py \
    --quants none int8 nf4 \
    --sparsities 0 0.3 0.5 0.6 0.7 0.8 0.9 0.95

# (b) kept-set overlap fp16 vs int8/NF4  -> results/ranking_stability.csv (+ _perlayer.npz)
.venv/bin/python run_ranking_stability.py \
    --quants int8 nf4 --sparsities 0.3 0.5 0.7 0.9 --segments 4

# figures from the CSV/npz (no GPU)
.venv/bin/python plot_expr3.py
```
