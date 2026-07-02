# tau2-bench (agentic, multi-turn) vs activation sparsity — Gemma-4-12B

Downstream agentic metric for per-token FFN activation sparsity: the agent LLM is
Gemma-4-12B-it served by vLLM with the `oracle_gate` masker (keep each token's
top-(1−s) FFN neurons by |gelu(gate)·up|, zero the rest); the user simulator and
the NL-assertion judge are the SAME model, dense, on a separate engine. The
measurement is the dense→sparse **delta** under this fixed all-local harness.

## Setup

- **Model:** google/gemma-4-12B-it, bf16, vLLM (native gemma4 tool-call +
  reasoning parsers, no proxy). Agent-only thinking ON (`THINK=1`), per-turn cap
  `MAX_NEW=4096`, `MAXLEN=32768`, per-sim `TIMEOUT=1200`, temperature 0, seed 300.
- **Sparse topology:** s=0 → one engine (all roles); s>0 → two engines
  (agent=masked, user-sim+judge=dense), roles routed by per-call api_base.
- **Tasks:** 3 domains × fixed 40-task random subset (seed=0, frozen in
  `benchmarks/tau2/subsets/*_40.txt`), the SAME subset at every sparsity point
  (paired). 1 trial. Run as a SLURM job array on Vulcan L40S
  (`vulcan/tau2_vllm_array.slurm`), one array task per (domain, sparsity) point.
- **Not comparable to the model card's 69.0%**: that number is a 3-domain average
  under the standard harness (gpt-4.1 user-sim + gpt-4.1/4o-mini judges); ours is
  all-local (weaker Gemma user-sim dominates the absolute gap). The judge is the
  same dense engine at every point, so the delta is internally consistent.

## Result — pass^1 vs oracle_gate sparsity (40 tasks/point, 1 trial)

| sparsity | airline | retail | telecom |
|---|---|---|---|
| 0.00 | 0.550 | 0.475 | 0.225 |
| 0.50 | 0.550 | 0.675 | 0.250 |
| 0.70 | 0.500 | 0.600 | 0.200 |
| 0.80 | 0.600 | 0.400 | 0.200 |

Per-point SE ≈ ±8pp (N=40, binary). Termination audit: no timeout /
context-window deaths anywhere (max sim 903s < 1200; the earlier MAXLEN=16384
overflow was fixed in 760a2e8 before this sweep's final runs).

## Read

- **airline & retail: no measurable pass^1 loss up to s=0.8.** Consistent with
  the PPL result (quant⟂sparsity, composes to ~0.8) and BFCL simple_python
  (flat to 0.8): the oracle_gate headroom carries to long-horizon agentic tasks.
- **retail s=0.5/0.7 sit ABOVE dense** (+20pp / +12.5pp). Paired per-task diff
  dense→s0.5: 10 improved vs 2 worsened (McNemar p≈0.04, but post-hoc-selected
  from 9 deltas → treat as unverified). Claim only "no loss"; the apparent gain
  needs a different-seed replication before it means anything.
- **telecom: flat ~0.20–0.25 but read with care.** (a) Absolute floor — the 12B
  model is weak on this dual-control domain regardless of sparsity. (b)
  `infrastructure_error` count RISES with sparsity (5 → 4 → 7 → 9 of 40) and tau2
  EXCLUDES those sims from pass^1, so the telecom cells average over only 31–36
  sims and may be optimistically biased if infra failures hit harder tasks.
  The infra-vs-sparsity trend itself is an unexplained signal (cause not yet
  investigated; dense also has 5, so part is baseline harness friction).
- telecom's reward is ENV_ASSERTION — deterministic code checks, no LLM judge —
  so its numbers are immune to the local-judge concerns that apply to
  retail (DB+NL) and airline (COMMUNICATE+DB).

## Caveats / next

- 40-task subsets, 1 trial: fine for the paired delta shape, not for absolute
  claims. Confirmation run = full task sets (114/50/114) and/or trials>1 at the
  interesting points.
- Open items: (1) identify the telecom infrastructure_error cause (grep the
  array logs); (2) different-seed replication of retail dense + s0.5 (e.g.
  SEED=301) to settle the apparent gain; (3) oracle_gate is the oracle ceiling —
  a deployable predictor sits below it.
- Raw data: `archive/tau2_sweep_<date>_<commit>/` on Vulcan — 12×
  `results.json` (full conversations + per-task rewards + termination reasons),
  per-point metrics, engine + SLURM logs. Everything in this file is
  recomputable from those.
