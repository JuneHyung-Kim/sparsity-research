# BFCL accuracy vs activation sparsity — capability groups

Gemma-4-12B, bf16, THINK=1, oracle_gate, 50 random cases/subcategory (seed 0).
Vulcan L40S run; results reported by the user (Vulcan is no-push -- recorded here
from the reported numbers; raw run artifacts + sweep_scores.csv/png stay on Vulcan).
Count-weighted accuracy (sum correct / sum total). s=0 = dense.

## multi_turn

| method | s=0 | s=0.5 | s=0.7 | s=0.8 | s=0.9 |
|---|---|---|---|---|---|
| oracle_gate | 59.50% | 62.00% | 57.00% | 48.50% | 2.00% |

## memory

| method | s=0 | s=0.5 | s=0.7 | s=0.8 | s=0.9 |
|---|---|---|---|---|---|
| oracle_gate | 10.26% | 9.40% | 10.26% | 10.26% | 0.00% |

## OVERALL (all categories)

| method | s=0 | s=0.5 | s=0.7 | s=0.8 | s=0.9 |
|---|---|---|---|---|---|
| oracle_gate | 41.32% | 42.59% | 39.75% | 34.38% | 1.26% |

## Read

- multi_turn: near-free to s=0.7 (57-62%, within the ~7%/subcat noise of dense 59.5),
  gentle at 0.8 (48.5), total collapse at 0.9 (2.0). Cliff between 0.8 and 0.9.
- memory: flat ~10% through 0.8 then 0 at 0.9 -- but dense is only 10.26%, i.e. the
  model is already weak at memory (little to lose). This curve reflects dense weakness,
  NOT sparsity tolerance.
- s=0.9 breaks everything (model effectively non-functional).
