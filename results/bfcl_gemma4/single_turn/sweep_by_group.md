# BFCL accuracy vs activation sparsity — capability groups

Count-weighted accuracy (sum correct / sum total). s=0 = dense.

## simple (single call)

| method | s=0 | s=0.5 | s=0.7 | s=0.85 |
|---|---|---|---|---|
| oracle_gate | 83.29% | 83.04% | 81.81% | 58.91% |

## compositional (multi/parallel)

| method | s=0 | s=0.5 | s=0.7 | s=0.85 |
|---|---|---|---|---|
| oracle_gate | 80.92% | 80.15% | 78.38% | 30.06% |

## abstention (irrelevance)

| method | s=0 | s=0.5 | s=0.7 | s=0.85 |
|---|---|---|---|---|
| oracle_gate | 87.19% | 86.57% | 88.26% | 95.73% |

## relevance (should call)

| method | s=0 | s=0.5 | s=0.7 | s=0.85 |
|---|---|---|---|---|
| oracle_gate | 75.00% | 81.25% | 62.50% | 12.50% |

## OVERALL (all single-turn)

| method | s=0 | s=0.5 | s=0.7 | s=0.85 |
|---|---|---|---|---|
| oracle_gate | 83.36% | 82.78% | 82.12% | 56.66% |
