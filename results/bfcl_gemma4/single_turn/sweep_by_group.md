# BFCL accuracy vs activation sparsity — capability groups

Count-weighted accuracy (sum correct / sum total). s=0 = dense.

## simple (single call)

| method | s=0 | s=0.5 | s=0.7 | s=0.8 | s=0.85 | s=0.9 |
|---|---|---|---|---|---|---|
| oracle_gate | 83.29% | 83.04% | 81.81% | 78.34% | 58.91% | 23.89% |

## compositional (multi/parallel)

| method | s=0 | s=0.5 | s=0.7 | s=0.8 | s=0.85 | s=0.9 |
|---|---|---|---|---|---|---|
| oracle_gate | 80.92% | 80.15% | 78.38% | 57.12% | 30.06% | 16.89% |

## abstention (irrelevance)

| method | s=0 | s=0.5 | s=0.7 | s=0.8 | s=0.85 | s=0.9 |
|---|---|---|---|---|---|---|
| oracle_gate | 87.19% | 86.57% | 88.26% | 91.73% | 95.73% | 97.95% |

## relevance (should call)

| method | s=0 | s=0.5 | s=0.7 | s=0.8 | s=0.85 | s=0.9 |
|---|---|---|---|---|---|---|
| oracle_gate | 75.00% | 81.25% | 62.50% | 37.50% | 12.50% | 0.00% |

## OVERALL (all categories)

| method | s=0 | s=0.5 | s=0.7 | s=0.8 | s=0.85 | s=0.9 |
|---|---|---|---|---|---|---|
| oracle_gate | 83.36% | 82.78% | 82.12% | 72.43% | 56.66% | 43.39% |
