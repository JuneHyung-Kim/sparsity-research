#!/usr/bin/env python3
"""Freeze a fixed random task subset per tau2 domain for the sparsity sweep.

The sweep compares dense vs sparse on the SAME tasks (paired), so the subset must
be identical at every sparsity point -- that pairing is what makes the dense->sparse
delta readable at small N. tau2's --num-tasks just takes the FIRST N tasks
(runner/helpers.py: `tasks[:num_tasks]`), NOT a random sample, so we draw the ids
here with a fixed seed and pass them via --task-ids (run_vllm.sh reads this file).

Ids are drawn from exactly what the runner loads for a plain `--domain <d>` run:
batch.py does `task_set_name = config.task_set_name or config.domain` with the
default "base" split, i.e. get_tasks(task_set_name=<domain>, task_split_name="base").

Run in the tau2 venv (imports tau2); no GPU/network needed (local json only):
    .venv-tau2/bin/python benchmarks/tau2/make_subsets.py
Writes benchmarks/tau2/subsets/<domain>_<n>.txt, one task id per line. Deterministic:
same --seed -> same subset. Generate on the login node (or locally) and sync the
files, since compute nodes are offline.
"""
import argparse
import random
from pathlib import Path

from tau2.runner.helpers import get_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["retail", "airline", "telecom"])
    ap.add_argument("--split", default="base")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "subsets"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dom in args.domains:
        tasks = get_tasks(task_set_name=dom, task_split_name=args.split)
        ids = sorted(t.id for t in tasks)                 # deterministic input order
        k = min(args.n, len(ids))
        sample = sorted(random.Random(args.seed).sample(ids, k))
        out = out_dir / f"{dom}_{args.n}.txt"
        out.write_text("\n".join(sample) + "\n")
        print(f"[make_subsets] {dom}: base={len(ids)} -> sampled {k} "
              f"(seed={args.seed}) -> {out}")


if __name__ == "__main__":
    main()
