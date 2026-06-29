"""Reduce a tau2-bench results.json to a compact metrics record.

Runs in the tau2 venv (.venv-tau2) -- it imports tau2 to reuse the official
metric computation (pass^k via math.comb on per-task trial successes, plus
avg_reward). Writes a small JSON tagged with the sparsity/method/domain so a
sweep of these files can be collected into a curve by benchmarks/tau2/plot.py.
"""
import argparse
import json
from pathlib import Path

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", help="path to tau2 results.json")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = Results.load(Path(args.results_json))
    metrics = compute_metrics(results)
    record = {
        "domain": args.domain,
        "sparsity": args.sparsity,
        "method": args.method,
        **metrics.as_dict(),     # avg_reward, pass_hat_<k>, total_simulations, ...
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    pass1 = record.get("pass_hat_1")
    print(f"[tau2_score] {args.domain} s={args.sparsity} "
          f"avg_reward={record.get('avg_reward'):.4f} "
          f"pass^1={pass1:.4f} ({record.get('total_simulations')} sims) -> {out}")


if __name__ == "__main__":
    main()
