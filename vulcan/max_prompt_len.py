#!/usr/bin/env python3
"""Report the max input/output/total token lengths actually seen per BFCL category.

Sizing the server --batch needs the real worst-case sequence length the KV cache must
hold = input + generated tokens, which is dominated by multi_turn_long_context. Rather
than guess, this scans existing BFCL *_result.json files (each record carries per-step
input_token_count / output_token_count) and reports, per category, the max input, max
output, and the seqlen to guarantee = max_input + the new max_new cap.

The INPUT length is independent of max_new, so a COMPLETED prior run still reports the
real prompt sizes even if its outputs were capped lower (e.g. surviving bfcl_run_s50..
from the old THINK=0/1024 sweep). No GPU needed -- pure JSON parsing.

Usage (run on the login node, from the repo root):
    python vulcan/max_prompt_len.py                       # scans $BFCL_RUN_BASE or $PWD
    python vulcan/max_prompt_len.py --runs-dir /path --max-new 4096
"""
import argparse
import glob
import json
import os


def flat(x):
    if isinstance(x, list):
        for v in x:
            yield from flat(v)
    elif isinstance(x, (int, float)):
        yield int(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.environ.get("BFCL_RUN_BASE", "."))
    ap.add_argument("--max-new", type=int, default=4096,
                    help="the cap the upcoming run will use; seqlen-to-guarantee = max_in + this")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.runs_dir, "bfcl_run_s*", "result", "**",
                                   "*_result.json"), recursive=True)
    if not files:
        print(f"no *_result.json under {args.runs_dir}/bfcl_run_s*/result/ "
              f"(point --runs-dir at the sweep output base)")
        return

    agg = {}  # category -> [max_in, max_out]
    for f in files:
        cat = os.path.basename(f).replace("BFCL_v4_", "").replace("_result.json", "")
        cur = agg.setdefault(cat, [0, 0])
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ins = list(flat(rec.get("input_token_count", [])))
                outs = list(flat(rec.get("output_token_count", [])))
                if ins:
                    cur[0] = max(cur[0], max(ins))
                if outs:
                    cur[1] = max(cur[1], max(outs))

    print(f"scanned {len(files)} result file(s) under {args.runs_dir}\n")
    print(f"{'category':<30}{'max_in':>9}{'max_out':>9}{'guarantee':>11}")
    print(f"{'':<30}{'':>9}{'(seen)':>9}{'in+'+str(args.max_new):>11}")
    worst = 0
    for cat in sorted(agg):
        mi, mo = agg[cat]
        guarantee = mi + args.max_new
        worst = max(worst, guarantee)
        print(f"{cat:<30}{mi:>9}{mo:>9}{guarantee:>11}")
    print(f"\nseqlen to guarantee (max_in + {args.max_new}) across all categories: "
          f"{worst} tokens")
    print("-> probe the server batch at this seqlen; pick the largest batch that fits.")


if __name__ == "__main__":
    main()
