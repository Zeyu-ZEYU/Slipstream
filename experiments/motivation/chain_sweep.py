#!/usr/bin/env python3
"""Figure 2: the chain saturates early.

Section 3.2, first observation. Chain drafting with the drafter proposing `m`
tokens as one sequence, over a range of budgets; what is recorded is the
acceptance length on each workload.

    python experiments/motivation/chain_sweep.py --budgets 1 2 3 4 5 6 7 8

The output json has the shape `figures/data/acceptance_vs_draft.json` expects,
so `python figures/acceptance_vs_draft.py --save` redraws the figure from a
fresh run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import datasets
from experiments.acceptance import build_harness
from experiments.common import RunSpec, write_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--drafter", default="oracle", choices=["oracle", "mtp"])
    parser.add_argument("--hit-rate", type=float, default=0.6)
    parser.add_argument("--workloads", nargs="+", default=["math", "sharegpt"])
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=24)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/motivation/chain_sweep.json")
    args = parser.parse_args()

    spec = RunSpec(model=args.model, drafter=args.drafter, hit_rate=args.hit_rate, seed=args.seed)
    harness = build_harness(spec)
    out = {"drafts": args.budgets, "tau": {}, "note": "chain drafting, one guess per position"}
    for workload in args.workloads:
        prompts = datasets.load(
            workload, args.requests, prompt_tokens=args.prompt_tokens, seed=args.seed
        )
        out["tau"][workload] = {}
        for budget in args.budgets:
            result = harness.measure(prompts, None, budget, args.max_output_tokens)
            out["tau"][workload][str(budget)] = round(result.tau_mean, 4)
            print(f"{workload:14s} m={budget:3d} tau={result.tau_mean:.3f} rounds={result.rounds}")
    out["source"] = {"model": args.model, "drafter": args.drafter, "hit_rate": args.hit_rate}
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
