#!/usr/bin/env python3
"""Figure 3 and Figure A3: only trees turn budget into tokens.

Section 3.2, third part. The budget goes from 3 up, for the chain and for
trees of width 2 and 4. The chain flattens by about six; the (6,2) tree
flattens when its 22-candidate pool runs out; the (6,4) tree, whose pool holds
84, keeps gaining.

    python experiments/motivation/budget_sweep.py --budgets 3 4 6 8 10 14 18 22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import datasets
from experiments.acceptance import build_harness
from experiments.common import RunSpec, write_result
from slipstream.tree import candidate_pool_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 4, 6, 8, 10, 14, 18, 22])
    parser.add_argument("--shapes", nargs="+", default=["chain", "6,2", "6,4"])
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--drafter", default="oracle", choices=["oracle", "mtp"])
    parser.add_argument("--hit-rate", type=float, default=0.6)
    parser.add_argument("--workloads", nargs="+", default=["math", "sharegpt"])
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=24)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/motivation/budget_sweep.json")
    args = parser.parse_args()

    spec = RunSpec(model=args.model, drafter=args.drafter, hit_rate=args.hit_rate, seed=args.seed)
    harness = build_harness(spec)
    out = {"budgets": args.budgets, "series": {}, "pools": {}}
    for text in args.shapes:
        if text != "chain":
            depth, width = (int(p) for p in text.split(","))
            out["pools"][text] = candidate_pool_size(depth, width)

    for text in args.shapes:
        shape = None if text == "chain" else tuple(int(p) for p in text.split(","))
        out["series"][text] = {}
        for budget in args.budgets:
            taus = []
            for workload in args.workloads:
                prompts = datasets.load(
                    workload, args.requests, prompt_tokens=args.prompt_tokens, seed=args.seed
                )
                taus.append(
                    harness.measure(prompts, shape, budget, args.max_output_tokens).tau_mean
                )
            mean = sum(taus) / len(taus)
            out["series"][text][str(budget)] = round(mean, 4)
            print(f"{text:6s} m={budget:3d} tau={mean:.3f}")
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
