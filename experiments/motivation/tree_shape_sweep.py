#!/usr/bin/env python3
"""Figure A2: at a fixed budget, shape is secondary.

Section 3.2, second observation. The same budget `m=8` spent on a chain and on
six tree shapes, depths 6 and 8 and widths 2 to 4. Every tree beats the chain;
across the shapes the acceptance length barely moves, because the budget
binds, not the shape.

    python experiments/motivation/tree_shape_sweep.py --budget 8
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

SHAPES = [(6, 2), (6, 3), (6, 4), (8, 2), (8, 3), (8, 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--shapes", nargs="+", default=[f"{d},{w}" for d, w in SHAPES])
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--drafter", default="oracle", choices=["oracle", "mtp"])
    parser.add_argument("--hit-rate", type=float, default=0.6)
    parser.add_argument("--workloads", nargs="+", default=["math", "sharegpt"])
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=24)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/motivation/tree_shape_sweep.json")
    args = parser.parse_args()

    shapes = [tuple(int(part) for part in text.split(",")) for text in args.shapes]
    spec = RunSpec(model=args.model, drafter=args.drafter, hit_rate=args.hit_rate, seed=args.seed)
    harness = build_harness(spec)
    out = {"chain": {}, "shapes": {}, "budget": args.budget, "pools": {}}
    for depth, width in shapes:
        out["pools"][f"({depth},{width})"] = candidate_pool_size(depth, width)

    for workload in args.workloads:
        prompts = datasets.load(
            workload, args.requests, prompt_tokens=args.prompt_tokens, seed=args.seed
        )
        chain = harness.measure(prompts, None, args.budget, args.max_output_tokens)
        out["chain"][workload] = round(chain.tau_mean, 4)
        print(f"{workload:14s} chain tau={chain.tau_mean:.3f}")
        for depth, width in shapes:
            key = f"({depth},{width})"
            result = harness.measure(prompts, (depth, width), args.budget, args.max_output_tokens)
            out["shapes"].setdefault(key, {})[workload] = round(result.tau_mean, 4)
            print(
                f"{workload:14s} {key:7s} tau={result.tau_mean:.3f} "
                f"lead={result.tau_mean - chain.tau_mean:+.3f} pool={out['pools'][key]}"
            )
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
