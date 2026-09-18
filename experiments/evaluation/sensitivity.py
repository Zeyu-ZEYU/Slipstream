#!/usr/bin/env python3
"""Figure A4 and Section 6.5: the two settings chosen by hand.

The window `W`, the one capacity the provider sets, and the children per step
`k`, the one drafter setting the design leaves open.

    python experiments/evaluation/sensitivity.py --windows 0 4 8 16 24 32
    python experiments/evaluation/sensitivity.py --children 2 4 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import datasets
from experiments.common import (
    add_common_arguments,
    closed_loop,
    spec_from_arguments,
    write_result,
)


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--windows", type=int, nargs="+", default=[0, 4, 8, 16, 24, 32])
    parser.add_argument("--children", type=int, nargs="+", default=[2, 4, 8])
    parser.set_defaults(in_flight=[4, 32], out="results/evaluation/sensitivity.json")
    args = parser.parse_args()

    prompts = datasets.load(
        args.workload, max(8, args.requests * args.clients),
        path=args.dataset or None,
        prompt_tokens=args.prompt_tokens, seed=args.seed,
    )
    out = {"window": {}, "children_per_step": {}}
    for in_flight in args.in_flight:
        for window in args.windows:
            spec = spec_from_arguments(
                args, "slipstream", in_flight,
                overrides={"provider.window": window}, label=f"W={window}",
            )
            result = closed_loop(spec, prompts)
            out["window"].setdefault(str(in_flight), {})[str(window)] = round(
                result.goodput_tokens_per_s, 2
            )
            print(
                f"R={in_flight:3d} W={window:3d} goodput={result.goodput_tokens_per_s:8.1f} "
                f"tau={result.tau_mean:.2f} s={result.first_segment_mean:.1f}"
            )
        for children in args.children:
            spec = spec_from_arguments(
                args, "slipstream", in_flight,
                overrides={"drafter.children_per_step": children}, label=f"k={children}",
            )
            result = closed_loop(spec, prompts)
            out["children_per_step"].setdefault(str(in_flight), {})[str(children)] = round(
                result.goodput_tokens_per_s, 2
            )
            print(
                f"R={in_flight:3d} k={children:3d} goodput={result.goodput_tokens_per_s:8.1f} "
                f"tau={result.tau_mean:.2f}"
            )
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
