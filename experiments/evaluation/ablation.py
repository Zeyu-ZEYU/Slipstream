#!/usr/bin/env python3
"""Figure 10: the ablation.

Section 6.3 removes one mechanism at a time and reports the goodput lost
against the full system. Every variant is a setting, not a separate code path:

  First segment only  scheduler.background = False
  No continuation     scheduler.continuation = False
  Level order         scheduler.order = "level"
  Always eager        provider.policy = "eager"
  Always lazy         provider.policy = "lazy"
  Admit all           scheduler.admission = "all"
  No priorities       scheduler.priorities = False

    python experiments/evaluation/ablation.py --in-flight 4 32
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

VARIANTS = {
    "First segment only": {"scheduler.background": False},
    "No continuation": {"scheduler.continuation": False},
    "Level order": {"scheduler.order": "level"},
    "Always eager": {"provider.policy": "eager"},
    "Always lazy": {"provider.policy": "lazy"},
    "Admit all": {"scheduler.admission": "all"},
    "No priorities": {"scheduler.priorities": False},
}


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.set_defaults(in_flight=[4, 32], out="results/evaluation/ablation.json")
    args = parser.parse_args()

    prompts = datasets.load(
        args.workload, max(8, args.requests * args.clients),
        path=args.dataset or None,
        prompt_tokens=args.prompt_tokens, seed=args.seed,
    )
    out = {"losses": {}, "full": {}}
    for in_flight in args.in_flight:
        full = closed_loop(spec_from_arguments(args, "slipstream", in_flight), prompts)
        out["full"][str(in_flight)] = full.goodput_tokens_per_s
        print(f"Slipstream          R={in_flight:3d} goodput={full.goodput_tokens_per_s:8.1f} tok/s")
        for name in args.variants:
            spec = spec_from_arguments(
                args, "slipstream", in_flight, overrides=dict(VARIANTS[name]), label=name
            )
            result = closed_loop(spec, prompts)
            loss = 0.0
            if full.goodput_tokens_per_s > 0:
                loss = 100 * (1 - result.goodput_tokens_per_s / full.goodput_tokens_per_s)
            out["losses"].setdefault(name, {})[str(in_flight)] = round(loss, 2)
            print(
                f"  {name:18s} R={in_flight:3d} goodput={result.goodput_tokens_per_s:8.1f} "
                f"loss={loss:6.1f}%"
            )
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
