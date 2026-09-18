#!/usr/bin/env python3
"""Figure 9: overall performance.

Section 6.2. The five methods of Section 6.1, each over a sweep of the load,
reporting goodput and the median time per output token. Load rises along each
curve from R = 4 to R = 32 requests in flight at the provider, R over the
number of clients per client.

    python experiments/evaluation/overall.py --in-flight 4 8 16 32

The output json is what `analysis/collect.py` turns into the DATA block of
`figures/overall_perf.py`.
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
    summarize,
    write_result,
)
from slipstream.methods import METHODS


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.set_defaults(in_flight=[4, 8, 16, 32], out="results/evaluation/overall.json")
    args = parser.parse_args()

    prompts = datasets.load(
        args.workload, max(8, args.requests * args.clients),
        path=args.dataset or None,
        prompt_tokens=args.prompt_tokens, seed=args.seed,
    )
    results = []
    for method in args.methods:
        for in_flight in args.in_flight:
            spec = spec_from_arguments(args, method, in_flight)
            results.append(closed_loop(spec, prompts))
            last = results[-1]
            print(
                f"{method:16s} R={in_flight:3d} goodput={last.goodput_tokens_per_s:8.1f} tok/s "
                f"TPOT P50={last.tpot_p50_ms:7.1f} ms tau={last.tau_mean:.2f}"
            )
    print()
    print(summarize(results))
    payload = {
        "runs": [r.as_dict() for r in results],
        "workload": datasets.describe(args.workload, prompts),
    }
    print("wrote", write_result(args.out, payload))


if __name__ == "__main__":
    main()
