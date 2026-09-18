#!/usr/bin/env python3
"""Section 6.7: what the stream adds to a round's critical path.

Slipstream adds six decisions: on the client, ending the first segment, the
one-at-a-time socket hand-off, and the estimates between rounds; on the
provider, forming a foreground batch, its passengers, and applying a verdict
marked continue. Everything else runs while hidden states are in flight.

Every one of the six is timed separately in the running system
(`slipstream/costs.py`, `OverheadCounters`), so this driver only has to run
the system and print the counters.

    python experiments/evaluation/overhead.py --in-flight 4 32
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

LABELS = {
    "end_first_segment_us": "ending the first segment (client)",
    "socket_handoff_us": "one-at-a-time socket hand-off (client)",
    "between_round_estimates_us": "estimates between rounds (client)",
    "form_foreground_batch_us": "forming a foreground batch (provider)",
    "passenger_selection_us": "choosing passengers (provider)",
    "apply_verdict_us": "applying a verdict (provider)",
}


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(in_flight=[4, 32], out="results/evaluation/overhead.json")
    args = parser.parse_args()

    prompts = datasets.load(
        args.workload, max(8, args.requests * args.clients),
        path=args.dataset or None,
        prompt_tokens=args.prompt_tokens, seed=args.seed,
    )
    out = {"runs": {}}
    for in_flight in args.in_flight:
        result = closed_loop(spec_from_arguments(args, "slipstream", in_flight), prompts)
        tpot_us = result.tpot_p50_ms * 1e3
        rows = {}
        for key, label in LABELS.items():
            stats = result.overhead.get(key, {})
            if not stats.get("n"):
                continue
            rows[key] = {
                "label": label,
                **{k: round(v, 3) for k, v in stats.items() if k != "n"},
                "n": stats["n"],
                "p50_share_of_tpot_percent": (
                    round(100 * stats["p50_us"] / tpot_us, 4) if tpot_us else None
                ),
            }
            print(
                f"{label:42s} p50={stats['p50_us']:9.1f} us  p99={stats['p99_us']:9.1f} us  "
                f"n={stats['n']}"
            )
        out["runs"][str(in_flight)] = {
            "tpot_p50_ms": round(result.tpot_p50_ms, 3),
            "decisions": rows,
            "note": result.note,
        }
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
