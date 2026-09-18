#!/usr/bin/env python3
"""Table 2: stream compression.

Section 6.4 measures what compression buys, and whether placing the bits by
acceptance probability beats compressing everything:

  Uncompressed    every hidden state at full precision, the default
  All-FP8         every hidden state at FP8, half the bytes
  Slipstream FP4  background tokens at FP4, the first segment at FP8, the root
                  exact, with admission pricing background tokens at their
                  compressed cost

Task scores need the real models and the real request sets; what this driver
reports is goodput and the codec's own error, which is the part that does not
depend on the checkpoint.

    python experiments/evaluation/stream_compression.py --in-flight 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import datasets
from experiments.common import (
    add_common_arguments,
    closed_loop,
    spec_from_arguments,
    write_result,
)
from slipstream.codec import payload_bytes, quantization_error

ARMS = {
    "Uncompressed": {},
    "All-FP8": {
        "compression.enabled": True,
        "compression.root_precision": "fp8",
        "compression.first_segment_precision": "fp8",
        "compression.background_precision": "fp8",
    },
    "Slipstream FP4": {
        "compression.enabled": True,
        "compression.root_precision": "bf16",
        "compression.first_segment_precision": "fp8",
        "compression.background_precision": "fp4",
    },
}


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(in_flight=[32], out="results/evaluation/compression.json")
    args = parser.parse_args()

    prompts = datasets.load(
        args.workload, max(8, args.requests * args.clients),
        path=args.dataset or None,
        prompt_tokens=args.prompt_tokens, seed=args.seed,
    )
    torch.manual_seed(args.seed)
    sample = torch.randn(4096)
    out = {
        "arms": {},
        "codec": {
            precision: {
                "bytes_per_hidden_state": payload_bytes(4096, precision),
                "relative_element_error": round(quantization_error(sample, precision), 4),
            }
            for precision in ("bf16", "fp8", "fp4")
        },
    }
    for in_flight in args.in_flight:
        baseline = None
        for name, overrides in ARMS.items():
            spec = spec_from_arguments(
                args, "slipstream", in_flight, overrides=dict(overrides), label=name
            )
            result = closed_loop(spec, prompts)
            if baseline is None:
                baseline = result.goodput_tokens_per_s
            lift = 100 * (result.goodput_tokens_per_s / baseline - 1) if baseline else 0.0
            out["arms"].setdefault(name, {})[str(in_flight)] = {
                "goodput_tokens_per_s": round(result.goodput_tokens_per_s, 2),
                "lift_over_uncompressed_percent": round(lift, 2),
                "bytes_up": result.bytes_up,
                "tpot_p50_ms": round(result.tpot_p50_ms, 2),
            }
            print(
                f"{name:16s} R={in_flight:3d} goodput={result.goodput_tokens_per_s:8.1f} "
                f"({lift:+5.1f}%) bytes up={result.bytes_up}"
            )
    for precision, stats in out["codec"].items():
        print(
            f"{precision:5s} {stats['bytes_per_hidden_state']:5d} bytes per hidden state, "
            f"relative element error {stats['relative_element_error']:.3f}"
        )
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
