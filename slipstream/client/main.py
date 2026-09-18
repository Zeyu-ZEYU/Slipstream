"""Run one client: the first layer, the head, the drafter, and the scheduler.

The engine hands hidden states to the Rust gateway over shared memory, in
stream order (Section 5). `deploy/client.sh` starts both.

    python -m slipstream.client.main --model /path/to/checkpoint \
        --shm .shm/client0 --method slipstream --requests 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from ..config import Config
from ..methods import METHODS, NAMES
from ..models.mtp import MTPDrafter
from ..models.split_model import build_tiny, build_tiny_hybrid, load_split
from ..transport.shm import ShmTransport
from .engine import ClientEngine


def build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--client-layers", type=int, default=1)
    parser.add_argument("--mtp", default="", help="path to the vendor's module")
    parser.add_argument("--shm", default=".shm/client0")
    parser.add_argument("--create-rings", action="store_true")
    parser.add_argument("--method", default="slipstream", choices=list(METHODS))
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--in-flight", type=int, default=1,
                        help="requests this client keeps in flight, K")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--workload", default="generated")
    parser.add_argument("--dataset", default="", help="a jsonl of prompts")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--compression", action="store_true",
                        help="turn stream compression on (Section 4.5)")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arguments().parse_args(argv)
    sys.path.insert(0, ".")
    from experiments import datasets  # noqa: E402  (optional, for the prompts)

    cfg = Config()
    cfg.provider.window = args.window
    cfg.decode.max_output_tokens = args.max_output_tokens
    if args.compression:
        cfg.compression.enabled = True
    if args.mtp:
        cfg.model.mtp_path = args.mtp

    if args.model == "tiny":
        builder = build_tiny_hybrid if args.hybrid else build_tiny
        client_shard, _, spec = builder()
    else:
        client_shard, _, spec = load_split(
            args.model, client_layers=args.client_layers, dtype=args.dtype, device=args.device
        )
    drafter = (
        MTPDrafter.load(client_shard, args.mtp, cfg.drafter)
        if args.mtp
        else MTPDrafter(client_shard, None, cfg.drafter)
    )
    transport = ShmTransport(
        f"{args.shm}/up.ring",
        f"{args.shm}/down.ring",
        create=args.create_rings,
        client_id=args.client_id,
    )
    engine = ClientEngine(client_shard, drafter, transport, cfg)
    method = METHODS[args.method](engine)

    prompts = datasets.load(
        args.workload,
        args.requests,
        path=args.dataset or None,
        tokenizer=args.model if args.model != "tiny" else None,
        prompt_tokens=args.prompt_tokens,
        vocab=spec.vocab_size,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "role": "client",
                "client_id": args.client_id,
                "method": NAMES[args.method],
                "model": args.model,
                "hidden_size": spec.hidden_size,
                "requests": args.requests,
                "rings": args.shm,
            }
        ),
        flush=True,
    )

    metrics = []
    for index in range(args.requests):
        result = method.run_request(
            args.client_id * 100_000 + index + 1,
            prompts[index % len(prompts)],
            args.max_output_tokens,
        )
        metrics.append(result.as_dict())
        print(json.dumps({"request": result.as_dict()}), flush=True)

    tpots = sorted(m["tpot_ms"] for m in metrics if m["tpot_ms"] > 0)
    summary = {
        "client_id": args.client_id,
        "method": NAMES[args.method],
        "requests": len(metrics),
        "output_tokens": sum(m["output_tokens"] for m in metrics),
        "tpot_p50_ms": statistics.median(tpots) if tpots else 0.0,
        "tau_mean": (
            sum(m["tau_mean"] for m in metrics) / len(metrics) if metrics else 0.0
        ),
        "engine": engine.snapshot(),
        "transport": transport.snapshot(),
    }
    print(json.dumps({"summary": summary}), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "requests": metrics}, fh, indent=2)
    transport.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
