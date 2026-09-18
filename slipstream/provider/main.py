"""Run the provider: the middle layers, the two queues, and the window.

The engine owns the GPU and the shared-memory rings; the Rust gateway owns the
socket (Section 5). `deploy/provider.sh` starts both.

    python -m slipstream.provider.main --model /path/to/checkpoint --shm .shm/provider
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

from ..config import Config
from ..models.split_model import build_tiny, build_tiny_hybrid, load_split
from ..transport.shm import ShmProviderBridge
from .engine import ProviderEngine


def build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny", help="'tiny' or a checkpoint path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--client-layers", type=int, default=1)
    parser.add_argument("--shm", default=".shm/provider", help="directory of the two rings")
    parser.add_argument("--create-rings", action="store_true",
                        help="create the rings instead of opening the gateway's")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--passengers", type=int, default=4)
    parser.add_argument(
        "--policy", default="opportunistic", choices=["opportunistic", "eager", "lazy"]
    )
    parser.add_argument("--state-budget-gb", type=float, default=0.0,
                        help="free speculative-state budget the window is divided from")
    parser.add_argument("--hybrid", action="store_true", help="a tiny hybrid model, for a smoke run")
    parser.add_argument("--report-every", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arguments().parse_args(argv)
    cfg = Config()
    cfg.provider.window = args.window
    cfg.provider.batch_passengers = args.passengers
    cfg.provider.policy = args.policy
    if args.state_budget_gb > 0:
        cfg.provider.state_budget_bytes = int(args.state_budget_gb * 1024**3)

    if args.model == "tiny":
        builder = build_tiny_hybrid if args.hybrid else build_tiny
        _, provider_shard, spec = builder()
    else:
        _, provider_shard, spec = load_split(
            args.model, client_layers=args.client_layers, dtype=args.dtype, device=args.device
        )
    engine = ProviderEngine(provider_shard, cfg.provider)
    bridge = ShmProviderBridge(
        engine,
        f"{args.shm}/up.ring",
        f"{args.shm}/down.ring",
        create=args.create_rings,
    )
    print(
        json.dumps(
            {
                "role": "provider",
                "model": args.model,
                "layers_held": len(provider_shard.layers),
                "hidden_size": spec.hidden_size,
                "window": args.window,
                "passengers": args.passengers,
                "policy": args.policy,
                "rings": args.shm,
            }
        ),
        flush=True,
    )

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    last_report = time.monotonic()
    while running:
        if bridge.step() == 0:
            time.sleep(0.0005)
        engine.expire_rounds()
        if args.report_every and time.monotonic() - last_report > args.report_every:
            print(json.dumps({"provider": engine.snapshot()}), flush=True)
            last_report = time.monotonic()
    bridge.close()
    print(json.dumps({"provider": engine.snapshot(), "stopped": True}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
