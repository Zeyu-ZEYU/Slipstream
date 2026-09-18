#!/usr/bin/env python3
"""Figure 12: at scale.

Section 6.6 runs many clients per provider and many providers: eight providers
behind a router that sends every new request to the provider with the fewest
requests in flight and never moves it, and clients on emulated residential
links, `K` requests per client.

Two ways to run it:

  * `--providers N --clients M` in one process, which is what this driver
    does: it builds N provider engines, routes each client to one of them by
    the same least-requests rule, and shapes every client's link;
  * on a real cluster with the Rust gateway behind Envoy, which
    `deploy/cluster/launch_at_scale.sh` and `deploy/envoy/envoy.yaml` set up,
    and `deploy/netem.sh` shapes.

    python experiments/evaluation/at_scale.py --providers 2 --clients 6 --k 1 2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import datasets
from experiments.common import (
    HOME_LINK,
    PAPER_HIDDEN_BYTES,
    RunSpec,
    build_config,
    build_drafter,
    build_model,
    write_result,
)
from slipstream.client.engine import ClientEngine
from slipstream.methods import METHODS
from slipstream.provider.engine import ProviderEngine
from slipstream.transport.link import LinkModel
from slipstream.transport.loopback import LoopbackTransport, ProviderHub


class LeastRequestsRouter:
    """The router of Section 6.6: every new request goes to the provider with
    the fewest requests in flight, and never moves."""

    def __init__(self, providers: int) -> None:
        self.in_flight = [0] * providers
        self._lock = threading.Lock()

    def pick(self) -> int:
        with self._lock:
            index = min(range(len(self.in_flight)), key=lambda i: self.in_flight[i])
            self.in_flight[index] += 1
            return index

    def release(self, index: int) -> None:
        with self._lock:
            self.in_flight[index] = max(0, self.in_flight[index] - 1)

    def spread(self) -> int:
        with self._lock:
            return max(self.in_flight) - min(self.in_flight)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", type=int, default=2)
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2],
                        help="requests in flight per client, K (Section 6.6)")
    parser.add_argument("--method", default="slipstream", choices=list(METHODS))
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=24)
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--drafter", default="oracle", choices=["oracle", "mtp"])
    parser.add_argument("--hit-rate", type=float, default=0.6)
    parser.add_argument("--workload", default="generated")
    parser.add_argument("--wire-hidden-bytes", type=int, default=PAPER_HIDDEN_BYTES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/evaluation/at_scale.json")
    args = parser.parse_args()

    spec = RunSpec(
        method=args.method, clients=args.clients, model=args.model,
        drafter=args.drafter, hit_rate=args.hit_rate, seed=args.seed,
        max_output_tokens=args.max_output_tokens,
    )
    cfg = build_config(spec)
    torch.manual_seed(args.seed)
    client_shard, provider_shard, _ = build_model(spec, cfg)
    prompts = datasets.load(
        args.workload, max(8, args.clients * max(args.k)),
        prompt_tokens=args.prompt_tokens, seed=args.seed,
    )
    link = LinkModel(
        uplink_mbps=HOME_LINK.uplink_mbps,
        downlink_mbps=HOME_LINK.downlink_mbps,
        one_way_delay_ms=HOME_LINK.one_way_delay_ms,
        charge_bytes=(args.wire_hidden_bytes or None),
    )

    out = {"runs": {}, "providers": args.providers, "clients": args.clients,
           "link": link.describe()}
    for k in args.k:
        providers = [ProviderEngine(provider_shard, cfg.provider) for _ in range(args.providers)]
        hubs = [ProviderHub(engine) for engine in providers]
        router = LeastRequestsRouter(args.providers)
        assignments = [router.pick() for _ in range(args.clients)]
        transports, engines = [], []
        for index, provider_index in enumerate(assignments):
            transport = LoopbackTransport(
                providers[provider_index], link=link, client_id=index
            )
            hubs[provider_index].register(transport)
            transports.append(transport)
            engines.append(
                ClientEngine(
                    client_shard,
                    build_drafter(spec, cfg, client_shard, provider_shard),
                    transport,
                    cfg,
                )
            )
        for hub in hubs:
            hub.start()

        metrics: list = []
        lock = threading.Lock()

        def serve(index: int) -> None:
            method = METHODS[args.method](engines[index])
            for slot in range(k):
                result = method.run_request(
                    index * 1000 + slot + 1,
                    prompts[(index * k + slot) % len(prompts)],
                    args.max_output_tokens,
                )
                with lock:
                    metrics.append(result)

        started = time.monotonic()
        threads = [threading.Thread(target=serve, args=(i,)) for i in range(args.clients)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.monotonic() - started
        for hub in hubs:
            hub.close()
        for transport in transports:
            transport.close()

        tokens = sum(m.output_tokens for m in metrics)
        tpots = sorted(m.tpot_ms for m in metrics if m.tpot_ms > 0)
        row = {
            "requests_in_flight_total": args.clients * k,
            "per_provider": args.clients * k / args.providers,
            "goodput_tokens_per_s": round(tokens / wall, 2) if wall else 0.0,
            "tpot_p50_ms": round(statistics.median(tpots), 2) if tpots else 0.0,
            "tau_mean": round(
                sum(m.tau_mean for m in metrics) / len(metrics), 3
            ) if metrics else 0.0,
            "router_spread_requests": router.spread(),
            "provider_batches": [p.batches_run for p in providers],
        }
        out["runs"][str(k)] = row
        print(
            f"K={k:3d} R={row['requests_in_flight_total']:4d} "
            f"goodput={row['goodput_tokens_per_s']:8.1f} tok/s "
            f"TPOT P50={row['tpot_p50_ms']:7.1f} ms tau={row['tau_mean']:.2f} "
            f"router spread={row['router_spread_requests']}"
        )
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
