"""What every experiment needs: a deployment, a closed-loop driver, and the
two metrics.

Section 6.1 fixes the shape of a run. Load is closed-loop: each client keeps
`R / clients` requests in flight and starts the next as one finishes, so the
provider always has `R` in flight and `R` sets the load. Goodput, the
provider's view, is output tokens per second summed over the clients. Time per
output token, one user's view, is the time from a request's first output token
to its last divided by the output tokens in between, reported as the median
over requests. Prefill, which no method changes, is outside both, and
statistics cover the steady state.

A deployment here is one provider engine and a number of clients, each with
its own link. Which transport carries the traffic is a flag:

  direct    one process, no link, deterministic; for correctness runs
  loopback  one process, links shaped in software; the default
  shm       the Rust gateway over a real network (see deploy/)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field

import torch

from slipstream.client.engine import ClientEngine
from slipstream.config import Config
from slipstream.costs import OverheadCounters
from slipstream.methods import METHODS, NAMES
from slipstream.models import MTPDrafter, build_tiny, build_tiny_hybrid, load_split
from slipstream.models.oracle import OracleDrafter
from slipstream.provider.engine import ProviderEngine
from slipstream.transport.link import LinkModel
from slipstream.transport.loopback import DirectTransport, LoopbackTransport, ProviderHub

# The paper's residential link (Section 3.1).
HOME_LINK = LinkModel(uplink_mbps=20.0, downlink_mbps=100.0, one_way_delay_ms=20.0)

# A hidden state of a model with hidden size 4096, at 16-bit precision
# (Section 3.1). A functional run on the tiny model can charge the link for
# this instead of its own 512 bytes, which puts the run in the paper's regime.
PAPER_HIDDEN_BYTES = 8192


@dataclass
class RunSpec:
    method: str = "slipstream"
    clients: int = 4
    requests_in_flight: int = 4
    requests_per_client: int = 4
    max_output_tokens: int = 32
    prompt_tokens: int = 24
    workload: str = "generated"
    model: str = "tiny"
    drafter: str = "oracle"
    hit_rate: float = 0.6
    transport: str = "loopback"
    hybrid: bool = False
    seed: int = 0
    label: str = ""
    # Bytes the link charges per frame; None charges the real payload.
    wire_hidden_bytes: int | None = None
    # Overrides applied to the configuration, as dotted keys, for example
    # {"provider.policy": "eager"}.
    overrides: dict = field(default_factory=dict)


@dataclass
class RunResult:
    spec: dict
    goodput_tokens_per_s: float
    tpot_p50_ms: float
    tpot_p99_ms: float
    tau_mean: float
    first_segment_mean: float
    rounds: int
    output_tokens: int
    wall_s: float
    bytes_up: int
    bytes_down: int
    provider: dict
    client: dict
    overhead: dict
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    for key, value in overrides.items():
        section, _, field_name = key.partition(".")
        target = getattr(cfg, section)
        if not hasattr(target, field_name):
            raise KeyError(f"no such setting: {key}")
        setattr(target, field_name, value)
    return cfg


def build_config(spec: RunSpec) -> Config:
    cfg = Config()
    cfg.seed = spec.seed
    cfg.decode.max_output_tokens = spec.max_output_tokens
    cfg.model.path = spec.model
    return apply_overrides(cfg, spec.overrides)


def build_model(spec: RunSpec, cfg: Config):
    if spec.model == "tiny":
        if spec.hybrid:
            client, provider, model_spec = build_tiny_hybrid(seed=spec.seed)
            for shard in (client, provider):
                for layer in shard.layers:
                    if layer.self_attn is not None:
                        torch.nn.init.zeros_(layer.self_attn.o_proj.weight)
        else:
            client, provider, model_spec = build_tiny(
                seed=spec.seed, context_free=(spec.drafter == "oracle")
            )
        return client, provider, model_spec
    client, provider, model_spec = load_split(
        spec.model,
        client_layers=cfg.model.client_layers,
        dtype=cfg.model.dtype,
        device=cfg.model.client_device,
    )
    return client, provider, model_spec


def build_drafter(spec: RunSpec, cfg: Config, client, provider):
    if spec.drafter == "oracle":
        return OracleDrafter(
            client, provider, cfg.drafter, hit_rate=spec.hit_rate, seed=spec.seed
        )
    if cfg.model.mtp_path:
        return MTPDrafter.load(client, cfg.model.mtp_path, cfg.drafter)
    return MTPDrafter(client, None, cfg.drafter)


class Deployment:
    """One provider, several clients, one link each."""

    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.cfg = build_config(spec)
        torch.manual_seed(spec.seed)
        client_shard, provider_shard, self.model_spec = build_model(spec, self.cfg)
        self.overhead = OverheadCounters()
        self.provider = ProviderEngine(
            provider_shard, self.cfg.provider, overhead=self.overhead
        )
        self.clients: list[ClientEngine] = []
        self.transports: list = []
        self.hub = ProviderHub(self.provider) if spec.transport != "direct" else None
        for index in range(spec.clients):
            if spec.transport == "direct":
                transport = DirectTransport(self.provider, client_id=index)
            else:
                link = HOME_LINK
                if spec.wire_hidden_bytes:
                    link = LinkModel(
                        uplink_mbps=HOME_LINK.uplink_mbps,
                        downlink_mbps=HOME_LINK.downlink_mbps,
                        one_way_delay_ms=HOME_LINK.one_way_delay_ms,
                        charge_bytes=spec.wire_hidden_bytes,
                    )
                transport = LoopbackTransport(
                    self.provider, link=link, client_id=index
                )
            drafter = build_drafter(spec, self.cfg, client_shard, provider_shard)
            engine = ClientEngine(
                client_shard, drafter, transport, self.cfg, overhead=self.overhead
            )
            self.clients.append(engine)
            self.transports.append(transport)
            if self.hub is not None:
                self.hub.register(transport)
        if self.hub is not None:
            # One provider loop serves every client, as a server would.
            self.hub.start()

    def close(self) -> None:
        if self.hub is not None:
            self.hub.close()
        for transport in self.transports:
            transport.close()


def closed_loop(spec: RunSpec, prompts: list[list[int]]) -> RunResult:
    """Run `spec` and return its two metrics.

    Each client keeps `requests_per_client` requests in flight and starts the
    next as one finishes; the run stops once every client has served
    `requests_per_client` of them, which is the steady state the paper
    measures.
    """
    deployment = Deployment(spec)
    method_class = METHODS[spec.method]
    per_client = max(1, spec.requests_in_flight // max(1, spec.clients))
    results: list[list] = [[] for _ in deployment.clients]
    errors: list[Exception] = []

    def serve(index: int) -> None:
        engine = deployment.clients[index]
        method = method_class(engine)
        try:
            for slot in range(per_client):
                request_id = index * 1000 + slot + 1
                prompt = prompts[(index * per_client + slot) % len(prompts)]
                results[index].append(
                    method.run_request(request_id, prompt, spec.max_output_tokens)
                )
        except Exception as error:  # pragma: no cover - surfaced in the result
            errors.append(error)

    started = time.monotonic()
    threads = [
        threading.Thread(target=serve, args=(index,), name=f"client{index}")
        for index in range(spec.clients)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - started
    deployment.close()
    if errors:
        raise errors[0]

    metrics = [m for per in results for m in per]
    tokens = sum(m.output_tokens for m in metrics)
    tpots = sorted(m.tpot_ms for m in metrics if m.tpot_ms > 0)
    taus = [m.tau_mean for m in metrics if m.tau_mean > 0]
    segments = [
        value
        for engine in deployment.clients
        for value in engine.stats["first_segment_lengths"]
    ]
    return RunResult(
        spec=asdict(spec),
        goodput_tokens_per_s=tokens / wall if wall > 0 else 0.0,
        tpot_p50_ms=statistics.median(tpots) if tpots else 0.0,
        tpot_p99_ms=tpots[min(len(tpots) - 1, int(0.99 * len(tpots)))] if tpots else 0.0,
        tau_mean=sum(taus) / len(taus) if taus else 0.0,
        first_segment_mean=sum(segments) / len(segments) if segments else 0.0,
        rounds=sum(engine.stats["rounds"] for engine in deployment.clients),
        output_tokens=tokens,
        wall_s=wall,
        bytes_up=sum(t.snapshot().get("bytes_up", 0) for t in deployment.transports),
        bytes_down=sum(t.snapshot().get("bytes_down", 0) for t in deployment.transports),
        provider=deployment.provider.snapshot(),
        client=deployment.clients[0].snapshot(),
        overhead=deployment.overhead.summary(),
        note=note_for(spec),
    )


def note_for(spec: RunSpec) -> str:
    """The settings a run used, recorded with its result."""
    parts = [f"model={spec.model}", f"drafter={spec.drafter}"]
    if spec.drafter == "oracle":
        parts.append(f"hit_rate={spec.hit_rate}")
    parts.append(f"transport={spec.transport}")
    if spec.wire_hidden_bytes:
        parts.append(f"link charges {spec.wire_hidden_bytes} bytes a hidden state")
    if spec.hybrid:
        parts.append("hybrid model")
    return ", ".join(parts)


# --------------------------------------------------------------------- output


def write_result(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument(
        "--in-flight",
        type=int,
        nargs="+",
        default=[4],
        help="requests in flight at the provider, R (Section 6.1)",
    )
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=24)
    parser.add_argument("--requests", type=int, default=4, help="requests per client")
    parser.add_argument("--model", default="tiny", help="'tiny' or a checkpoint path")
    parser.add_argument(
        "--workload",
        default="sharegpt",
        choices=["sharegpt", "livecodebench", "math", "longbench", "generated"],
        help="which request set the run names; with --dataset it is read from "
        "disk, without one its prompt lengths are generated from that set's "
        "distribution",
    )
    parser.add_argument(
        "--dataset", default="", help="a jsonl of prompts for --workload"
    )
    parser.add_argument("--drafter", default="oracle", choices=["oracle", "mtp"])
    parser.add_argument("--hit-rate", type=float, default=0.6)
    parser.add_argument("--transport", default="loopback", choices=["direct", "loopback", "shm"])
    parser.add_argument("--hybrid", action="store_true", help="a hybrid model, with recurrent layers")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wire-hidden-bytes",
        type=int,
        default=PAPER_HIDDEN_BYTES,
        help="bytes the software-shaped link charges per hidden state; 8192 is "
        "a model of hidden size 4096 at 16-bit precision (Section 3.1), and 0 "
        "charges the payload itself",
    )
    parser.add_argument("--out", default="")
    return parser


def spec_from_arguments(args, method: str, in_flight: int, **extra) -> RunSpec:
    return RunSpec(
        method=method,
        clients=args.clients,
        requests_in_flight=in_flight,
        requests_per_client=args.requests,
        max_output_tokens=args.max_output_tokens,
        prompt_tokens=args.prompt_tokens,
        workload=args.workload,
        model=args.model,
        drafter=args.drafter,
        hit_rate=args.hit_rate,
        transport=args.transport,
        hybrid=args.hybrid,
        seed=args.seed,
        wire_hidden_bytes=(args.wire_hidden_bytes or None),
        **extra,
    )


def summarize(results: list[RunResult]) -> str:
    lines = [
        f"{'method':16s} {'R':>4s} {'goodput':>9s} {'TPOT P50':>9s} {'tau':>5s} {'s':>4s}"
    ]
    for result in results:
        lines.append(
            f"{NAMES.get(result.spec['method'], result.spec['method']):16s} "
            f"{result.spec['requests_in_flight']:4d} "
            f"{result.goodput_tokens_per_s:9.1f} {result.tpot_p50_ms:9.1f} "
            f"{result.tau_mean:5.2f} {result.first_segment_mean:4.1f}"
        )
    return "\n".join(lines)
