#!/usr/bin/env python3
"""Figure 4: where a round's time goes.

Section 3.3 times one round of a single request and splits it into five
stages: drafting and encoding on the client, the uplink, the middle on the
provider, the downlink, and decoding and verification on the client. What the
figure shows is that the two network legs together take more than all three
compute stages, at every budget.

    python experiments/motivation/stage_breakdown.py --budgets 3 6 9 12

The uplink and downlink are charged at the rate of the link the run is shaped
for, so on a real deployment (`deploy/`) these are wall-clock measurements,
and on a single machine they are the software-shaped link of
`slipstream/transport/link.py`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import datasets
from experiments.common import HOME_LINK, RunSpec, build_config, build_drafter, build_model, write_result
from slipstream.codec import encode, payload_bytes
from slipstream.provider.engine import ProviderEngine
from slipstream.tree import ROOT, DraftTree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--shape", default="6,2")
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--drafter", default="oracle", choices=["oracle", "mtp"])
    parser.add_argument("--hit-rate", type=float, default=0.6)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=0,
        help="charge the link for a hidden state of this many elements instead "
        "of the model's own; 4096 is the paper's models, whose hidden state is "
        "8 KB at 16-bit precision (Section 3.1)",
    )
    parser.add_argument("--out", default="results/motivation/stage_breakdown.json")
    args = parser.parse_args()

    depth, width = (int(part) for part in args.shape.split(","))
    spec = RunSpec(model=args.model, drafter=args.drafter, hit_rate=args.hit_rate, seed=args.seed)
    cfg = build_config(spec)
    client, provider_shard, model_spec = build_model(spec, cfg)
    drafter = build_drafter(spec, cfg, client, provider_shard)
    provider = ProviderEngine(provider_shard, cfg.provider)
    hidden_size = model_spec.hidden_size
    wire_size = args.hidden_size or hidden_size
    prompts = datasets.load("math", 1, prompt_tokens=args.prompt_tokens, seed=args.seed)

    out = {
        "budgets": args.budgets,
        "stages": {},
        "link": HOME_LINK.describe(),
        "hidden_bytes": payload_bytes(wire_size, "bf16"),
        "wire_hidden_size": wire_size,
        "model_hidden_size": hidden_size,
    }
    for budget in args.budgets:
        totals = {"draft_encode": 0.0, "uplink": 0.0, "middle": 0.0, "downlink": 0.0, "decode_verify": 0.0}
        for _ in range(args.rounds):
            tokens = list(prompts[0])
            # Drafting and encoding, on the client.
            started = time.perf_counter()
            tree = DraftTree(root_token=tokens[-1], round_id=1)
            seed_hidden = torch.zeros(hidden_size)
            hidden_by_position = {ROOT: seed_hidden}
            while len(tree) - 1 < budget:
                candidate = tree.best_first_candidate()
                if candidate is None or tree[candidate].depth >= depth:
                    break
                proposal = drafter.propose(
                    hidden_by_position[candidate], tree[candidate].token, tree[candidate].depth, width=width
                )
                for position, child in zip(
                    tree.add_children(candidate, proposal.tokens, proposal.probs), proposal.hidden
                ):
                    hidden_by_position[position] = child
            positions = [node.position for node in tree.nodes][: budget + 1]
            encoded = client.encode(
                token_ids=[tree[p].token for p in positions],
                positions=positions,
                parents=[tree[p].parent for p in positions],
                absolute_positions=[len(tokens) + tree[p].depth for p in positions],
                cache=None,
                round_id=1,
            )
            payloads = [encode(row, "bf16") for row in encoded]
            totals["draft_encode"] += (time.perf_counter() - started) * 1e3

            # The uplink: every drafted token is one hidden state.
            per_state = payload_bytes(wire_size, "bf16")
            totals["uplink"] += sum(HOME_LINK.uplink_ms(per_state) for _ in payloads)

            # The middle, on the provider.
            started = time.perf_counter()
            outputs = provider_shard.run(
                request_id=1,
                positions=positions,
                parents=[tree[p].parent for p in positions],
                hidden=encoded,
                round_id=1,
                absolute_positions=[len(tokens) + tree[p].depth for p in positions],
            )
            totals["middle"] += (time.perf_counter() - started) * 1e3

            # The downlink carries the same count back.
            totals["downlink"] += sum(
                HOME_LINK.downlink_ms(per_state) for _ in positions
            )

            # Decoding and verification, on the client.
            started = time.perf_counter()
            for row in outputs:
                client.argmax_token(row)
            totals["decode_verify"] += (time.perf_counter() - started) * 1e3

        stages = {name: value / args.rounds for name, value in totals.items()}
        stages["round_ms"] = sum(stages.values())
        stages["network_share"] = (stages["uplink"] + stages["downlink"]) / stages["round_ms"]
        out["stages"][str(budget)] = {k: round(v, 3) for k, v in stages.items()}
        print(
            f"m={budget:3d} round={stages['round_ms']:7.1f} ms  "
            f"network={100 * stages['network_share']:4.1f}%  "
            f"up={stages['uplink']:6.1f} mid={stages['middle']:6.1f} down={stages['downlink']:6.1f}"
        )
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
