"""Measuring acceptance, with no network in the way.

Section 3.2 asks what a draft budget buys, for a chain and for a tree. That is
a property of the model and its drafter alone, so the sweeps behind Figures 2,
3 and A2 need no split and no link: draft a tree of the given shape and
budget, verify it against the target, and record the acceptance length.

  tau = accepted draft tokens + 1 (the bonus token)

The harness runs both shards in one process, since here they are just the
model. It is the same drafting and the same rerank the deployment uses, so a
shape or a budget measured here is the shape or budget the deployment would
ship.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slipstream.config import DrafterConfig
from slipstream.models.mtp import MTPDrafter
from slipstream.models.split_model import ClientShard, ProviderShard
from slipstream.tree import ROOT, DraftTree


@dataclass
class AcceptanceResult:
    shape: str
    budget: int
    rounds: int
    tau_mean: float
    tau_values: list[float]
    accepted_by_rank: dict[int, int]
    drafted_by_rank: dict[int, int]
    rank1_accept_rate: float
    rank2_rescues: float

    def as_dict(self) -> dict:
        return {
            "shape": self.shape,
            "budget": self.budget,
            "rounds": self.rounds,
            "tau": self.tau_mean,
            "accepted_by_rank": self.accepted_by_rank,
            "drafted_by_rank": self.drafted_by_rank,
            "rank1_accept_rate": self.rank1_accept_rate,
            "rank2_rescue_rate": self.rank2_rescues,
        }


class AcceptanceHarness:
    def __init__(
        self,
        client: ClientShard,
        provider: ProviderShard,
        drafter: MTPDrafter,
        request_id: int = 77,
    ) -> None:
        self.client = client
        self.provider = provider
        self.drafter = drafter
        self.request_id = request_id
        self._round = 0

    @torch.no_grad()
    def _target_outputs(self, tree: DraftTree, positions: list[int], prefix: int) -> dict:
        """Run the whole tree through both shards, as a one-shot round would."""
        self._round += 1
        tokens = [tree[p].token for p in positions]
        parents = [tree[p].parent for p in positions]
        absolute = [prefix + tree[p].depth for p in positions]
        hidden = self.client.encode(
            token_ids=tokens,
            positions=positions,
            parents=parents,
            absolute_positions=absolute,
            cache=None,
            round_id=self._round,
        )
        outputs = self.provider.run(
            request_id=self.request_id,
            positions=positions,
            parents=parents,
            hidden=hidden,
            round_id=self._round,
            absolute_positions=absolute,
        )
        return {p: outputs[i] for i, p in enumerate(positions)}

    def draft(self, root_token: int, seed_hidden, shape: tuple[int, int] | None, budget: int, prefix: int) -> DraftTree:
        """Draft a chain (`shape=None`) or a tree of depth and width `shape`,
        then rerank to `budget` the way Section 2.2 describes."""
        tree = DraftTree(root_token=root_token, round_id=self._round + 1)
        width = 1 if shape is None else shape[1]
        depth = budget if shape is None else shape[0]
        hidden = {ROOT: seed_hidden}
        while len(tree) - 1 < max(budget, width):
            candidate = tree.best_first_candidate()
            if candidate is None or tree[candidate].depth >= depth:
                break
            proposal = self.drafter.propose(
                hidden[candidate], tree[candidate].token, prefix + tree[candidate].depth, width=width
            )
            positions = tree.add_children(candidate, proposal.tokens, proposal.probs)
            for position, child_hidden in zip(positions, proposal.hidden):
                hidden[position] = child_hidden
            if len(tree) - 1 >= 8 * max(1, budget):
                break
        keep = (
            [p for p in tree.trunk()][:budget]
            if shape is None
            else tree.connected_top_m(budget)
        )
        tree.assign_ranks()
        return tree.subtree(keep)

    def measure(
        self,
        prompts: list[list[int]],
        shape: tuple[int, int] | None,
        budget: int,
        max_output_tokens: int = 24,
    ) -> AcceptanceResult:
        taus: list[float] = []
        accepted_by_rank: dict[int, int] = {}
        drafted_by_rank: dict[int, int] = {}
        rank1_seen = rank1_accepted = 0
        rank1_miss = rank2_hit = 0

        for prompt in prompts:
            tokens = list(prompt)
            prefix = len(tokens)
            # One local forward pass gives the first token and the drafter's
            # seed, which is what prefill does in a deployment.
            self._round += 1
            hidden = self.client.encode(
                token_ids=tokens,
                positions=list(range(len(tokens))),
                parents=[0] + list(range(len(tokens) - 1)),
                absolute_positions=list(range(len(tokens))),
                cache=None,
                round_id=self._round,
            )
            outputs = self.provider.run(
                request_id=self.request_id,
                positions=list(range(len(tokens))),
                parents=[0] + list(range(len(tokens) - 1)),
                hidden=hidden,
                round_id=self._round,
                absolute_positions=list(range(len(tokens))),
            )
            seed = outputs[-1]
            root = self.client.argmax_token(seed)
            produced = 1

            while produced < max_output_tokens:
                tree = self.draft(root, seed, shape, budget, prefix)
                positions = [node.position for node in tree.nodes]
                results = self._target_outputs(tree, positions, prefix)
                tree.assign_ranks()
                for node in tree.nodes[1:]:
                    drafted_by_rank[node.rank] = drafted_by_rank.get(node.rank, 0) + 1

                accepted: list[int] = []
                cursor = ROOT
                while True:
                    target = self.client.argmax_token(results[cursor])
                    match = next(
                        (c for c in tree[cursor].children if tree[c].token == target),
                        None,
                    )
                    if cursor == ROOT:
                        children = tree[ROOT].children
                        if children:
                            rank1 = min(children, key=lambda c: tree[c].rank)
                            rank1_seen += 1
                            if tree[rank1].token == target:
                                rank1_accepted += 1
                            else:
                                rank1_miss += 1
                                second = [c for c in children if tree[c].rank == 2]
                                if second and tree[second[0]].token == target:
                                    rank2_hit += 1
                    if match is None:
                        bonus = target
                        break
                    accepted.append(match)
                    accepted_by_rank[tree[match].rank] = (
                        accepted_by_rank.get(tree[match].rank, 0) + 1
                    )
                    cursor = match
                taus.append(len(accepted) + 1)
                for position in accepted:
                    tokens.append(tree[position].token)
                tokens.append(bonus)
                produced += len(accepted) + 1
                prefix += len(accepted) + 1
                root = bonus
                seed = results[accepted[-1]] if accepted else results[ROOT]

        return AcceptanceResult(
            shape="chain" if shape is None else f"({shape[0]},{shape[1]})",
            budget=budget,
            rounds=len(taus),
            tau_mean=sum(taus) / len(taus) if taus else 0.0,
            tau_values=taus,
            accepted_by_rank=accepted_by_rank,
            drafted_by_rank=drafted_by_rank,
            rank1_accept_rate=rank1_accepted / rank1_seen if rank1_seen else 0.0,
            rank2_rescues=rank2_hit / rank1_miss if rank1_miss else 0.0,
        )


def build_harness(spec, cfg=None):
    """A harness on whichever model the run asked for."""
    from experiments.common import build_config, build_drafter, build_model

    cfg = cfg or build_config(spec)
    client, provider, _ = build_model(spec, cfg)
    drafter = build_drafter(spec, cfg, client, provider)
    return AcceptanceHarness(client, provider, drafter)
