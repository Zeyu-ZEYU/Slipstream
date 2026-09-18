"""A drafter that proposes from the target itself.

The interface is the drafter's: given the hidden state at a node and the
node's token, return the `k` most probable children together, with their
probabilities. This one builds that proposal around the token the target model
would emit, placing it at a chosen rank with a chosen probability, so a test
can set the acceptance rate it wants and check what the design does with it:

  hit_rate=1.0           every first choice is the target's token, so the whole
                         first segment is accepted
  hit_rate=0.6, rank=2   the first choice differs in four proposals out of ten
                         and the second choice is the target's, which is the
                         case branch continuation exists for (Section 4.2); a
                         chain, having no second choice, simply misses

It reads the target through both shards, so it runs where both are in one
process: the tests, and a single-machine run asked for with `--drafter
oracle`. `MTPDrafter` is the drafter of Section 2.2, which runs on the client
alone and is what `--drafter mtp` selects.
"""

from __future__ import annotations

import random

import torch

from ..config import DrafterConfig
from .mtp import MTPDrafter, Proposal
from .split_model import ClientShard, ProviderShard


class OracleDrafter(MTPDrafter):
    def __init__(
        self,
        client: ClientShard,
        provider: ProviderShard,
        cfg: DrafterConfig | None = None,
        hit_rate: float = 1.0,
        rank: int = 2,
        seed: int = 0,
        request_id: int = 10_000,
    ) -> None:
        super().__init__(client, None, cfg)
        self.provider = provider
        self.hit_rate = hit_rate
        self.rank = rank
        self.rng = random.Random(seed)
        self._request_id = request_id
        self._next_round = 1

    @torch.no_grad()
    def propose(
        self, hidden: torch.Tensor, token: int, position: int, width: int | None = None
    ) -> Proposal:
        width = width or self.cfg.children_per_step
        base = super().propose(hidden, token, position, width=width)

        truth = self._target_token(token, position)
        if truth is None:
            return base

        hit = self.rng.random() < self.hit_rate
        if not hit and width == 1:
            # A chain has no second choice, so a miss is a miss.
            return base
        tokens = [t for t in base.tokens if t != truth]
        slot = 0 if hit else min(self.rank - 1, width - 1)
        tokens.insert(slot, truth)
        tokens = tokens[:width]
        # A proposal whose probabilities fall with rank, so path probabilities
        # and the acceptance table behave as they would with a trained module.
        ladder = ([0.7, 0.15, 0.08, 0.04, 0.02, 0.01] + [0.005] * width)[:width]
        return Proposal(
            tokens=tokens,
            probs=ladder,
            hidden=[base.hidden[0]] * len(tokens),
        )

    @torch.no_grad()
    def _target_token(self, token: int, position: int) -> int | None:
        """What the served model itself emits after `token`, computed by
        running both shards in this process."""
        try:
            self._next_round += 1
            cache_round = self._next_round
            hidden = self.client.encode(
                token_ids=[token],
                positions=[0],
                parents=[0],
                absolute_positions=[position],
                cache=None,
                round_id=cache_round,
            )
            out = self.provider.run(
                request_id=self._request_id,
                positions=[0],
                parents=[0],
                hidden=hidden,
                round_id=cache_round,
                absolute_positions=[position],
            )
            return self.client.argmax_token(out[0])
        except Exception:  # pragma: no cover - only reachable in tests
            return None
