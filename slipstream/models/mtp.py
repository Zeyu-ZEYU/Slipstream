"""The drafter: the model's own multi-token-prediction module.

Section 2.2 fixes what the drafter is and where it runs. In self-speculation
the target drafts for itself: a small module sits on top of the target and
reuses its hidden states and language-model head. In a split deployment the
module runs on the client, which has everything it needs, the hidden states
the provider returns each round and the embeddings and head it already holds,
so it drafts without a round trip.

One trained next-token layer is applied recursively for deeper positions. A
drafting step takes the hidden state at a node and the node's token, and
produces the `k` most probable children together, from one softmax
(Section 4.2). Because the step never touches the network, drafting overlaps
transmission: the root's hidden state goes up while the first step runs, and
every later step runs while the previous hidden state is in flight.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import DrafterConfig
from .layers import DecoderLayer, RMSNorm
from .split_model import ClientShard, ModelSpec


class MTPModule(nn.Module):
    """One next-token layer, applied recursively.

    The shape follows the vendor modules the paper uses: the hidden state of
    the position and the embedding of the token that follows it are projected
    together and run through a single decoder layer, and the target's own norm
    and head turn the result into a distribution.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.hidden_norm = RMSNorm(spec.hidden_size, spec.rms_eps)
        self.token_norm = RMSNorm(spec.hidden_size, spec.rms_eps)
        self.input_proj = nn.Linear(2 * spec.hidden_size, spec.hidden_size, bias=False)
        self.layer = DecoderLayer(spec.layer_config(0))

    @torch.no_grad()
    def step(
        self, hidden: torch.Tensor, token_embedding: torch.Tensor, position: int
    ) -> torch.Tensor:
        """One recursive application, for one node."""
        combined = torch.cat(
            [self.hidden_norm(hidden), self.token_norm(token_embedding)], dim=-1
        )
        x = self.input_proj(combined).unsqueeze(0)
        positions = torch.tensor([position], device=x.device)
        out, _, _ = self.layer.forward_attention(
            x, positions, prefix_kv=None, tree_kv=None, tree_mask=None
        )
        return out.squeeze(0)


@dataclass
class Proposal:
    tokens: list[int]
    probs: list[float]
    # The drafter's hidden state at each child, from which the child's own
    # children are drafted.
    hidden: list[torch.Tensor]


class MTPDrafter:
    """Wraps the module and the client's head into the interface the
    scheduler uses."""

    def __init__(
        self,
        client: ClientShard,
        module: MTPModule | None = None,
        cfg: DrafterConfig | None = None,
    ) -> None:
        self.client = client
        self.cfg = cfg or DrafterConfig()
        self.module = module or MTPModule(client.spec)
        self.module = self.module.to(
            device=next(client.parameters()).device,
            dtype=next(client.parameters()).dtype,
        ).eval()

    @torch.no_grad()
    def propose(
        self, hidden: torch.Tensor, token: int, position: int, width: int | None = None
    ) -> Proposal:
        """The `width` most probable children of one node, together."""
        width = width or self.cfg.children_per_step
        device = next(self.client.parameters()).device
        embedding = self.client.embed_tokens(torch.tensor([token], device=device))[0]
        drafted = self.module.step(hidden.to(device), embedding, position)
        logits = self.client.decode(drafted.unsqueeze(0))[0].to(torch.float32)
        if self.cfg.temperature != 1.0:
            logits = logits / self.cfg.temperature
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(width, probs.numel()))
        return Proposal(
            tokens=[int(t) for t in top.indices.tolist()],
            probs=[float(p) for p in top.values.tolist()],
            hidden=[drafted for _ in top.indices],
        )

    def state_dict(self):
        return self.module.state_dict()

    @classmethod
    def load(
        cls, client: ClientShard, path: str, cfg: DrafterConfig | None = None
    ) -> "MTPDrafter":
        """Load a vendor module from a checkpoint saved by `torch.save`."""
        module = MTPModule(client.spec)
        module.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(client, module, cfg)
