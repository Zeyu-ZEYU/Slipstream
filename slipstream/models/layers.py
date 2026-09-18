"""Transformer and recurrent layers with explicit tree attention.

The provider runs batches of draft tokens that belong to a tree, not to a
sequence. Section 5 says how: a foreground batch attends to the prefix with
the regular kernel and to the tree under an exact ancestor mask, merged by
log-sum-exp. `attend_split` below is that merge, written out; it returns
exactly what attention over the concatenation would, which
`tests/test_attention.py` checks.

Hybrid models keep one recurrent state per path rather than per sequence, so
`RecurrentLayer` threads each position's state from its parent's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LayerConfig:
    hidden_size: int = 256
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int | None = None
    intermediate_size: int = 512
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    # "attention" or "recurrent". A hybrid model interleaves the two.
    kind: str = "attention"
    # Recurrent state size, when kind == "recurrent".
    state_size: int = 16

    @property
    def dim_per_head(self) -> int:
        return self.head_dim or (self.hidden_size // self.num_heads)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (x * self.weight.to(torch.float32)).to(dtype)


def rope_cos_sin(
    positions: torch.Tensor, dim: int, theta: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x` is [tokens, heads, dim]; `cos`/`sin` are [tokens, dim]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def attend_split(
    query: torch.Tensor,
    prefix_key: torch.Tensor | None,
    prefix_value: torch.Tensor | None,
    tree_key: torch.Tensor | None,
    tree_value: torch.Tensor | None,
    tree_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Attention over a prefix and a tree, merged by log-sum-exp.

    Shapes, with `n` query tokens, `h` heads and `d` per head:

      query        [n, h, d]
      prefix_*     [p, h, d]      every query token may attend to all of it
      tree_*       [t, h, d]      visible only where `tree_mask` allows
      tree_mask    [n, t] boolean, True where the tree position is an ancestor
                   of the query token or the token itself

    The two partial softmaxes are combined with their own maxima and
    denominators, which is what a partitioned attention kernel does and what
    keeps a streamed round bit-comparable with a one-shot round.
    """
    scale = 1.0 / math.sqrt(query.shape[-1])
    parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    if prefix_key is not None and prefix_key.shape[0] > 0:
        scores = torch.einsum("nhd,phd->nhp", query, prefix_key) * scale
        maximum = scores.amax(dim=-1)
        weights = torch.exp(scores - maximum.unsqueeze(-1))
        parts.append(
            (
                maximum,
                weights.sum(dim=-1),
                torch.einsum("nhp,phd->nhd", weights, prefix_value),
            )
        )

    if tree_key is not None and tree_key.shape[0] > 0:
        scores = torch.einsum("nhd,thd->nht", query, tree_key) * scale
        if tree_mask is not None:
            bias = torch.where(
                tree_mask.unsqueeze(1),
                torch.zeros((), dtype=scores.dtype, device=scores.device),
                torch.full((), float("-inf"), dtype=scores.dtype, device=scores.device),
            )
            scores = scores + bias
        maximum = scores.amax(dim=-1)
        # A query token whose row is entirely masked out contributes nothing;
        # keep its maximum finite so the exponent stays defined.
        finite = torch.isfinite(maximum)
        safe_max = torch.where(finite, maximum, torch.zeros_like(maximum))
        weights = torch.exp(scores - safe_max.unsqueeze(-1))
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        parts.append(
            (
                torch.where(finite, maximum, torch.full_like(maximum, -1e30)),
                weights.sum(dim=-1),
                torch.einsum("nht,thd->nhd", weights, tree_value),
            )
        )

    if not parts:
        return torch.zeros_like(query)

    overall = parts[0][0]
    for maximum, _, _ in parts[1:]:
        overall = torch.maximum(overall, maximum)
    numerator = torch.zeros_like(query)
    denominator = torch.zeros_like(overall)
    for maximum, total, weighted in parts:
        factor = torch.exp(maximum - overall)
        numerator = numerator + weighted * factor.unsqueeze(-1)
        denominator = denominator + total * factor
    return numerator / denominator.clamp(min=1e-20).unsqueeze(-1)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Grouped-query attention: expand key and value heads to query heads."""
    if repeats == 1:
        return x
    tokens, heads, dim = x.shape
    return x.unsqueeze(2).expand(tokens, heads, repeats, dim).reshape(
        tokens, heads * repeats, dim
    )


class Attention(nn.Module):
    def __init__(self, cfg: LayerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dim_per_head
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * d, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * d, cfg.hidden_size, bias=False)

    def project(self, x: torch.Tensor, positions: torch.Tensor):
        cfg = self.cfg
        d = cfg.dim_per_head
        q = self.q_proj(x).view(-1, cfg.num_heads, d)
        k = self.k_proj(x).view(-1, cfg.num_kv_heads, d)
        v = self.v_proj(x).view(-1, cfg.num_kv_heads, d)
        cos, sin = rope_cos_sin(positions, d, cfg.rope_theta, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None,
        tree_kv: tuple[torch.Tensor, torch.Tensor] | None,
        tree_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        repeats = cfg.num_heads // cfg.num_kv_heads
        q, k, v = self.project(x, positions)

        prefix_k = prefix_v = None
        if prefix_kv is not None and prefix_kv[0].shape[0] > 0:
            prefix_k = repeat_kv(prefix_kv[0], repeats)
            prefix_v = repeat_kv(prefix_kv[1], repeats)
        tree_k = repeat_kv(tree_kv[0], repeats) if tree_kv is not None else None
        tree_v = repeat_kv(tree_kv[1], repeats) if tree_kv is not None else None

        out = attend_split(q, prefix_k, prefix_v, tree_k, tree_v, tree_mask)
        out = self.o_proj(out.reshape(x.shape[0], -1))
        # The caller appends (k, v) to the tree cache, keyed by position.
        return out, k, v


class MLP(nn.Module):
    def __init__(self, cfg: LayerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RecurrentLayer(nn.Module):
    """A gated linear recurrence, standing in for the recurrent block of a
    hybrid model.

    The point for this artifact is the state, not the kernel: a draft tree
    needs one state per path, so the provider carries `state[parent]` into
    `state[position]` and keeps them apart. Section 4.3 sizes the window from
    exactly this state.
    """

    def __init__(self, cfg: LayerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.hidden_size, cfg.state_size, bias=False)
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.state_size, bias=False)
        self.out_proj = nn.Linear(cfg.state_size, cfg.hidden_size, bias=False)
        self.decay = nn.Parameter(torch.full((cfg.state_size,), -1.0))

    def step(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One token. `state` is [state_size]; returns the output and the new
        state."""
        decay = torch.sigmoid(self.decay).to(x.dtype)
        gate = torch.sigmoid(self.gate_proj(x))
        new_state = decay * state + gate * self.in_proj(x)
        return self.out_proj(new_state), new_state

    def state_bytes(self) -> int:
        return self.cfg.state_size * 4


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LayerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.mlp = MLP(cfg)
        if cfg.kind == "attention":
            self.self_attn = Attention(cfg)
            self.recurrent = None
        else:
            self.self_attn = None
            self.recurrent = RecurrentLayer(cfg)

    @property
    def is_recurrent(self) -> bool:
        return self.recurrent is not None

    def forward_attention(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        prefix_kv,
        tree_kv,
        tree_mask,
    ):
        residual = x
        hidden = self.input_layernorm(x)
        attended, k, v = self.self_attn(hidden, positions, prefix_kv, tree_kv, tree_mask)
        x = residual + attended
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k, v

    def forward_recurrent(
        self,
        x: torch.Tensor,
        parents_in_batch: list[int],
        inherited_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """One recurrent layer over a batch of tree positions.

        `parents_in_batch[i]` is the index of token `i`'s parent inside this
        batch, or -1 when the parent's state comes from the cache, in which
        case `inherited_states[i]` holds it. The batch must be in topological
        order, which the stream guarantees: a parent always precedes its
        children (Section 4.2, "Stream order").
        """
        residual = x
        hidden = self.input_layernorm(x)
        outputs: list[torch.Tensor] = []
        new_states: list[torch.Tensor] = []
        for i in range(hidden.shape[0]):
            parent = parents_in_batch[i]
            base = new_states[parent] if parent >= 0 else inherited_states[i]
            out, state = self.recurrent.step(hidden[i], base)
            outputs.append(out)
            new_states.append(state)
        x = residual + torch.stack(outputs)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_states
