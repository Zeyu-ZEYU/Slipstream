"""The two shards of a split deployment.

Section 2.1 fixes the split this artifact implements: the trusted client runs
the encoder and the decoder, in our deployments the model's first layer and
its language-model head, which turns a hidden state into a token; the
untrusted provider runs the middle, all remaining layers. Only obfuscated
hidden states cross the boundary.

  ClientShard    embedding, the first `client_layers` layers, the final norm
                 and the language-model head, plus the drafter's module
  ProviderShard  every layer in between, with the tree attention of
                 Section 5 and one recurrent state per path

The shards are plain torch modules, so they run on any GPU or on the CPU. A
tiny configuration is built in for tests; `load_split` maps a Llama-style
Hugging Face checkpoint onto the same classes. The paper's two models run
through the vLLM backend instead, which `vllm_integration/` documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .cache import RequestCache
from .layers import DecoderLayer, LayerConfig, RMSNorm


@dataclass
class ModelSpec:
    vocab_size: int = 512
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 512
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5
    # Indices of layers that are recurrent rather than attention, which is how
    # a hybrid Mamba-Transformer model is laid out.
    recurrent_layers: tuple[int, ...] = ()
    state_size: int = 16
    # Layers the client holds at the front.
    client_layers: int = 1
    dtype: torch.dtype = torch.float32

    def layer_config(self, index: int) -> LayerConfig:
        return LayerConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            intermediate_size=self.intermediate_size,
            rms_eps=self.rms_eps,
            rope_theta=self.rope_theta,
            kind="recurrent" if index in self.recurrent_layers else "attention",
            state_size=self.state_size,
        )

    @property
    def is_hybrid(self) -> bool:
        return bool(self.recurrent_layers)


class ClientShard(nn.Module):
    """Encoder and decoder. Never leaves the client."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embed_tokens = nn.Embedding(spec.vocab_size, spec.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(spec.layer_config(i)) for i in range(spec.client_layers)]
        )
        self.norm = RMSNorm(spec.hidden_size, spec.rms_eps)
        self.lm_head = nn.Linear(spec.hidden_size, spec.vocab_size, bias=False)
        self.cache = RequestCache(num_layers=spec.client_layers, request_id=0)

    # ------------------------------------------------------------- encoding

    @torch.no_grad()
    def encode(
        self,
        token_ids: list[int],
        positions: list[int],
        parents: list[int],
        absolute_positions: list[int],
        cache: RequestCache | None = None,
        round_id: int = 0,
    ) -> torch.Tensor:
        """Run the embedding and the client's layers over a batch of tree
        positions, returning the hidden states that go up.

        `positions` and `parents` are the round's tree, the same numbering the
        provider is told about, and `absolute_positions` are the rotary
        positions. The client runs the same tree attention as the provider,
        over its own cache, so a streamed round and a one-shot round encode
        the tree identically.
        """
        cache = cache or self.cache
        if cache.round_id != round_id:
            cache.start_round(round_id)
        for position, parent in zip(positions, parents):
            cache.note_arrival(round_id, position, parent)

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = self.embed_tokens(torch.tensor(token_ids, device=device)).to(dtype)
        pos = torch.tensor(absolute_positions, device=device)

        index_of = {p: i for i, p in enumerate(positions)}
        parents_in_batch = []
        for position in positions:
            parent = cache.parent_of(position)
            parents_in_batch.append(-1 if parent == position else index_of.get(parent, -1))

        for index, layer in enumerate(self.layers):
            x = _run_layer(
                layer,
                cache.layers[index],
                x,
                pos,
                parents_in_batch,
                list(positions),
                cache=cache,
            )
        cache.ran.update(positions)
        return x

    def commit(self, cache: RequestCache, leaf: int) -> list[int]:
        return cache.commit_path(leaf)

    # ------------------------------------------------------------- decoding

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """The language-model head: a hidden state becomes a distribution over
        tokens. Only the client can do this, which is why only the client can
        verify (Section 4.1, "Constraint")."""
        return self.lm_head(self.norm(hidden.to(next(self.parameters()).dtype)))

    @torch.no_grad()
    def argmax_token(self, hidden: torch.Tensor) -> int:
        return int(self.decode(hidden.unsqueeze(0)).argmax(dim=-1).item())

    def hidden_size(self) -> int:
        return self.spec.hidden_size


class ProviderShard(nn.Module):
    """The middle. Sees hidden states and positions, never tokens."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.layers = nn.ModuleList(
            [
                DecoderLayer(spec.layer_config(i))
                for i in range(spec.client_layers, spec.num_layers)
            ]
        )
        self.caches: dict[int, RequestCache] = {}

    def cache_for(self, request_id: int) -> RequestCache:
        if request_id not in self.caches:
            self.caches[request_id] = RequestCache(
                num_layers=len(self.layers), request_id=request_id
            )
        return self.caches[request_id]

    @torch.no_grad()
    def run(
        self,
        request_id: int,
        positions: list[int],
        parents: list[int],
        hidden: torch.Tensor,
        round_id: int,
        absolute_positions: list[int] | None = None,
    ) -> torch.Tensor:
        """Run one batch of tree positions through the middle.

        The batch is in stream order, so parents precede children. Every token
        attends to the committed prefix with the regular path and to this
        round's tree under the exact ancestor mask, and each recurrent layer
        threads the parent's state (Section 5).
        """
        cache = self.cache_for(request_id)
        for position, parent in zip(positions, parents):
            cache.note_arrival(round_id, position, parent)

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = hidden.to(device=device, dtype=dtype)

        index_of = {p: i for i, p in enumerate(positions)}
        parents_in_batch = [index_of.get(cache.parent_of(p), -1) for p in positions]
        # A token whose parent is itself is the round's root: its parent state
        # is the committed one.
        for i, p in enumerate(positions):
            if cache.parent_of(p) == p:
                parents_in_batch[i] = -1

        if absolute_positions is None:
            prefix_len = cache.layers[0].prefix_len
            absolute_positions = [
                prefix_len + len(cache.ancestors(p)) - 1 for p in positions
            ]
        pos = torch.tensor(absolute_positions, device=device)

        for layer_index, layer in enumerate(self.layers):
            x = _run_layer(
                layer,
                cache.layers[layer_index],
                x,
                pos,
                parents_in_batch,
                positions,
                cache=cache,
            )
        cache.ran.update(positions)
        return x

    def commit(self, request_id: int, leaf: int) -> list[int]:
        return self.cache_for(request_id).commit_path(leaf)

    def release(self, request_id: int) -> None:
        self.caches.pop(request_id, None)

    def state_bytes(self, request_id: int) -> int:
        return self.cache_for(request_id).bytes_held()


def _run_layer(
    layer: DecoderLayer,
    layer_cache,
    x: torch.Tensor,
    positions: torch.Tensor,
    parents_in_batch: list[int],
    tree_positions: list[int],
    cache: RequestCache | None = None,
) -> torch.Tensor:
    """One layer over a batch of tree positions, for either shard."""
    if layer.is_recurrent:
        inherited = []
        for i, _ in enumerate(tree_positions):
            if parents_in_batch[i] >= 0:
                inherited.append(_zero_state(layer, x))
            else:
                parent = (
                    cache.parent_of(tree_positions[i]) if cache is not None else None
                )
                state = None
                if parent is not None:
                    state = layer_cache.tree_state.get(parent)
                if state is None:
                    state = layer_cache.prefix_state
                inherited.append(
                    state if state is not None else _zero_state(layer, x)
                )
        out, new_states = layer.forward_recurrent(x, parents_in_batch, inherited)
        for position, state in zip(tree_positions, new_states):
            layer_cache.tree_state[position] = state
        return out

    # Tree entries this round has already computed, plus this batch's.
    cached_positions = [
        p for p in layer_cache.tree_key.keys() if p not in set(tree_positions)
    ]
    cached_k = [layer_cache.tree_key[p] for p in cached_positions]
    cached_v = [layer_cache.tree_value[p] for p in cached_positions]

    # The batch's own keys and values, needed before attention so that a child
    # can attend to a parent that arrived in the same batch.
    hidden = layer.input_layernorm(x)
    q, k, v = layer.self_attn.project(hidden, positions)

    all_positions = cached_positions + list(tree_positions)
    tree_k = torch.cat([torch.stack(cached_k), k]) if cached_k else k
    tree_v = torch.cat([torch.stack(cached_v), v]) if cached_v else v

    mask = _ancestor_mask(all_positions, tree_positions, cache, parents_in_batch)
    mask = mask.to(x.device)

    from .layers import attend_split, repeat_kv

    repeats = layer.cfg.num_heads // layer.cfg.num_kv_heads
    prefix_kv = (
        (
            repeat_kv(layer_cache.prefix_key, repeats),
            repeat_kv(layer_cache.prefix_value, repeats),
        )
        if layer_cache.prefix_key is not None and layer_cache.prefix_len > 0
        else (None, None)
    )
    attended = attend_split(
        q,
        prefix_kv[0],
        prefix_kv[1],
        repeat_kv(tree_k, repeats),
        repeat_kv(tree_v, repeats),
        mask,
    )
    attended = layer.self_attn.o_proj(attended.reshape(x.shape[0], -1))
    x = x + attended
    x = x + layer.mlp(layer.post_attention_layernorm(x))

    for i, position in enumerate(tree_positions):
        layer_cache.tree_key[position] = k[i]
        layer_cache.tree_value[position] = v[i]
    return x


def _zero_state(layer: DecoderLayer, x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        layer.recurrent.cfg.state_size, device=x.device, dtype=x.dtype
    )


def _ancestor_mask(
    all_positions: list[int],
    batch_positions: list[int],
    cache: RequestCache | None,
    parents_in_batch: list[int],
) -> torch.Tensor:
    """True where a tree position is visible to a query token: itself or one of
    its ancestors."""
    index = {p: i for i, p in enumerate(all_positions)}
    mask = torch.zeros(len(batch_positions), len(all_positions), dtype=torch.bool)
    for i, position in enumerate(batch_positions):
        if cache is not None:
            chain = cache.ancestors(position)
        else:
            chain = _chain_from_batch(i, batch_positions, parents_in_batch)
        for ancestor in chain:
            j = index.get(ancestor)
            if j is not None:
                mask[i][j] = True
        mask[i][index[position]] = True
    return mask


def _chain_from_batch(
    i: int, batch_positions: list[int], parents_in_batch: list[int]
) -> list[int]:
    chain = [batch_positions[i]]
    cursor = i
    while parents_in_batch[cursor] >= 0:
        cursor = parents_in_batch[cursor]
        chain.append(batch_positions[cursor])
    chain.reverse()
    return chain


# --------------------------------------------------------------------- build


def build_tiny(
    spec: ModelSpec | None = None, seed: int = 0, context_free: bool = False
) -> tuple[ClientShard, ProviderShard, ModelSpec]:
    """A small random model with the same structure as a real one. The tests
    and the functional smoke run use it, so the artifact can be exercised end
    to end without a checkpoint or a GPU.

    With `context_free=True` the attention output projections are zeroed, so a
    token's output depends only on that token and the model is a fixed
    next-token map. That makes acceptance deterministic, which is what lets a
    test assert what the accepted path should be. It changes nothing about how
    the tree, the queues or the protocol behave.
    """
    spec = spec or ModelSpec()
    torch.manual_seed(seed)
    client = ClientShard(spec).eval()
    provider = ProviderShard(spec).eval()
    if context_free:
        for shard in (client, provider):
            for layer in shard.layers:
                if layer.self_attn is not None:
                    torch.nn.init.zeros_(layer.self_attn.o_proj.weight)
    return client, provider, spec


def build_tiny_hybrid(seed: int = 0):
    """Same, with every third layer recurrent, which exercises the per-path
    state of Section 4.3."""
    spec = ModelSpec(recurrent_layers=(2, 4), num_layers=6)
    return build_tiny(spec, seed=seed)


def load_split(
    path: str,
    client_layers: int = 1,
    dtype: str = "bfloat16",
    device: str = "cpu",
) -> tuple[ClientShard, ProviderShard, ModelSpec]:
    """Load a Llama-style checkpoint and split it.

    The mapping is the obvious one: the embedding, the first `client_layers`
    decoder layers, the final norm and the head go to the client; the
    remaining decoder layers go to the provider. Mixture-of-experts and hybrid
    checkpoints, including the paper's two models, are served through the vLLM
    backend instead (see `vllm_integration/README.md`).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    hf_config = AutoConfig.from_pretrained(path)
    spec = ModelSpec(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        intermediate_size=hf_config.intermediate_size,
        rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        rms_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
        client_layers=client_layers,
        dtype=torch_dtype,
    )
    source = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype)
    client, provider = ClientShard(spec), ProviderShard(spec)
    _copy_weights(source, client, provider, spec)
    client = client.to(device=device, dtype=torch_dtype).eval()
    provider = provider.to(device=device, dtype=torch_dtype).eval()
    return client, provider, spec


def _copy_weights(source, client: ClientShard, provider: ProviderShard, spec: ModelSpec) -> None:
    base = getattr(source, "model", source)
    client.embed_tokens.weight.data.copy_(base.embed_tokens.weight.data)
    client.norm.weight.data.copy_(base.norm.weight.data)
    client.lm_head.weight.data.copy_(source.lm_head.weight.data)
    for index in range(spec.num_layers):
        src = base.layers[index]
        if index < spec.client_layers:
            dst = client.layers[index]
        else:
            dst = provider.layers[index - spec.client_layers]
        dst.input_layernorm.weight.data.copy_(src.input_layernorm.weight.data)
        dst.post_attention_layernorm.weight.data.copy_(
            src.post_attention_layernorm.weight.data
        )
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(dst.self_attn, name).weight.data.copy_(
                getattr(src.self_attn, name).weight.data
            )
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(dst.mlp, name).weight.data.copy_(getattr(src.mlp, name).weight.data)
