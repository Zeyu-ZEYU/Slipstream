"""The provider's per-request state.

Appendix C fixes what the provider may keep. Per round it keeps the tree it has
received so far, each position's parent and whether its hidden state has
arrived and run, and the key and value entries and recurrent states of the
positions it has run. It never holds a token identifier or a probability.

Closing a round needs no verdict: the next round's first hidden state carries a
new round identifier and a parent pointer to the accepted leaf, and the
provider commits that leaf's ancestors, appending their key and value entries
and keeping the leaf's recurrent state, and releases the rest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class LayerCache:
    """Committed prefix, plus this round's speculative entries."""

    # [tokens, kv_heads, dim] for the committed prefix.
    prefix_key: torch.Tensor | None = None
    prefix_value: torch.Tensor | None = None
    # Speculative entries of the current round, by position.
    tree_key: dict[int, torch.Tensor] = field(default_factory=dict)
    tree_value: dict[int, torch.Tensor] = field(default_factory=dict)
    # Recurrent layers keep one state per path instead.
    prefix_state: torch.Tensor | None = None
    tree_state: dict[int, torch.Tensor] = field(default_factory=dict)

    @property
    def prefix_len(self) -> int:
        return 0 if self.prefix_key is None else int(self.prefix_key.shape[0])

    def clear_round(self) -> None:
        self.tree_key.clear()
        self.tree_value.clear()
        self.tree_state.clear()

    def keep_only(self, positions: set[int]) -> None:
        for store in (self.tree_key, self.tree_value, self.tree_state):
            for position in list(store.keys()):
                if position not in positions:
                    del store[position]

    def commit(self, positions: list[int]) -> None:
        """Append the key and value entries of `positions`, in order, to the
        prefix, and keep the last one's recurrent state."""
        keys = [self.tree_key[p] for p in positions if p in self.tree_key]
        values = [self.tree_value[p] for p in positions if p in self.tree_value]
        if keys:
            stacked_k = torch.stack(keys)
            stacked_v = torch.stack(values)
            self.prefix_key = (
                stacked_k
                if self.prefix_key is None
                else torch.cat([self.prefix_key, stacked_k])
            )
            self.prefix_value = (
                stacked_v
                if self.prefix_value is None
                else torch.cat([self.prefix_value, stacked_v])
            )
        states = [self.tree_state[p] for p in positions if p in self.tree_state]
        if states:
            self.prefix_state = states[-1]
        self.clear_round()

    def bytes_held(self) -> int:
        def size(tensor: torch.Tensor | None) -> int:
            return 0 if tensor is None else tensor.numel() * tensor.element_size()

        total = size(self.prefix_key) + size(self.prefix_value) + size(self.prefix_state)
        for store in (self.tree_key, self.tree_value, self.tree_state):
            total += sum(size(t) for t in store.values())
        return total


class RequestCache:
    """One per request on the provider."""

    def __init__(self, num_layers: int, request_id: int) -> None:
        self.request_id = request_id
        self.layers = [LayerCache() for _ in range(num_layers)]
        # Positions the provider has run this round.
        self.ran: set[int] = set()
        # Parent of every position of this round, the only tree structure the
        # provider is told about.
        self.parents: dict[int, int] = {}
        self.round_id: int | None = None
        self.last_touch = time.monotonic()

    # ------------------------------------------------------------ structure

    def note_arrival(self, round_id: int, position: int, parent: int) -> None:
        if self.round_id != round_id:
            self.start_round(round_id)
        self.parents[position] = parent
        self.last_touch = time.monotonic()

    def start_round(self, round_id: int) -> None:
        self.round_id = round_id
        self.ran.clear()
        self.parents.clear()
        for layer in self.layers:
            layer.clear_round()

    def ancestors(self, position: int) -> list[int]:
        """Positions from the round's root down to `position`, from parent
        pointers alone."""
        chain = [position]
        seen = {position}
        while True:
            parent = self.parents.get(chain[-1], chain[-1])
            if parent == chain[-1] or parent in seen:
                break
            chain.append(parent)
            seen.add(parent)
        chain.reverse()
        return chain

    def parent_of(self, position: int) -> int:
        return self.parents.get(position, position)

    # ------------------------------------------------------------ lifecycle

    def commit_path(self, leaf: int) -> list[int]:
        """Commit the accepted leaf's ancestors and release the rest."""
        path = self.ancestors(leaf)
        for layer in self.layers:
            layer.commit(path)
        self.ran.clear()
        self.parents.clear()
        self.round_id = None
        self.last_touch = time.monotonic()
        return path

    def keep_subtree(self, root: int) -> None:
        """After a verdict marked continue, the provider keeps what is at and
        under the matched position and discards the rest."""
        keep = {root}
        frontier = [root]
        children: dict[int, list[int]] = {}
        for position, parent in self.parents.items():
            children.setdefault(parent, []).append(position)
        while frontier:
            nxt = []
            for position in frontier:
                for child in children.get(position, []):
                    if child not in keep:
                        keep.add(child)
                        nxt.append(child)
            frontier = nxt
        keep |= set(self.ancestors(root))
        for layer in self.layers:
            layer.keep_only(keep)
        self.ran &= keep
        self.parents = {p: q for p, q in self.parents.items() if p in keep}

    def expired(self, round_time_s: float, factor: float = 1.0) -> bool:
        """A round's state expires after about one round time."""
        return (time.monotonic() - self.last_touch) > max(
            0.05, round_time_s * factor
        )

    def bytes_held(self) -> int:
        return sum(layer.bytes_held() for layer in self.layers)

    def speculative_paths(self) -> int:
        """Distinct paths whose recurrent state the provider is holding, which
        is what the window is sized from."""
        return max(
            (len(layer.tree_state) for layer in self.layers),
            default=0,
        )
