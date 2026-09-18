"""Cunningham: the one prior split-inference design with speculation.

Section 6.1 and Appendix D. The method runs no drafter: it guesses n-grams
from the text so far and ships a step's guesses in one piece. The original
takes its n-grams from lookahead decoding, which also runs the model over its
lookahead window every step; in a split, those positions would cross the
network as hidden states too, on top of the guesses. This re-implementation
therefore takes its n-grams from prompt lookup, which ships only the guess:
the last two tokens are matched against the prompt and the output so far, and
the five tokens after the most recent match go up as the draft.

Because prompt lookup depends only on the text, the guesses this produces are
exactly those the live method would make, which is what lets the acceptance be
replayed from greedy outputs.
"""

from __future__ import annotations

import time

from ..client.engine import RequestState, RoundResult
from ..scheduler import Frame, Klass
from .base import Method
from ..tree import ROOT


def prompt_lookup(tokens: list[int], n: int = 2, k: int = 5) -> list[int]:
    """The `k` tokens after the most recent match of the last `n` tokens."""
    if len(tokens) <= n:
        return []
    needle = tokens[-n:]
    for start in range(len(tokens) - n - 1, -1, -1):
        if tokens[start : start + n] == needle:
            guess = tokens[start + n : start + n + k]
            if guess:
                return list(guess)
    return []


class Cunningham(Method):
    name = "Cunningham"

    def __init__(self, engine, ngram: int = 2, guesses: int = 5) -> None:
        super().__init__(engine)
        self.ngram = ngram
        self.guesses = guesses

    def run_round(self, state: RequestState) -> RoundResult:
        engine = self.engine
        started = time.monotonic()
        tree = self.new_tree(state)
        self.send_root(state, tree)

        # The draft is a chain of n-gram guesses, with no drafter and no
        # probabilities.
        guess = prompt_lookup(state.tokens, self.ngram, self.guesses)
        parent = ROOT
        for token in guess:
            positions = tree.add_children(parent, [int(token)], [1.0])
            parent = positions[0]

        shipped = tree.ancestors(parent)
        for index, position in enumerate(shipped):
            node = tree[position]
            hidden = (
                self._root_hidden
                if position == ROOT
                else self.encode_node(state, tree, position)
            )
            frame = Frame(
                request_id=state.request_id,
                round_id=tree.round_id,
                position=position,
                parent=node.parent,
                alpha=1.0,
                klass=Klass.FIRST_SEGMENT,
                precision="bf16",
                output_precision="bf16",
                payload_bytes=2 * int(hidden.numel()),
            )
            node.sent = True
            engine.send(
                state, frame, hidden, first_segment_end=(index == len(shipped) - 1)
            )

        engine.wait_for(state, shipped, timeout_s=30.0)
        accepted, cursor, target, matched = engine.verify(state, tree, set(shipped))
        bonus = target if matched is None else None
        commit_leaf = accepted[-1] if accepted else ROOT
        if engine.output_of(state, commit_leaf) is None:
            commit_leaf = ROOT
        accepted_tokens = [tree[p].token for p in accepted]
        engine.close_round(state, tree, accepted, bonus, commit_leaf)
        engine.scheduler.drop_round(tree.round_id)

        return RoundResult(
            accepted_positions=accepted,
            accepted_tokens=[t for t in accepted_tokens if t is not None],
            bonus_token=bonus,
            first_segment_length=len(shipped) - 1,
            drafted=len(guess),
            sent=len(shipped),
            continuations=0,
            round_ms=(time.monotonic() - started) * 1e3,
            commit_leaf=commit_leaf,
        )
