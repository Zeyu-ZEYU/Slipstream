"""Baseline: split decoding without speculation.

One hidden state up and one down per token, which is what the privacy split
costs today (Section 6.1). Every round carries exactly one position, so the
round time is the fixed part of a round plus one hidden state each way, and the
acceptance length is one.
"""

from __future__ import annotations

import time

from ..client.engine import RequestState, RoundResult
from ..tree import ROOT
from .base import Method


class Baseline(Method):
    name = "Baseline"

    def run_round(self, state: RequestState) -> RoundResult:
        engine = self.engine
        started = time.monotonic()
        tree = self.new_tree(state)

        self.send_root(state, tree)
        frame = engine.scheduler.next_frame()
        tree[ROOT].sent = True
        engine.send(state, frame, self._root_hidden, first_segment_end=True)

        engine.wait_for(state, [ROOT], timeout_s=30.0)
        token = engine.target_token(state, ROOT)
        engine.close_round(state, tree, [], token, ROOT)

        return RoundResult(
            accepted_positions=[],
            accepted_tokens=[],
            bonus_token=token,
            first_segment_length=0,
            drafted=0,
            sent=1,
            continuations=0,
            round_ms=(time.monotonic() - started) * 1e3,
            commit_leaf=ROOT,
        )
