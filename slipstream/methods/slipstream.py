"""Slipstream: the round of Section 4.

One round, in the order the paper describes it:

  1. The root's hidden state needs no drafting and goes up first, while the
     first drafting step runs.
  2. The drafter expands the tree best-first, always the drafted token with
     the highest path probability that has no children yet, producing one
     token's `k` most probable children together. Each child is classified by
     Equation 1 with Delta = c: while alpha_v clears c tau / T it joins the
     first segment, the prefix the round waits for.
  3. The first segment's hidden states go up in order of alpha_v, and the
     client marks the last of them. The provider runs the segment as one batch
     when the marked hidden state arrives.
  4. Drafting continues. Tokens that failed the first test are background
     tokens; they enter the stream while Equation 2 admits them and the
     provider's window has room, and they travel behind the first segment.
  5. The round waits for the first segment's outputs, verifies them, and finds
     the accepted path and the target's own token at the first miss. If that
     token matches a streamed token, the round continues on it, again by
     Equation 1: the client sends a verdict marked continue and the provider
     runs whatever it has not run at and under the matched token.
  6. Otherwise the token is the round's bonus token and the round closes at no
     extra cost: the next round's first hidden state points to the accepted
     leaf.
"""

from __future__ import annotations

import time

from ..client.engine import RequestState, RoundResult
from ..scheduler import Klass
from ..tree import ROOT, DraftTree
from .base import Method


class Slipstream(Method):
    name = "Slipstream"

    def run_round(self, state: RequestState) -> RoundResult:
        engine = self.engine
        started = time.monotonic()
        tree = self.new_tree(state)

        # --- 1. the root goes up first -----------------------------------
        root_frame = self.send_root(state, tree)
        root_hidden = self._root_hidden

        # --- 2. draft the first segment ----------------------------------
        drafter_hidden = {ROOT: state.seed_hidden}
        drafted = 0

        while drafted < engine.cfg.drafter.max_nodes:
            candidate = self._next_candidate(tree)
            if candidate is None:
                break
            seed = drafter_hidden.get(candidate)
            if seed is None:
                break
            step_started = time.perf_counter()
            proposal = engine.drafter.propose(
                seed,
                tree[candidate].token,
                engine.absolute_position(state, tree[candidate].depth),
            )
            engine.cost.observe_draft_step((time.perf_counter() - step_started) * 1e3)
            positions = tree.add_children(candidate, proposal.tokens, proposal.probs)
            drafted += len(positions)
            for position, hidden in zip(positions, proposal.hidden):
                drafter_hidden[position] = hidden
            admitted_to_segment = False
            for position in positions:
                frame = engine.scheduler.enqueue(tree, position, state.request_id)
                if frame is not None and frame.klass is Klass.FIRST_SEGMENT:
                    admitted_to_segment = True
            if not admitted_to_segment:
                # The first segment is complete: no newly drafted token clears
                # the threshold, and acceptance along the trunk only falls
                # from here.
                break

        # --- 3. send the first segment, marking its last hidden state -----
        expected = engine.scheduler.count_pending(Klass.FIRST_SEGMENT)
        segment_positions: list[int] = []
        for index in range(expected):
            frame = engine.scheduler.next_frame()
            if frame is None:
                break
            if frame.klass is not Klass.FIRST_SEGMENT:
                engine.scheduler.push_back(frame)
                break
            hidden = (
                root_hidden
                if frame.position == ROOT
                else self.encode_node(state, tree, frame.position)
            )
            tree[frame.position].sent = True
            last = index == expected - 1
            if last:
                # Ending the first segment is one decision: the client knows,
                # when it hands this state to the link, that the next token
                # failed the test, because the drafter stays a step ahead.
                decision_started = time.perf_counter()
                engine.scheduler.mark_first_segment_end(frame)
                engine.overhead.add(
                    "end_first_segment_us",
                    (time.perf_counter() - decision_started) * 1e6,
                )
            engine.send(state, frame, hidden, first_segment_end=last)
            segment_positions.append(frame.position)

        # --- 4. keep drafting; background tokens use the idle uplink ------
        self._stream_background(state, tree, drafter_hidden)
        drafted = max(drafted, len(tree) - 1)

        # --- 5. wait for the first segment, then verify -------------------
        engine.wait_for(state, segment_positions, timeout_s=30.0)
        sent_positions = {n.position for n in tree.nodes if n.sent}
        accepted, cursor, target, matched = engine.verify(state, tree, sent_positions)

        continuations = 0
        while matched is not None and engine.cfg.scheduler.continuation:
            # Branch continuation: the target's token is a streamed token whose
            # output the provider may not have produced yet.
            gain = self._continuation_gain(tree, matched, accepted)
            wait_ms = engine.cost.background_delay_ms
            if not engine.cost.continuation_pays(gain, wait_ms):
                break
            engine.send_verdict(state, accepted, continue_round=True)
            continuations += 1
            wait_started = time.monotonic()
            if not engine.wait_for(state, [matched], timeout_s=30.0):
                break
            engine.cost.observe_background_delay((time.monotonic() - wait_started) * 1e3)
            accepted, cursor, target, matched = engine.verify(
                state, tree, sent_positions
            )

        # --- 6. close ------------------------------------------------------
        bonus = target if matched is None else None
        commit_leaf = self._deepest_with_output(state, tree, accepted)
        if bonus is not None:
            accepted_for_output = accepted
        else:
            # No bonus token: the round advanced by its accepted draft tokens,
            # and the next round's root is the last of them.
            accepted_for_output = accepted
        accepted_tokens = [tree[p].token for p in accepted_for_output]
        engine.close_round(state, tree, accepted_for_output, bonus, commit_leaf)
        engine.cost.observe_arrivals(state.arrival_times_ns)
        state.arrival_times_ns.clear()

        return RoundResult(
            accepted_positions=accepted_for_output,
            accepted_tokens=[t for t in accepted_tokens if t is not None],
            bonus_token=bonus,
            first_segment_length=max(0, len(segment_positions) - 1),
            drafted=drafted,
            sent=len(sent_positions),
            continuations=continuations,
            round_ms=(time.monotonic() - started) * 1e3,
            commit_leaf=commit_leaf,
        )

    # ------------------------------------------------------------- helpers

    def _next_candidate(self, tree: DraftTree) -> int | None:
        """The best-first candidate: the admitted token with the highest path
        probability that has no children yet.

        The stream only grows under tokens it has admitted, so a token the
        rule turned away is never expanded (Section 4.2, "Streaming from the
        drafter").
        """
        if not tree[ROOT].children:
            return ROOT
        candidates = [
            n.position
            for n in tree.nodes
            if n.admitted
            and not n.children
            and n.position != ROOT
            and n.depth < self.engine.cfg.drafter.max_depth
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (tree[p].q, -p))

    def _stream_background(
        self, state: RequestState, tree: DraftTree, drafter_hidden: dict
    ) -> int:
        """Send the background stream, drafting further while the rule and the
        window allow."""
        engine = self.engine
        sent = 0
        while True:
            frame = engine.scheduler.next_frame()
            if frame is None:
                if len(tree) - 1 >= engine.cfg.drafter.max_nodes:
                    break
                candidate = self._next_candidate(tree)
                if candidate is None:
                    break
                seed = drafter_hidden.get(candidate)
                if seed is None:
                    break
                proposal = engine.drafter.propose(
                    seed,
                    tree[candidate].token,
                    engine.absolute_position(state, tree[candidate].depth),
                )
                positions = tree.add_children(
                    candidate, proposal.tokens, proposal.probs
                )
                for position, hidden in zip(positions, proposal.hidden):
                    drafter_hidden[position] = hidden
                admitted = 0
                for position in positions:
                    if engine.scheduler.enqueue(tree, position, state.request_id):
                        admitted += 1
                if admitted == 0:
                    break
                continue
            hidden = self.encode_node(state, tree, frame.position)
            tree[frame.position].sent = True
            engine.send(state, frame, hidden)
            sent += 1
        return sent

    def _continuation_gain(
        self, tree: DraftTree, matched: int, accepted: list[int]
    ) -> float:
        """One token from the matched token's own output, plus alpha_u over
        alpha_a for every streamed token under it."""
        base = tree[matched].alpha or 1e-6
        gain = 1.0
        for position in tree.descendants(matched):
            if position == matched:
                continue
            node = tree[position]
            if node.sent:
                gain += min(1.0, node.alpha / base)
        _ = accepted
        return gain

    def _deepest_with_output(
        self, state: RequestState, tree: DraftTree, accepted: list[int]
    ) -> int:
        """The deepest accepted position whose output came back, which is what
        the provider can commit."""
        leaf = ROOT
        for position in accepted:
            if self.engine.output_of(state, position) is not None:
                leaf = position
            else:
                break
        return leaf
