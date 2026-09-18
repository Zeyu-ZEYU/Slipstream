"""What the five comparison methods share.

Section 6.1: "Five methods share the split, models, and code, and differ in
what a round ships and waits for." This module is that shared part, the
prefill and the request loop; each method supplies `run_round`.

Prefill is outside both metrics, since no method changes it: the client runs
its first layer over the prompt, the hidden states go up as one segment, the
provider runs them as one batch and commits them, and the output at the last
prompt position gives the request's first output token.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..client.engine import ClientEngine, RequestState, RoundResult
from ..scheduler import Frame, Klass
from ..tree import ROOT, DraftTree


@dataclass
class RequestMetrics:
    request_id: int
    output_tokens: int
    rounds: int
    prefill_ms: float
    decode_ms: float
    tpot_ms: float
    tau_mean: float
    bytes_up: int
    bytes_down: int

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "output_tokens": self.output_tokens,
            "rounds": self.rounds,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "tpot_ms": self.tpot_ms,
            "tau_mean": self.tau_mean,
            "bytes_up": self.bytes_up,
            "bytes_down": self.bytes_down,
        }


class Method:
    """One way to spend a round."""

    name = "method"

    def __init__(self, engine: ClientEngine) -> None:
        self.engine = engine

    # --------------------------------------------------------------- prefill

    def prefill(self, state: RequestState) -> float:
        """Run the prompt and take the request's first output token."""
        engine = self.engine
        started = time.monotonic()
        state.round_id += 1
        tokens = state.prompt
        positions = list(range(len(tokens)))
        absolute = list(range(len(tokens)))
        parents = [0] + list(range(len(tokens) - 1))
        hidden = engine.encode(state, tokens, positions, parents, absolute)

        last = len(tokens) - 1
        for index, position in enumerate(positions):
            frame = Frame(
                request_id=state.request_id,
                round_id=state.round_id,
                position=position,
                parent=position - 1 if position > 0 else 0,
                alpha=1.0,
                klass=Klass.FIRST_SEGMENT,
                precision="bf16",
                output_precision="bf16",
                payload_bytes=2 * int(hidden[index].numel()),
            )
            engine.send(state, frame, hidden[index], first_segment_end=(position == last))

        if not engine.wait_for(state, [last], timeout_s=30.0):
            raise TimeoutError("prefill outputs did not come back")
        first_token = engine.target_token(state, last)
        state.tokens.append(int(first_token))
        state.output_tokens.append(int(first_token))
        state.seed_hidden = engine.output_of(state, last)
        state.first_token_time = time.monotonic()
        state.last_token_time = state.first_token_time
        state.pending_commit = (state.round_id, last)
        state.committed_positions = len(tokens)
        state.outputs.clear()
        return (time.monotonic() - started) * 1e3

    # ----------------------------------------------------------- the request

    def run_round(self, state: RequestState) -> RoundResult:
        raise NotImplementedError

    def run_request(
        self, request_id: int, prompt: list[int], max_output_tokens: int
    ) -> RequestMetrics:
        engine = self.engine
        state = engine.new_request(request_id, prompt, max_output_tokens)
        prefill_ms = self.prefill(state)
        decode_started = time.monotonic()
        taus: list[float] = []
        while not state.done:
            result = self.run_round(state)
            engine.note_round(result)
            taus.append(result.advanced)
            if result.advanced == 0:
                # A round always advances by at least one token; stop rather
                # than spin if a transport has gone away.
                break
        engine.send_verdict(state, [], continue_round=False, end_of_request=True)
        decode_ms = (time.monotonic() - decode_started) * 1e3
        produced = len(state.output_tokens)
        # Time per output token, one user's view: from the request's first
        # output token to its last, divided by the output tokens in between.
        if state.first_token_time and state.last_token_time and produced > 1:
            tpot = (state.last_token_time - state.first_token_time) * 1e3 / (produced - 1)
        else:
            tpot = 0.0
        return RequestMetrics(
            request_id=request_id,
            output_tokens=produced,
            rounds=len(taus),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            tpot_ms=tpot,
            tau_mean=sum(taus) / len(taus) if taus else 0.0,
            bytes_up=engine.transport.snapshot().get("bytes_up", 0),
            bytes_down=engine.transport.snapshot().get("bytes_down", 0),
        )

    # ----------------------------------------------------------- small helpers

    def new_tree(self, state: RequestState) -> DraftTree:
        state.round_id += 1
        return DraftTree(root_token=state.last_token, round_id=state.round_id)

    def send_root(self, state: RequestState, tree: DraftTree) -> Frame:
        """The root's hidden state needs no drafting and goes up first."""
        engine = self.engine
        hidden = engine.encode(
            state,
            [tree.root.token],
            [ROOT],
            [ROOT],
            [engine.absolute_position(state, 0)],
        )
        frame = engine.scheduler.enqueue(tree, ROOT, state.request_id)
        assert frame is not None, "the root is always admitted"
        self._root_hidden = hidden[0]
        return frame

    def encode_node(self, state: RequestState, tree: DraftTree, position: int):
        node = tree[position]
        return self.engine.encode(
            state,
            [node.token],
            [position],
            [node.parent],
            [self.engine.absolute_position(state, node.depth)],
        )[0]
