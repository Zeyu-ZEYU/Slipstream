"""The client's engine.

The client is the trusted side. It runs the encoder and the decoder, in this
deployment the model's first layer and its language-model head, it holds the
drafter, and it is the only side that can turn a hidden state into a token and
compare it with the draft (Section 4.1, "Constraint"). Every decision about
the tree is made here and communicated as positions.

This module provides the primitives a round is built from. The five comparison
methods of Section 6.1 live in `slipstream/methods/` and differ only in how
they use them:

  encode           run the first layer over a batch of tree positions
  send             hand one hidden state to the transport, in stream order
  collect          take whatever outputs have come back
  target_token     apply the head to an output and read the target's token
  verify           walk the accepted path and find the first miss
  close_round      commit the accepted leaf, without a verdict
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from ..alpha import AlphaTable
from ..codec import PrecisionPolicy, decode, encode
from ..config import Config
from ..costs import CostModel, OverheadCounters
from ..models.cache import RequestCache
from ..models.mtp import MTPDrafter
from ..models.obfuscation import Obfuscation
from ..models.split_model import ClientShard
from ..scheduler import Frame, Klass, StreamScheduler
from ..transport.messages import HiddenStateMsg, OutputMsg, VerdictMsg
from ..tree import ROOT, DraftTree


@dataclass
class RequestState:
    """One request in flight on this client."""

    request_id: int
    prompt: list[int]
    max_output_tokens: int
    tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    round_id: int = 0
    # Positions of the current round whose outputs have come back.
    outputs: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    # The target hidden state the drafter seeds from, at the last position the
    # provider has run for us.
    seed_hidden: torch.Tensor | None = None
    # The round and leaf the next hidden state should commit.
    pending_commit: tuple[int, int] | None = None
    cache: RequestCache | None = None
    # Absolute position of the next token, which is what rotary embeddings
    # need. The prompt occupies [0, len(prompt)).
    committed_positions: int = 0
    first_token_time: float | None = None
    last_token_time: float | None = None
    start_time: float = field(default_factory=time.monotonic)
    rounds: int = 0
    arrival_times_ns: list[int] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return len(self.output_tokens) >= self.max_output_tokens

    @property
    def last_token(self) -> int:
        return self.tokens[-1]


@dataclass
class RoundResult:
    accepted_positions: list[int]
    accepted_tokens: list[int]
    bonus_token: int | None
    first_segment_length: int
    drafted: int
    sent: int
    continuations: int
    round_ms: float
    commit_leaf: int

    @property
    def advanced(self) -> int:
        """Tokens the round added to the output: the accepted draft tokens and
        the bonus token, which is the acceptance length tau."""
        return len(self.accepted_tokens) + (1 if self.bonus_token is not None else 0)


class ClientEngine:
    def __init__(
        self,
        shard: ClientShard,
        drafter: MTPDrafter,
        transport,
        cfg: Config | None = None,
        overhead: OverheadCounters | None = None,
    ) -> None:
        self.shard = shard
        self.drafter = drafter
        self.transport = transport
        self.cfg = cfg or Config()
        self.overhead = overhead or OverheadCounters()

        self.alpha = AlphaTable(self.cfg.alpha)
        self.cost = CostModel(self.cfg.cost)
        self.precision = PrecisionPolicy(self.cfg.compression)
        self.obfuscation = Obfuscation(
            kind=self.cfg.model.obfuscation, epsilon=self.cfg.model.ldp_epsilon
        )
        self.scheduler = StreamScheduler(
            cfg=self.cfg.scheduler,
            cost=self.cost,
            alpha_table=self.alpha,
            precision=self.precision,
            window=self.cfg.provider.window,
            hidden_size=shard.hidden_size(),
        )
        # Statistics the experiment drivers read.
        self.stats = {
            "rounds": 0,
            "drafted": 0,
            "sent": 0,
            "accepted": 0,
            "continuations": 0,
            "bytes_up": 0,
            "first_segment_lengths": [],
            "tau": [],
            "emitted": [],
        }

    # ------------------------------------------------------------- encoding

    def new_request(self, request_id: int, prompt: list[int], max_output_tokens: int) -> RequestState:
        state = RequestState(
            request_id=request_id,
            prompt=list(prompt),
            max_output_tokens=max_output_tokens,
            tokens=list(prompt),
            cache=RequestCache(num_layers=len(self.shard.layers), request_id=request_id),
        )
        return state

    def encode(
        self,
        state: RequestState,
        tokens: list[int],
        positions: list[int],
        parents: list[int],
        absolute: list[int],
    ) -> torch.Tensor:
        """Run the first layer over a batch of tree positions and obfuscate the
        result, which is what leaves the client."""
        hidden = self.shard.encode(
            token_ids=tokens,
            positions=positions,
            parents=parents,
            absolute_positions=absolute,
            cache=state.cache,
            round_id=state.round_id,
        )
        return torch.stack([self.obfuscation.apply(row) for row in hidden])

    # --------------------------------------------------------------- sending

    def send(
        self,
        state: RequestState,
        frame: Frame,
        hidden: torch.Tensor,
        *,
        first_segment_end: bool = False,
    ) -> None:
        """Hand one hidden state to the transport, in stream order."""
        started = time.perf_counter()
        payload = encode(hidden, frame.precision, self.cfg.compression.fp4_block)
        # What the hand-off itself costs: encoding the payload and filling in
        # the header. The time the frame then spends on the link is the link's,
        # not the design's, so it is not counted here (Section 6.7).
        msg = HiddenStateMsg(
            request_id=state.request_id,
            round_id=frame.round_id,
            position=frame.position,
            parent=frame.parent,
            payload=payload,
            num_elements=int(hidden.numel()),
            payload_precision=frame.precision,
            output_precision=frame.output_precision,
            first_segment_end=first_segment_end or frame.first_segment_end,
            traffic_class=frame.klass.value,
        )
        if state.pending_commit is not None:
            msg.commits_round_id, msg.commits_leaf = state.pending_commit
            state.pending_commit = None
        self.overhead.add("socket_handoff_us", (time.perf_counter() - started) * 1e6)
        self.transport.send_hidden_state(msg)
        self.scheduler.on_sent(frame)
        self.stats["sent"] += 1
        self.stats["bytes_up"] += len(payload)

    def send_verdict(
        self,
        state: RequestState,
        accepted: list[int],
        *,
        continue_round: bool,
        end_of_request: bool = False,
    ) -> None:
        self.transport.send_verdict(
            VerdictMsg(
                request_id=state.request_id,
                round_id=state.round_id,
                accepted=list(accepted),
                continue_round=continue_round,
                end_of_request=end_of_request,
            )
        )
        self.scheduler.on_verdict(state.round_id, accepted)

    # -------------------------------------------------------------- receiving

    def collect(self, state: RequestState, timeout_s: float = 0.0) -> int:
        """Take whatever outputs have come back and decode them."""
        messages = self.transport.poll_outputs(timeout_s)
        for msg in messages:
            self._absorb(state, msg)
        return len(messages)

    def _absorb(self, state: RequestState, msg: OutputMsg) -> None:
        hidden = decode(
            msg.payload, msg.payload_precision, msg.num_elements, self.cfg.compression.fp4_block
        )
        state.outputs[(msg.round_id, msg.position)] = hidden
        self.scheduler.set_window(msg.window)
        if msg.arrival_time_ns:
            state.arrival_times_ns.append(msg.arrival_time_ns)

    def wait_for(
        self, state: RequestState, positions: list[int], timeout_s: float = 5.0
    ) -> bool:
        """Wait until the outputs of `positions` are in, or time out."""
        deadline = time.monotonic() + timeout_s
        while True:
            if all((state.round_id, p) in state.outputs for p in positions):
                return True
            if time.monotonic() >= deadline:
                return False
            self.collect(state, timeout_s=0.002)

    def output_of(self, state: RequestState, position: int) -> torch.Tensor | None:
        return state.outputs.get((state.round_id, position))

    # ------------------------------------------------------------- verifying

    def target_token(self, state: RequestState, position: int) -> int | None:
        """The token the served model itself emits after `position`. Only the
        client can compute this."""
        hidden = self.output_of(state, position)
        if hidden is None:
            return None
        return self.shard.argmax_token(hidden)

    def verify(
        self, state: RequestState, tree: DraftTree, sent_positions: set[int]
    ) -> tuple[list[int], int, int | None, int | None]:
        """Walk the accepted path from the root.

        Returns the accepted positions, the position where the walk stopped,
        the target's own token there when it is known, and the position of a
        streamed token that matches it, if any. A round advances by the
        accepted draft tokens plus one bonus token, so the first miss is
        where the round ends unless continuation takes over
        (Section 4.2, "Branch continuation").
        """
        accepted: list[int] = []
        cursor = ROOT
        while True:
            token = self.target_token(state, cursor)
            if token is None:
                # The output of `cursor` has not come back: the round either
                # waits for it or stops here.
                return accepted, cursor, None, None
            match = None
            for child in tree[cursor].children:
                if child in sent_positions and tree[child].token == token:
                    match = child
                    break
            if match is None:
                return accepted, cursor, token, None
            accepted.append(match)
            cursor = match
            if self.output_of(state, match) is None:
                return accepted, match, token, match

    # ---------------------------------------------------------------- rounds

    def close_round(
        self,
        state: RequestState,
        tree: DraftTree,
        accepted: list[int],
        bonus_token: int | None,
        commit_leaf: int,
    ) -> None:
        """Finish a round.

        Closing needs no verdict: the next round's first hidden state carries a
        new round identifier and a parent pointer to the accepted leaf, and the
        provider commits that leaf's ancestors and releases the rest
        (Appendix C).
        """
        started = time.perf_counter()
        for position in accepted:
            token = tree[position].token
            if token is not None:
                state.tokens.append(token)
                state.output_tokens.append(token)
                self.stats["emitted"].append(token)
        if bonus_token is not None:
            state.tokens.append(bonus_token)
            state.output_tokens.append(bonus_token)
            self.stats["emitted"].append(int(bonus_token))
        now = time.monotonic()
        if state.first_token_time is None and state.output_tokens:
            state.first_token_time = now
        state.last_token_time = now

        # The drafter seeds from the deepest output the provider returned.
        seed = self.output_of(state, commit_leaf)
        if seed is not None:
            state.seed_hidden = seed
        if self.output_of(state, commit_leaf) is not None:
            # The provider ran this position, so it can commit the path to it.
            state.pending_commit = (state.round_id, commit_leaf)
            state.committed_positions += len(self.tree_path(tree, commit_leaf))
            if state.cache is not None:
                self.shard.commit(state.cache, commit_leaf)

        # Fold this round's verdict into the acceptance table, counting only
        # tokens whose verdict the client has seen.
        observations: list[tuple[float, bool]] = []
        accepted_set = set(accepted)
        for node in tree.nodes:
            if node.position == ROOT or not node.sent:
                continue
            if self.output_of(state, node.parent) is None:
                continue  # no verdict for this token
            observations.append((node.q, node.position in accepted_set))
        if observations:
            self.alpha.update(observations)
        self.scheduler.drop_round(state.round_id)
        state.outputs = {
            key: value for key, value in state.outputs.items() if key[0] != state.round_id
        }
        self.overhead.add(
            "between_round_estimates_us", (time.perf_counter() - started) * 1e6
        )

    @staticmethod
    def tree_path(tree: DraftTree, leaf: int) -> list[int]:
        return tree.ancestors(leaf)

    def absolute_position(self, state: RequestState, depth: int) -> int:
        """Rotary position of a node at `depth` in the current round."""
        return state.committed_positions + depth

    def note_round(self, result: RoundResult) -> None:
        self.stats["rounds"] += 1
        self.stats["drafted"] += result.drafted
        self.stats["accepted"] += len(result.accepted_tokens)
        self.stats["continuations"] += result.continuations
        self.stats["first_segment_lengths"].append(result.first_segment_length)
        self.stats["tau"].append(result.advanced)
        self.cost.observe_round(
            first_segment_len=result.first_segment_length,
            round_ms=result.round_ms,
            accepted_tokens=result.advanced,
        )

    def snapshot(self) -> dict:
        taus = self.stats["tau"]
        segments = self.stats["first_segment_lengths"]
        return {
            **{k: v for k, v in self.stats.items() if not isinstance(v, list)},
            "tau_mean": sum(taus) / len(taus) if taus else 0.0,
            "first_segment_mean": sum(segments) / len(segments) if segments else 0.0,
            "cost": self.cost.snapshot(),
            "scheduler": self.scheduler.snapshot(),
            "alpha_rounds": self.alpha.rounds_seen,
        }
