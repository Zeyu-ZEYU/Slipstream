"""The client's stream scheduler.

This is the part of the design that decides what a round waits for and what it
sends (Sections 4.2 and 4.4). It sits beside the drafter, where the path
probabilities are, and it applies one rule twice:

  Delta = c        does this token join the first segment, the prefix the
                   round waits for?
  Delta = c_u rho  does a token that failed the first test still enter the
                   stream as a background token?

It also enforces the two traffic classes on the uplink: first-segment hidden
states of any request go before background hidden states of any request, and
among background tokens the one with the highest alpha_v goes first. Bytes
handed to the transport can be neither reordered nor recalled, so the
scheduler hands it one hidden state at a time and keeps the unsent buffer
below one hidden state.

The ablations of Section 6.3 are configuration, not separate code paths:
`order="level"` is Level order, `admission="all"` is Admit all,
`priorities=False` is No priorities, `background=False` is First segment only,
and `continuation=False` is No continuation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .alpha import AlphaTable
from .codec import PrecisionPolicy
from .config import SchedulerConfig
from .costs import CostModel
from .tree import ROOT, DraftTree, Node


class Klass(str, Enum):
    FIRST_SEGMENT = "first_segment"
    BACKGROUND = "background"
    DROPPED = "dropped"


@dataclass
class Frame:
    """One hidden state waiting for the uplink."""

    request_id: int
    round_id: int
    position: int
    parent: int
    alpha: float
    klass: Klass
    precision: str
    output_precision: str
    payload_bytes: int
    first_segment_end: bool = False
    # Order within the level, used only by the Level order ablation.
    level: int = 0
    sequence: int = 0


class StreamScheduler:
    """One scheduler per client. It serves all of the client's requests, which
    is why admission prices the uplink the requests share."""

    def __init__(
        self,
        cfg: SchedulerConfig | None = None,
        cost: CostModel | None = None,
        alpha_table: AlphaTable | None = None,
        precision: PrecisionPolicy | None = None,
        window: int = 24,
        hidden_size: int = 4096,
    ) -> None:
        self.cfg = cfg or SchedulerConfig()
        self.cost = cost or CostModel()
        self.alpha = alpha_table or AlphaTable()
        self.precision = precision or PrecisionPolicy()
        self.window = window
        self.hidden_size = hidden_size

        self._queue: list[Frame] = []
        self._sequence = 0
        # Background tokens sent but without a verdict yet, per request.
        self._outstanding: dict[int, set[int]] = {}
        # Bytes the uplink has carried, split by class, for rho.
        self.first_segment_bytes = 0
        self.total_bytes = 0
        # The frame the transport is holding. The scheduler keeps the unsent
        # buffer below one hidden state, so at most one frame is in flight to
        # the socket at a time.
        self._in_socket: Frame | None = None

    # ------------------------------------------------------------ admission

    def estimate_alpha(self, tree: DraftTree, position: int) -> float:
        node = tree[position]
        if position == ROOT:
            node.alpha = 1.0
        else:
            node.alpha = self.alpha.estimate(node.q)
        return node.alpha

    def classify(self, tree: DraftTree, position: int) -> Klass:
        """Which class a freshly drafted token belongs to."""
        if position == ROOT:
            return Klass.FIRST_SEGMENT
        alpha = self.estimate_alpha(tree, position)
        node = tree[position]

        # Equation 1 with Delta = c decides the first segment.
        in_first = alpha > self.cost.first_segment_threshold
        depth_of_first_segment = sum(
            1 for n in tree.nodes if n.in_first_segment and n.position != ROOT
        )
        if depth_of_first_segment < self.cfg.min_first_segment:
            in_first = True
        if depth_of_first_segment >= self.cfg.max_first_segment:
            in_first = False
        if in_first:
            node.in_first_segment = True
            return Klass.FIRST_SEGMENT

        if not self.cfg.background:
            # First segment only: the stream is the first segment alone.
            return Klass.DROPPED

        # Equation 2 decides admission of a background token, priced at the
        # precision it will be sent at.
        precision = self.precision.uplink_precision(
            alpha, is_root=False, in_first_segment=False
        )
        scale = self.precision.uplink_cost_scale(precision)
        if self.cfg.admission == "rule":
            if alpha <= self.cost.background_threshold(scale):
                return Klass.DROPPED

        # The provider's window bounds what the client may have outstanding.
        outstanding = len(self._outstanding.get(tree.round_id, set()))
        if outstanding >= self.window:
            return Klass.DROPPED
        return Klass.BACKGROUND

    # -------------------------------------------------------------- queueing

    def enqueue(self, tree: DraftTree, position: int, request_id: int) -> Frame | None:
        """Classify a drafted token and, if it is admitted, put its hidden
        state in the uplink queue."""
        klass = self.classify(tree, position)
        if klass is Klass.DROPPED:
            return None
        node: Node = tree[position]
        alpha = node.alpha
        precision = self.precision.uplink_precision(
            alpha,
            is_root=(position == ROOT),
            in_first_segment=(klass is Klass.FIRST_SEGMENT),
        )
        from .codec import payload_bytes

        self._sequence += 1
        frame = Frame(
            request_id=request_id,
            round_id=tree.round_id,
            position=position,
            parent=node.parent,
            alpha=alpha,
            klass=klass,
            precision=precision,
            output_precision=self.precision.downlink_precision(precision),
            payload_bytes=payload_bytes(
                self.hidden_size, precision, self.precision.cfg.fp4_block
            ),
            level=node.depth,
            sequence=self._sequence,
        )
        self._queue.append(frame)
        node.admitted = True
        return frame

    def _sort_key(self, frame: Frame):
        if not self.cfg.priorities:
            # No priorities: arrival order, one queue for both classes.
            return (frame.sequence,)
        klass_rank = 0 if frame.klass is Klass.FIRST_SEGMENT else 1
        if self.cfg.order == "level":
            # Level order: the tree goes level by level instead of by alpha.
            return (klass_rank, frame.level, frame.sequence)
        return (klass_rank, -frame.alpha, frame.sequence)

    def next_frame(self) -> Frame | None:
        """The next hidden state to hand to the transport, or None when the
        queue is empty or the transport is still holding one."""
        if self.cfg.one_frame_at_a_time and self._in_socket is not None:
            return None
        if not self._queue:
            return None
        self._queue.sort(key=self._sort_key)
        frame = self._queue.pop(0)
        self._in_socket = frame
        return frame

    def push_back(self, frame: Frame) -> None:
        """Return a claimed frame to the queue without sending it. Used when a
        caller pops a frame of the wrong class, for instance the first
        background frame after the first segment."""
        if self._in_socket is frame:
            self._in_socket = None
        self._queue.append(frame)

    def count_pending(self, klass: Klass) -> int:
        return sum(1 for f in self._queue if f.klass is klass)

    def on_sent(self, frame: Frame) -> None:
        """The transport has taken the frame: account for it and release the
        one-frame buffer."""
        self._in_socket = None
        self.total_bytes += frame.payload_bytes
        if frame.klass is Klass.FIRST_SEGMENT:
            self.first_segment_bytes += frame.payload_bytes
        else:
            self._outstanding.setdefault(frame.round_id, set()).add(frame.position)
        self.cost.observe_uplink_share(self.first_segment_bytes, self.total_bytes)

    def pending(self) -> list[Frame]:
        return list(self._queue)

    def has_first_segment_pending(self) -> bool:
        return any(f.klass is Klass.FIRST_SEGMENT for f in self._queue)

    def mark_first_segment_end(self, frame: Frame) -> None:
        """The client sets the only marker of the protocol on the last hidden
        state of the first segment. It knows when it hands that state to the
        link whether the next token passes the test, because the drafter stays
        a step ahead."""
        frame.first_segment_end = True

    def drop_round(self, round_id: int, keep: set[int] | None = None) -> int:
        """When a round closes, unsent background tokens of that round leave
        the queue, as do those under a rejected position. Returns how many
        frames were dropped."""
        before = len(self._queue)
        self._queue = [
            f
            for f in self._queue
            if f.round_id != round_id or (keep is not None and f.position in keep)
        ]
        self._outstanding.pop(round_id, None)
        return before - len(self._queue)

    def drop_positions(self, round_id: int, positions: set[int]) -> int:
        before = len(self._queue)
        self._queue = [
            f
            for f in self._queue
            if not (f.round_id == round_id and f.position in positions)
        ]
        return before - len(self._queue)

    def on_verdict(self, round_id: int, positions: list[int]) -> None:
        """A verdict clears the outstanding set for the positions it covers."""
        out = self._outstanding.get(round_id)
        if out:
            out.difference_update(positions)

    def set_window(self, window: int) -> None:
        """The provider announces the window with every output."""
        self.window = max(0, int(window))

    # ------------------------------------------------------------ reporting

    def first_segment_length(self, tree: DraftTree) -> int:
        """`s`, draft tokens in the first segment, not counting the root."""
        return sum(1 for n in tree.nodes if n.in_first_segment and n.position != ROOT)

    def snapshot(self) -> dict:
        return {
            "queued": len(self._queue),
            "window": self.window,
            "outstanding": {k: len(v) for k, v in self._outstanding.items()},
            "first_segment_byte_share": (
                self.first_segment_bytes / self.total_bytes if self.total_bytes else 0.0
            ),
            "order": self.cfg.order,
            "admission": self.cfg.admission,
            "priorities": self.cfg.priorities,
        }
