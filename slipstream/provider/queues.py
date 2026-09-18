"""Opportunistic execution: the provider's half of the design.

Section 4.3 in full. The provider never knows whether a background token will
be needed, but it does know when a client is waiting, because only two
arrivals put work on a client's critical path:

  * the marked end of a first segment, after which the client waits for the
    segment's outputs;
  * a verdict marked continue, after which the client waits for the outputs of
    the accepted position and the tokens under it that have not run yet.

It keeps two queues, each first come first served across clients. The
foreground queue holds that waited-for work. The background queue holds
background tokens that have arrived but have no verdict yet, in the order the
client streamed them, which is the client's order of acceptance probability.

While the foreground queue is non-empty the provider runs foreground work and
fills each foreground batch with up to `B` passengers taken from the head of
the background queue, of any client. When the foreground queue is empty it
runs background work in batches of at most `B`, checking the foreground queue
before every batch. The two extremes of the policy are the two obvious
designs, eager and lazy, which Section 6.3 ablates.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ..config import ProviderConfig


class Klass(str, Enum):
    FIRST_SEGMENT = "first_segment"
    BACKGROUND = "background"


@dataclass
class Arrival:
    """One hidden state that has landed but has not run yet."""

    client_id: int
    request_id: int
    round_id: int
    position: int
    parent: int
    klass: Klass
    first_segment_end: bool
    payload: object
    num_elements: int
    payload_precision: str
    output_precision: str
    arrival_time_ns: int
    sequence: int = 0


@dataclass
class Batch:
    """What the provider runs next."""

    request_id: int
    client_id: int
    round_id: int
    arrivals: list[Arrival]
    # True when a client is waiting for this batch's outputs.
    foreground: bool
    # Passengers riding in a foreground batch: background tokens, of any
    # client, that the provider runs instead of giving them a batch of their
    # own. They stay background tokens.
    passengers: list[Arrival] = field(default_factory=list)

    def all_arrivals(self) -> list[Arrival]:
        return self.arrivals + self.passengers


class ProviderScheduler:
    def __init__(self, cfg: ProviderConfig | None = None) -> None:
        self.cfg = cfg or ProviderConfig()
        # Waited-for work, ready to run: groups of arrivals per (request,
        # round), in the order they became ready.
        self._foreground: deque[list[Arrival]] = deque()
        # Background tokens, in the order the clients streamed them.
        self._background: deque[Arrival] = deque()
        # Arrivals of a round that have landed but whose segment has not been
        # marked complete yet.
        self._staging: dict[tuple[int, int], list[Arrival]] = {}
        self._counter = itertools.count(1)
        # Positions the provider has been told to discard; their hidden states
        # are dropped if they arrive later.
        self._discarded: dict[tuple[int, int], set[int]] = {}
        self.dropped = 0
        self.clients_with_round: set[int] = set()

    # ------------------------------------------------------------- arrivals

    def on_arrival(self, arrival: Arrival) -> None:
        """Place one hidden state. First-segment tokens stage until the marker
        arrives; background tokens join the background queue immediately."""
        key = (arrival.request_id, arrival.round_id)
        if arrival.position in self._discarded.get(key, set()):
            self.dropped += 1
            return
        arrival.sequence = next(self._counter)
        self.clients_with_round.add(arrival.client_id)

        if arrival.klass is Klass.FIRST_SEGMENT:
            self._staging.setdefault(key, []).append(arrival)
            if arrival.first_segment_end:
                # The segment runs as one batch as soon as the marked hidden
                # state arrives: running each token as it landed would cost a
                # pass per token.
                self._foreground.append(self._staging.pop(key))
            return

        if self.cfg.policy == "eager":
            # Eager execution: every background token runs as it arrives.
            self._foreground.append([arrival])
            return
        self._background.append(arrival)

    def on_verdict(
        self,
        request_id: int,
        round_id: int,
        accepted: list[int],
        continue_round: bool,
        unrun_under_accepted: list[Arrival],
        discard: set[int],
    ) -> None:
        """A verdict moves the accepted position and the tokens under it that
        have not run yet into the foreground queue, and discards the rest."""
        key = (request_id, round_id)
        self._discarded.setdefault(key, set()).update(discard)
        self._background = deque(
            a
            for a in self._background
            if not ((a.request_id, a.round_id) == key and a.position in discard)
        )
        staged = self._staging.get(key)
        if staged:
            self._staging[key] = [a for a in staged if a.position not in discard]
        if continue_round and unrun_under_accepted:
            for arrival in unrun_under_accepted:
                arrival.klass = Klass.FIRST_SEGMENT
            self._foreground.append(list(unrun_under_accepted))

    def forget_round(self, request_id: int, round_id: int) -> None:
        key = (request_id, round_id)
        self._staging.pop(key, None)
        self._discarded.pop(key, None)
        self._background = deque(
            a for a in self._background if (a.request_id, a.round_id) != key
        )

    # ------------------------------------------------------------ scheduling

    def has_work(self) -> bool:
        return bool(self._foreground) or bool(self._background)

    @property
    def foreground_depth(self) -> int:
        return len(self._foreground)

    @property
    def background_depth(self) -> int:
        return len(self._background)

    def next_batch(self) -> Batch | None:
        """The next batch to run, following the policy."""
        if self._foreground:
            group = self._foreground.popleft()
            batch = Batch(
                request_id=group[0].request_id,
                client_id=group[0].client_id,
                round_id=group[0].round_id,
                arrivals=group,
                foreground=True,
            )
            batch.passengers = self._take_passengers(batch)
            return batch

        if self.cfg.policy == "lazy":
            # Lazy execution: background tokens run only as passengers, so
            # with an empty foreground queue there is nothing to do.
            return None

        if not self._background:
            return None
        # A background batch takes at most B tokens, in queue order, and only
        # from one request, since a batch shares one cache.
        head = self._background[0]
        group: list[Arrival] = []
        rest: deque[Arrival] = deque()
        for arrival in self._background:
            same = (
                arrival.request_id == head.request_id
                and arrival.round_id == head.round_id
            )
            if same and len(group) < self.cfg.batch_passengers:
                group.append(arrival)
            else:
                rest.append(arrival)
        self._background = rest
        return Batch(
            request_id=head.request_id,
            client_id=head.client_id,
            round_id=head.round_id,
            arrivals=group,
            foreground=False,
        )

    def _take_passengers(self, batch: Batch) -> list[Arrival]:
        """Fill a foreground batch from the head of the background queue.

        The provider cannot price a passenger by its acceptance probability,
        which only the client knows, so it bounds passengers by count; every
        passenger has passed the client's admission test, and the queue order
        is the client's order, so the heads of the streams are the tokens most
        likely to be needed.

        A batch shares one request's cache, so only background tokens of the
        same request and round can ride along.
        """
        if self.cfg.batch_passengers <= 0 or not self._background:
            return []
        taken: list[Arrival] = []
        rest: deque[Arrival] = deque()
        for arrival in self._background:
            fits = (
                arrival.request_id == batch.request_id
                and arrival.round_id == batch.round_id
                and len(taken) < self.cfg.batch_passengers
            )
            if fits:
                taken.append(arrival)
            else:
                rest.append(arrival)
        self._background = rest
        return taken

    # ---------------------------------------------------------- the window

    def window_for(self, active_clients: int, free_state_bytes: int | None) -> int:
        """The provider announces a window to each client: it divides its free
        speculative-state budget equally among the clients with a round in
        progress and sends each share with the outputs.

        With no budget configured the window is the fixed value the paper
        reports.
        """
        budget = (
            free_state_bytes
            if free_state_bytes is not None
            else self.cfg.state_budget_bytes
        )
        if budget is None:
            return self.cfg.window
        clients = max(1, active_clients)
        per_client = budget / clients
        paths = int(per_client // max(1, self.cfg.state_bytes_per_path))
        return max(0, min(self.cfg.window, paths))

    def snapshot(self) -> dict:
        return {
            "policy": self.cfg.policy,
            "foreground_groups": len(self._foreground),
            "background_tokens": len(self._background),
            "staged_rounds": len(self._staging),
            "dropped": self.dropped,
            "passengers_per_batch": self.cfg.batch_passengers,
        }
