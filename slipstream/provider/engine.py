"""The provider's engine.

It holds the middle layers only, and its scheduler is the queue policy of
Section 4.3. What it may keep is fixed by Appendix C: per round the tree it has
received so far, each position's parent and whether its hidden state has
arrived and run, and the key and value entries and recurrent states of the
positions it has run. It never holds a token identifier or a probability.

The engine is driven by whoever owns the connection: hand it arrivals with
`on_hidden_state` and `on_verdict`, then call `step()` until it returns
nothing. Each `step()` runs at most one batch and returns the outputs it
produced, foreground outputs first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..codec import decode, encode
from ..config import ProviderConfig
from ..costs import OverheadCounters
from ..models.split_model import ProviderShard
from ..transport.messages import HiddenStateMsg, OutputMsg, VerdictMsg, now_ns
from .queues import Arrival, Batch, Klass, ProviderScheduler


@dataclass
class _Landed:
    arrival: Arrival
    ran: bool = False


class ProviderEngine:
    def __init__(
        self,
        shard: ProviderShard,
        cfg: ProviderConfig | None = None,
        overhead: OverheadCounters | None = None,
        free_state_bytes: int | None = None,
    ) -> None:
        self.shard = shard
        self.cfg = cfg or ProviderConfig()
        self.queues = ProviderScheduler(self.cfg)
        self.overhead = overhead or OverheadCounters()
        self.free_state_bytes = free_state_bytes
        # Everything that has landed for a round, whether it has run or not.
        self._landed: dict[tuple[int, int], dict[int, _Landed]] = {}
        self._received_count: dict[int, int] = {}
        self._clients: dict[int, int] = {}
        self._round_time_s = 0.25
        self.batches_run = 0
        self.tokens_run = 0
        self.passengers_run = 0

    # ------------------------------------------------------------- arrivals

    def on_hidden_state(self, msg: HiddenStateMsg) -> None:
        """A hidden state has landed."""
        self._clients[msg.request_id] = msg.client_id
        self._received_count[msg.request_id] = (
            self._received_count.get(msg.request_id, 0) + 1
        )
        cache = self.shard.cache_for(msg.request_id)

        # A round closes without a verdict: the first hidden state of a new
        # round names the accepted leaf of the previous one, and the provider
        # commits that leaf's ancestors and releases the rest (Appendix C).
        if msg.commits_round_id and msg.commits_leaf >= 0:
            if cache.round_id == msg.commits_round_id:
                self.shard.commit(msg.request_id, msg.commits_leaf)
            self.queues.forget_round(msg.request_id, msg.commits_round_id)
            self._landed.pop((msg.request_id, msg.commits_round_id), None)

        if cache.round_id != msg.round_id:
            cache.start_round(msg.round_id)
        cache.note_arrival(msg.round_id, msg.position, msg.parent)

        arrival = Arrival(
            client_id=msg.client_id,
            request_id=msg.request_id,
            round_id=msg.round_id,
            position=msg.position,
            parent=msg.parent,
            klass=Klass(msg.traffic_class),
            first_segment_end=msg.first_segment_end,
            payload=msg.payload,
            num_elements=msg.num_elements,
            payload_precision=msg.payload_precision,
            output_precision=msg.output_precision,
            arrival_time_ns=now_ns(),
        )
        key = (msg.request_id, msg.round_id)
        self._landed.setdefault(key, {})[msg.position] = _Landed(arrival)
        self.queues.on_arrival(arrival)

    def on_verdict(self, msg: VerdictMsg) -> None:
        """Apply a verdict: keep what is at and under the last accepted
        position, discard the rest."""
        started = time.perf_counter()
        key = (msg.request_id, msg.round_id)
        landed = self._landed.get(key, {})
        cache = self.shard.cache_for(msg.request_id)

        if msg.end_of_request:
            self.shard.release(msg.request_id)
            self.queues.forget_round(*key)
            self._landed.pop(key, None)
            self.overhead.add(
                "apply_verdict_us", (time.perf_counter() - started) * 1e6
            )
            return

        leaf = msg.accepted[-1] if msg.accepted else 0
        keep = set(_subtree(cache.parents, leaf)) | set(cache.ancestors(leaf))
        discard = {p for p in landed if p not in keep}
        unrun = [
            entry.arrival
            for position, entry in landed.items()
            if position in keep and not entry.ran
        ]
        unrun.sort(key=lambda a: a.sequence)
        if msg.continue_round:
            cache.keep_subtree(leaf)
        self.queues.on_verdict(
            request_id=msg.request_id,
            round_id=msg.round_id,
            accepted=msg.accepted,
            continue_round=msg.continue_round,
            unrun_under_accepted=unrun,
            discard=discard,
        )
        self.overhead.add("apply_verdict_us", (time.perf_counter() - started) * 1e6)

    # ------------------------------------------------------------ execution

    def step(self) -> list[OutputMsg]:
        """Run at most one batch and return its outputs."""
        started = time.perf_counter()
        batch = self.queues.next_batch()
        if batch is None:
            return []
        self.overhead.add(
            "form_foreground_batch_us" if batch.foreground else "passenger_selection_us",
            (time.perf_counter() - started) * 1e6,
        )
        if batch.passengers:
            self.overhead.add("passenger_selection_us", 0.0)
        return self._run_batch(batch)

    def run_until_idle(self, max_batches: int = 1024) -> list[OutputMsg]:
        outputs: list[OutputMsg] = []
        for _ in range(max_batches):
            produced = self.step()
            if not produced:
                break
            outputs.extend(produced)
        return outputs

    def _run_batch(self, batch: Batch) -> list[OutputMsg]:
        arrivals = [
            a
            for a in batch.all_arrivals()
            if not self._landed.get((a.request_id, a.round_id), {})
            .get(a.position, _Landed(a))
            .ran
        ]
        if not arrivals:
            return []
        # Parents before children, which the stream already guarantees.
        arrivals = _topological(arrivals)
        batch_start = now_ns()

        hidden = torch.stack(
            [
                decode(a.payload, a.payload_precision, a.num_elements)
                for a in arrivals
            ]
        )
        outputs_hidden = self.shard.run(
            request_id=batch.request_id,
            positions=[a.position for a in arrivals],
            parents=[a.parent for a in arrivals],
            hidden=hidden,
            round_id=batch.round_id,
        )
        self.batches_run += 1
        self.tokens_run += len(arrivals)
        self.passengers_run += len(batch.passengers)

        window = self.queues.window_for(
            active_clients=max(1, len(self.queues.clients_with_round)),
            free_state_bytes=self.free_state_bytes,
        )
        out: list[OutputMsg] = []
        for arrival, row in zip(arrivals, outputs_hidden):
            entry = self._landed.get((arrival.request_id, arrival.round_id), {}).get(
                arrival.position
            )
            if entry is not None:
                entry.ran = True
            precision = arrival.output_precision or "bf16"
            out.append(
                OutputMsg(
                    request_id=arrival.request_id,
                    round_id=arrival.round_id,
                    position=arrival.position,
                    payload=encode(row, precision, self.cfg_fp4_block()),
                    num_elements=int(row.numel()),
                    payload_precision=precision,
                    window=window,
                    received_count=self._received_count.get(arrival.request_id, 0),
                    arrival_time_ns=arrival.arrival_time_ns,
                    batch_start_time_ns=batch_start,
                    traffic_class=arrival.klass.value,
                    client_id=self._clients.get(arrival.request_id, 0),
                )
            )
        # Foreground outputs go down first.
        out.sort(key=lambda o: 0 if o.traffic_class == "first_segment" else 1)
        return out

    def cfg_fp4_block(self) -> int:
        return 32

    # -------------------------------------------------------------- upkeep

    def expire_rounds(self, round_time_s: float | None = None) -> int:
        """A round's state expires after about one round time, and hidden
        states of an expired round are dropped."""
        round_time_s = round_time_s or self._round_time_s
        expired = 0
        for request_id, cache in list(self.shard.caches.items()):
            if cache.round_id is None:
                continue
            if cache.expired(round_time_s, self.cfg.round_expiry_round_times):
                self.queues.forget_round(request_id, cache.round_id)
                self._landed.pop((request_id, cache.round_id), None)
                cache.start_round(cache.round_id)
                expired += 1
        return expired

    def state_bytes(self) -> int:
        return sum(cache.bytes_held() for cache in self.shard.caches.values())

    def snapshot(self) -> dict:
        return {
            "queues": self.queues.snapshot(),
            "batches_run": self.batches_run,
            "tokens_run": self.tokens_run,
            "passengers_run": self.passengers_run,
            "state_bytes": self.state_bytes(),
            "requests": len(self.shard.caches),
        }


def _subtree(parents: dict[int, int], root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for position, parent in parents.items():
        if position != parent:
            children.setdefault(parent, []).append(position)
    out, frontier = [root], [root]
    while frontier:
        nxt = []
        for position in frontier:
            for child in children.get(position, []):
                out.append(child)
                nxt.append(child)
        frontier = nxt
    return out


def _topological(arrivals: list[Arrival]) -> list[Arrival]:
    """Order a batch so that a parent precedes its children."""
    by_position = {a.position: a for a in arrivals}
    depth: dict[int, int] = {}

    def compute(position: int, guard: int = 0) -> int:
        if position in depth:
            return depth[position]
        arrival = by_position[position]
        if arrival.parent == position or arrival.parent not in by_position or guard > 64:
            depth[position] = 0
        else:
            depth[position] = compute(arrival.parent, guard + 1) + 1
        return depth[position]

    for position in list(by_position):
        compute(position)
    return sorted(arrivals, key=lambda a: (depth[a.position], a.sequence))
