"""The three messages of the protocol, as plain Python objects.

These mirror `proto/slipstream.proto` field for field (Appendix C). The
engines only ever see these; a transport turns them into protocol-buffer
frames (`grpc_channel.py`), into shared-memory records for the Rust gateway
(`shm.py`), or passes them straight through in a single process
(`loopback.py`).

Keeping the message classes free of the generated code is what lets the tests
and the single-node functional run work without a protobuf toolchain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def now_ns() -> int:
    return time.monotonic_ns()


@dataclass
class HiddenStateMsg:
    """One obfuscated hidden state on its way up."""

    request_id: int
    round_id: int
    position: int
    parent: int
    payload: bytes
    num_elements: int
    payload_precision: str = "bf16"
    output_precision: str = "bf16"
    first_segment_end: bool = False
    traffic_class: str = "first_segment"
    send_time_ns: int = field(default_factory=now_ns)
    # Round whose accepted path this hidden state commits, if any. A round
    # closes without a verdict: the next round's first hidden state names the
    # accepted leaf through `parent`.
    commits_round_id: int = 0
    commits_leaf: int = -1
    client_id: int = 0

    def wire_bytes(self) -> int:
        return len(self.payload)


@dataclass
class OutputMsg:
    """One output of the middle on its way down."""

    request_id: int
    round_id: int
    position: int
    payload: bytes
    num_elements: int
    payload_precision: str = "bf16"
    window: int = 24
    received_count: int = 0
    arrival_time_ns: int = 0
    batch_start_time_ns: int = 0
    traffic_class: str = "first_segment"
    # Which client's request this output belongs to. The wire protocol has no
    # such field: the connection is per client, so the gateway already knows.
    # A provider engine shared by several clients in one process needs it to
    # route the output to the right transport.
    client_id: int = 0

    def wire_bytes(self) -> int:
        return len(self.payload)


@dataclass
class VerdictMsg:
    """The positions the client accepted. The only thing that goes up besides
    hidden states, and it carries no token and no probability."""

    request_id: int
    round_id: int
    accepted: list[int]
    continue_round: bool = False
    end_of_request: bool = False
    send_time_ns: int = field(default_factory=now_ns)
    client_id: int = 0

    def wire_bytes(self) -> int:
        # Identifiers and a handful of positions.
        return 16 + 4 * len(self.accepted)
