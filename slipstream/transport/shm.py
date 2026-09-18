"""The shared-memory ring the engine shares with the Rust gateway.

Section 5: "the engine hands hidden states to a Rust gateway over shared
memory, one at a time, in stream order". This is the Python half of that
hand-off. The layout is fixed and matches `gateway/src/ring.rs` byte for byte;
that file documents it, and `tests/test_ring_layout.py` checks the two agree.

Two rings make a transport: the engine writes hidden states and verdicts to
the up ring and reads outputs from the down ring, and the gateway does the
mirror image.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass

from .messages import HiddenStateMsg, OutputMsg, VerdictMsg

MAGIC = 0x534C5031  # "SLP1"
VERSION = 1
HEADER_BYTES = 64
RECORD_HEADER_BYTES = 64

KIND_HIDDEN_STATE = 1
KIND_VERDICT = 2
KIND_OUTPUT = 3

FLAG_FIRST_SEGMENT_END = 1 << 0
FLAG_BACKGROUND = 1 << 1
FLAG_CONTINUE = 1 << 2
FLAG_END_OF_REQUEST = 1 << 3

PRECISION_CODE = {"bf16": 1, "fp8": 2, "fp4": 3}
PRECISION_NAME = {1: "bf16", 2: "fp8", 3: "fp4"}

_HEADER = struct.Struct("<IIII QQ 32x")
_RECORD = struct.Struct("<Q II QQ II III I Q")


@dataclass
class Record:
    kind: int
    flags: int
    request_id: int
    round_id: int
    position: int
    parent: int
    payload_precision: int
    output_precision: int
    num_elements: int
    commits_round_id: int
    payload: bytes


class Ring:
    """One direction of the hand-off."""

    def __init__(self, path: str, slots: int = 1024, slot_bytes: int = 64 * 1024, create: bool = False):
        self.path = path
        if create:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            size = HEADER_BYTES + slots * slot_bytes
            with open(path, "wb") as fh:
                fh.write(b"\0" * size)
            self.slots, self.slot_bytes = slots, slot_bytes
        self._file = open(path, "r+b")
        self._map = mmap.mmap(self._file.fileno(), 0)
        if create:
            self._map[0:HEADER_BYTES] = _HEADER.pack(
                MAGIC, VERSION, slots, slot_bytes, 0, 0
            )
        magic, version, self.slots, self.slot_bytes, _, _ = _HEADER.unpack(
            self._map[0:HEADER_BYTES]
        )
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"not a slipstream ring: magic {magic:#x} version {version}")
        self.cursor = self._read_index()

    # -------------------------------------------------------------- indices

    def _write_index(self) -> int:
        return struct.unpack_from("<Q", self._map, 16)[0]

    def _set_write_index(self, value: int) -> None:
        struct.pack_into("<Q", self._map, 16, value)

    def _read_index(self) -> int:
        return struct.unpack_from("<Q", self._map, 24)[0]

    def _set_read_index(self, value: int) -> None:
        struct.pack_into("<Q", self._map, 24, value)

    @property
    def max_payload(self) -> int:
        return self.slot_bytes - RECORD_HEADER_BYTES

    # --------------------------------------------------------------- access

    def push(self, record: Record) -> bool:
        """Publish one record, or return False when the ring is full. A full
        ring is the back pressure that keeps the engine from running ahead of
        the link."""
        if len(record.payload) > self.max_payload:
            raise ValueError(
                f"payload of {len(record.payload)} bytes exceeds the slot's {self.max_payload}"
            )
        write, read = self._write_index(), self._read_index()
        if write - read >= self.slots:
            return False
        base = HEADER_BYTES + (write % self.slots) * self.slot_bytes
        _RECORD.pack_into(
            self._map,
            base,
            write + 1,
            record.kind,
            record.flags,
            record.request_id,
            record.round_id,
            record.position,
            record.parent,
            record.payload_precision,
            record.output_precision,
            record.num_elements,
            len(record.payload),
            record.commits_round_id,
        )
        start = base + RECORD_HEADER_BYTES
        self._map[start : start + len(record.payload)] = record.payload
        self._set_write_index(write + 1)
        return True

    def pop(self) -> Record | None:
        if self.cursor >= self._write_index():
            return None
        base = HEADER_BYTES + (self.cursor % self.slots) * self.slot_bytes
        (
            sequence,
            kind,
            flags,
            request_id,
            round_id,
            position,
            parent,
            payload_precision,
            output_precision,
            num_elements,
            payload_len,
            commits_round_id,
        ) = _RECORD.unpack_from(self._map, base)
        if sequence == 0:
            return None
        start = base + RECORD_HEADER_BYTES
        record = Record(
            kind=kind,
            flags=flags,
            request_id=request_id,
            round_id=round_id,
            position=position,
            parent=parent,
            payload_precision=payload_precision,
            output_precision=output_precision,
            num_elements=num_elements,
            commits_round_id=commits_round_id,
            payload=bytes(self._map[start : start + payload_len]),
        )
        self.cursor += 1
        self._set_read_index(self.cursor)
        return record

    def close(self) -> None:
        self._map.close()
        self._file.close()


# ------------------------------------------------------- messages to records


def hidden_state_to_record(msg: HiddenStateMsg) -> Record:
    flags = 0
    if msg.first_segment_end:
        flags |= FLAG_FIRST_SEGMENT_END
    if msg.traffic_class == "background":
        flags |= FLAG_BACKGROUND
    return Record(
        kind=KIND_HIDDEN_STATE,
        flags=flags,
        request_id=msg.request_id,
        round_id=msg.round_id,
        position=msg.position,
        parent=msg.parent,
        payload_precision=PRECISION_CODE[msg.payload_precision],
        output_precision=PRECISION_CODE[msg.output_precision],
        num_elements=msg.num_elements,
        commits_round_id=msg.commits_round_id,
        payload=msg.payload,
    )


def verdict_to_record(msg: VerdictMsg) -> Record:
    flags = 0
    if msg.continue_round:
        flags |= FLAG_CONTINUE
    if msg.end_of_request:
        flags |= FLAG_END_OF_REQUEST
    payload = b"".join(struct.pack("<I", p) for p in msg.accepted)
    return Record(
        kind=KIND_VERDICT,
        flags=flags,
        request_id=msg.request_id,
        round_id=msg.round_id,
        position=0,
        parent=0,
        payload_precision=1,
        output_precision=1,
        num_elements=len(msg.accepted),
        commits_round_id=0,
        payload=payload,
    )


def record_to_hidden_state(record: Record) -> HiddenStateMsg:
    return HiddenStateMsg(
        request_id=record.request_id,
        round_id=record.round_id,
        position=record.position,
        parent=record.parent,
        payload=record.payload,
        num_elements=record.num_elements,
        payload_precision=PRECISION_NAME[record.payload_precision],
        output_precision=PRECISION_NAME[record.output_precision],
        first_segment_end=bool(record.flags & FLAG_FIRST_SEGMENT_END),
        traffic_class="background" if record.flags & FLAG_BACKGROUND else "first_segment",
        commits_round_id=record.commits_round_id,
    )


def record_to_verdict(record: Record) -> VerdictMsg:
    accepted = [
        struct.unpack_from("<I", record.payload, offset)[0]
        for offset in range(0, len(record.payload), 4)
    ]
    return VerdictMsg(
        request_id=record.request_id,
        round_id=record.round_id,
        accepted=accepted,
        continue_round=bool(record.flags & FLAG_CONTINUE),
        end_of_request=bool(record.flags & FLAG_END_OF_REQUEST),
    )


def output_to_record(msg: OutputMsg) -> Record:
    flags = FLAG_BACKGROUND if msg.traffic_class == "background" else 0
    return Record(
        kind=KIND_OUTPUT,
        flags=flags,
        request_id=msg.request_id,
        round_id=msg.round_id,
        position=msg.position,
        # The gateway passes the window on to the client in this field.
        parent=msg.window,
        payload_precision=PRECISION_CODE[msg.payload_precision],
        output_precision=PRECISION_CODE[msg.payload_precision],
        num_elements=msg.num_elements,
        commits_round_id=msg.arrival_time_ns,
        payload=msg.payload,
    )


def record_to_output(record: Record) -> OutputMsg:
    return OutputMsg(
        request_id=record.request_id,
        round_id=record.round_id,
        position=record.position,
        payload=record.payload,
        num_elements=record.num_elements,
        payload_precision=PRECISION_NAME[record.payload_precision],
        window=record.parent,
        arrival_time_ns=record.commits_round_id,
        traffic_class="background" if record.flags & FLAG_BACKGROUND else "first_segment",
    )


# ------------------------------------------------------------- the transport


class ShmTransport:
    """What the client's engine talks to when the Rust gateway carries the
    traffic. The engine never touches a socket."""

    def __init__(
        self,
        up_path: str,
        down_path: str,
        slots: int = 1024,
        slot_bytes: int = 64 * 1024,
        create: bool = True,
        client_id: int = 0,
    ) -> None:
        self.up = Ring(up_path, slots, slot_bytes, create=create)
        self.down = Ring(down_path, slots, slot_bytes, create=create)
        self.client_id = client_id
        self.bytes_up = 0
        self.bytes_down = 0
        self.frames_up = 0

    def send_hidden_state(self, msg: HiddenStateMsg) -> None:
        record = hidden_state_to_record(msg)
        while not self.up.push(record):
            time.sleep(0.0002)
        self.bytes_up += len(record.payload)
        self.frames_up += 1

    def send_verdict(self, msg: VerdictMsg) -> None:
        record = verdict_to_record(msg)
        while not self.up.push(record):
            time.sleep(0.0002)

    def poll_outputs(self, timeout_s: float = 0.0) -> list[OutputMsg]:
        out: list[OutputMsg] = []
        deadline = time.monotonic() + timeout_s
        while True:
            record = self.down.pop()
            if record is None:
                if out or time.monotonic() >= deadline:
                    return out
                time.sleep(0.0002)
                continue
            if record.kind == KIND_OUTPUT:
                message = record_to_output(record)
                self.bytes_down += len(message.payload)
                out.append(message)

    def start_provider_loop(self, poll_s: float = 0.0) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        self.up.close()
        self.down.close()

    def snapshot(self) -> dict:
        return {
            "bytes_up": self.bytes_up,
            "bytes_down": self.bytes_down,
            "frames_up": self.frames_up,
            "link": {"kind": "shared memory to the gateway"},
        }


class ShmProviderBridge:
    """The provider's side: take arrivals the gateway wrote, run them, and put
    the outputs where the gateway will pick them up."""

    def __init__(self, engine, up_path: str, down_path: str, create: bool = True, **ring_kwargs):
        self.engine = engine
        self.up = Ring(up_path, create=create, **ring_kwargs)
        self.down = Ring(down_path, create=create, **ring_kwargs)

    def step(self) -> int:
        """Drain arrivals, run one batch, publish its outputs. Returns how
        many outputs it wrote."""
        while True:
            record = self.up.pop()
            if record is None:
                break
            if record.kind == KIND_HIDDEN_STATE:
                self.engine.on_hidden_state(record_to_hidden_state(record))
            elif record.kind == KIND_VERDICT:
                self.engine.on_verdict(record_to_verdict(record))
        written = 0
        for output in self.engine.step():
            while not self.down.push(output_to_record(output)):
                time.sleep(0.0002)
            written += 1
        return written

    def serve_forever(self, idle_s: float = 0.0005) -> None:
        while True:
            if self.step() == 0:
                time.sleep(idle_s)

    def close(self) -> None:
        self.up.close()
        self.down.close()
