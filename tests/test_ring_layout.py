"""The shared-memory ring, and that Python and Rust agree on its layout.

`slipstream/transport/shm.py` and `gateway/src/ring.rs` both write the ring the
engine shares with its gateway (Section 5). If the two ever disagree about a
field's offset, the gateway would ship garbage, so the offsets are checked
here against the numbers the Rust file documents.
"""

from __future__ import annotations

import struct

from slipstream.transport import shm
from slipstream.transport.messages import HiddenStateMsg, OutputMsg, VerdictMsg


def test_header_and_record_sizes_match_the_rust_layout():
    assert shm.HEADER_BYTES == 64
    assert shm.RECORD_HEADER_BYTES == 64
    assert shm._HEADER.size == shm.HEADER_BYTES
    assert shm._RECORD.size == shm.RECORD_HEADER_BYTES
    assert shm.MAGIC == 0x534C5031 and shm.VERSION == 1


def test_record_field_offsets_match_the_rust_layout():
    # The offsets ring.rs documents, checked by packing a record with known
    # values and reading each field back at its offset.
    record = shm.Record(
        kind=shm.KIND_HIDDEN_STATE,
        flags=shm.FLAG_FIRST_SEGMENT_END,
        request_id=0x1122334455667788,
        round_id=0x99AABBCCDDEEFF00,
        position=0x11223344,
        parent=0x55667788,
        payload_precision=2,
        output_precision=3,
        num_elements=4096,
        commits_round_id=0xDEADBEEF,
        payload=b"\x01\x02\x03\x04",
    )
    buffer = bytearray(shm.RECORD_HEADER_BYTES)
    shm._RECORD.pack_into(
        buffer, 0, 1, record.kind, record.flags, record.request_id, record.round_id,
        record.position, record.parent, record.payload_precision,
        record.output_precision, record.num_elements, len(record.payload),
        record.commits_round_id,
    )
    assert struct.unpack_from("<Q", buffer, 0)[0] == 1            # sequence
    assert struct.unpack_from("<I", buffer, 8)[0] == record.kind
    assert struct.unpack_from("<I", buffer, 12)[0] == record.flags
    assert struct.unpack_from("<Q", buffer, 16)[0] == record.request_id
    assert struct.unpack_from("<Q", buffer, 24)[0] == record.round_id
    assert struct.unpack_from("<I", buffer, 32)[0] == record.position
    assert struct.unpack_from("<I", buffer, 36)[0] == record.parent
    assert struct.unpack_from("<I", buffer, 40)[0] == record.payload_precision
    assert struct.unpack_from("<I", buffer, 44)[0] == record.output_precision
    assert struct.unpack_from("<I", buffer, 48)[0] == record.num_elements
    assert struct.unpack_from("<I", buffer, 52)[0] == len(record.payload)
    assert struct.unpack_from("<Q", buffer, 56)[0] == record.commits_round_id


def test_ring_round_trips_and_reports_back_pressure(tmp_path):
    path = str(tmp_path / "up.ring")
    writer = shm.Ring(path, slots=4, slot_bytes=256, create=True)
    reader = shm.Ring(path, create=False)
    message = HiddenStateMsg(
        request_id=3, round_id=2, position=5, parent=4, payload=b"\x07" * 32,
        num_elements=16, payload_precision="fp8", output_precision="fp4",
        first_segment_end=True, traffic_class="first_segment",
    )
    assert writer.push(shm.hidden_state_to_record(message))
    got = shm.record_to_hidden_state(reader.pop())
    assert (got.position, got.parent, got.num_elements) == (5, 4, 16)
    assert got.payload == message.payload
    assert got.payload_precision == "fp8" and got.output_precision == "fp4"
    assert got.first_segment_end and got.traffic_class == "first_segment"

    for index in range(4):
        assert writer.push(shm.hidden_state_to_record(message)) or index == 4
    assert not writer.push(shm.hidden_state_to_record(message))
    writer.close()
    reader.close()


def test_a_verdict_carries_only_positions(tmp_path):
    path = str(tmp_path / "up.ring")
    ring = shm.Ring(path, slots=2, slot_bytes=128, create=True)
    verdict = VerdictMsg(request_id=1, round_id=7, accepted=[1, 4, 9], continue_round=True)
    ring.push(shm.verdict_to_record(verdict))
    record = ring.pop()
    back = shm.record_to_verdict(record)
    assert back.accepted == [1, 4, 9]
    assert back.continue_round and not back.end_of_request
    assert record.num_elements == 3
    ring.close()


def test_an_output_carries_the_window(tmp_path):
    path = str(tmp_path / "down.ring")
    ring = shm.Ring(path, slots=2, slot_bytes=256, create=True)
    output = OutputMsg(
        request_id=2, round_id=3, position=6, payload=b"\x01" * 16, num_elements=8,
        payload_precision="bf16", window=24, traffic_class="background",
    )
    ring.push(shm.output_to_record(output))
    back = shm.record_to_output(ring.pop())
    assert back.window == 24 and back.traffic_class == "background"
    assert back.payload == output.payload
    ring.close()
