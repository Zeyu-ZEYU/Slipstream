"""The link between the two sides.

The deployment the paper measures puts a real residential broadband link here:
20 Mbps up, 100 Mbps down, and the round trip that link happens to have
(Section 3.1). Two other ways to get the same shape are useful for an
artifact:

  * `deploy/netem.sh` shapes a real interface with Linux tc and netem, which
    is how the at-scale run emulates 24 residential links (Section 6.6);
  * `LinkModel` paces a loopback transport in software, which lets the whole
    system run, and the mechanisms show up, on one machine with no network
    configuration at all.

`LinkModel` charges each frame its serialization time at the configured rate
and adds a one-way delay, which is what the first segment, the background
stream and the provider's idle time respond to. Every run records the link it
ran over, so two runs can be compared knob by knob.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class LinkModel:
    uplink_mbps: float = 20.0
    downlink_mbps: float = 100.0
    one_way_delay_ms: float = 20.0
    # Per-frame overhead, standing in for protocol framing and pacing.
    per_frame_overhead_ms: float = 0.2
    # Charge every frame as if its payload were this many bytes, so a run on
    # one machine carries the traffic of the model being studied: 8192 is a
    # hidden state of a model with hidden size 4096 at 16-bit precision
    # (Section 3.1). None charges the payload itself.
    charge_bytes: int | None = None

    def serialize_ms(self, num_bytes: int, mbps: float) -> float:
        if mbps <= 0:
            return 0.0
        bits = num_bytes * 8
        return bits / (mbps * 1e6) * 1e3

    def charged(self, num_bytes: int) -> int:
        """What the link bills for a frame of `num_bytes`."""
        return self.charge_bytes if self.charge_bytes else num_bytes

    def uplink_ms(self, num_bytes: int) -> float:
        return (
            self.serialize_ms(self.charged(num_bytes), self.uplink_mbps)
            + self.per_frame_overhead_ms
        )

    def downlink_ms(self, num_bytes: int) -> float:
        return (
            self.serialize_ms(self.charged(num_bytes), self.downlink_mbps)
            + self.per_frame_overhead_ms
        )

    @property
    def round_trip_ms(self) -> float:
        return 2 * self.one_way_delay_ms

    def describe(self) -> dict:
        return {
            "uplink_mbps": self.uplink_mbps,
            "downlink_mbps": self.downlink_mbps,
            "one_way_delay_ms": self.one_way_delay_ms,
            "round_trip_ms": self.round_trip_ms,
            "charge_bytes": self.charge_bytes,
            "kind": "shaped in software",
        }


class PacedChannel:
    """A one-directional channel that delivers frames in order, charging each
    one its serialization time and then the one-way delay.

    The channel is what makes the two traffic classes matter: bytes handed to
    it can be neither reordered nor recalled, so the scheduler hands it one
    hidden state at a time (Section 4.4, "Priorities").
    """

    def __init__(self, rate_mbps: float, one_way_delay_ms: float, link: LinkModel):
        self.rate_mbps = rate_mbps
        self.one_way_delay_ms = one_way_delay_ms
        self.link = link
        self._lock = threading.Lock()
        # When the channel is free again.
        self._free_at = time.monotonic()
        self.bytes_carried = 0
        self.frames_carried = 0

    def reserve(self, num_bytes: int) -> tuple[float, float]:
        """Claim the channel for one frame. Returns when the frame finishes
        serializing and when it arrives."""
        with self._lock:
            now = time.monotonic()
            start = max(now, self._free_at)
            serialize_s = (
                self.link.serialize_ms(self.link.charged(num_bytes), self.rate_mbps)
                + self.link.per_frame_overhead_ms
            ) / 1e3
            self._free_at = start + serialize_s
            self.bytes_carried += self.link.charged(num_bytes)
            self.frames_carried += 1
            return self._free_at, self._free_at + self.one_way_delay_ms / 1e3

    def busy_for(self) -> float:
        with self._lock:
            return max(0.0, self._free_at - time.monotonic())

    @staticmethod
    def sleep_until(deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
