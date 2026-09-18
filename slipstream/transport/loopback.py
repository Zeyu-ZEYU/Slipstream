"""A transport that keeps both engines in one process.

This is what the tests and the single-node functional run use. It speaks the
same three messages as the network transports, paces them over a `LinkModel`,
and drives the provider's engine on its own thread, so the behaviour the design
depends on is real and not simulated away:

  * the round waits only for the first segment, because only the marked frame
    puts work in the foreground queue;
  * background tokens travel behind it and run in the provider's idle time;
  * a first-segment hidden state waits behind at most one background hidden
    state, because the scheduler hands the channel one frame at a time.

What is emulated is only the link itself (`transport/link.py`).
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING

from .link import LinkModel, PacedChannel
from .messages import HiddenStateMsg, OutputMsg, VerdictMsg

if TYPE_CHECKING:  # pragma: no cover
    from ..provider.engine import ProviderEngine


class LoopbackTransport:
    """One instance per client. Several clients may share one provider engine,
    which is how the multi-client runs work."""

    def __init__(
        self,
        provider: "ProviderEngine",
        link: LinkModel | None = None,
        client_id: int = 0,
        uplink: PacedChannel | None = None,
        downlink: PacedChannel | None = None,
    ) -> None:
        self.provider = provider
        self.link = link or LinkModel()
        self.client_id = client_id
        # Each client has its own link; the provider's side is shared only in
        # so far as its GPU is.
        self.uplink = uplink or PacedChannel(
            self.link.uplink_mbps, self.link.one_way_delay_ms, self.link
        )
        self.downlink = downlink or PacedChannel(
            self.link.downlink_mbps, self.link.one_way_delay_ms, self.link
        )
        self.outputs: "queue.Queue[OutputMsg]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.bytes_up = 0
        self.bytes_down = 0
        self.frames_up = 0

    # ---------------------------------------------------------------- client

    def send_hidden_state(self, msg: HiddenStateMsg) -> None:
        """Hand one hidden state to the link. Returns once the channel has
        taken it, which is what keeps the unsent buffer below one frame."""
        msg.client_id = self.client_id
        num_bytes = msg.wire_bytes()
        done_serializing, arrival = self.uplink.reserve(num_bytes)
        self.bytes_up += num_bytes
        self.frames_up += 1
        PacedChannel.sleep_until(done_serializing)
        self._deliver_later(arrival, lambda: self.provider.on_hidden_state(msg))

    def send_verdict(self, msg: VerdictMsg) -> None:
        msg.client_id = self.client_id
        num_bytes = msg.wire_bytes()
        done_serializing, arrival = self.uplink.reserve(num_bytes)
        self.bytes_up += num_bytes
        PacedChannel.sleep_until(done_serializing)
        self._deliver_later(arrival, lambda: self.provider.on_verdict(msg))

    def poll_outputs(self, timeout_s: float = 0.0) -> list[OutputMsg]:
        out: list[OutputMsg] = []
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                out.append(self.outputs.get_nowait())
            except queue.Empty:
                if out or time.monotonic() >= deadline:
                    return out
                time.sleep(0.0005)

    # -------------------------------------------------------------- provider

    def start_provider_loop(self, poll_s: float = 0.0005) -> None:
        """Run the provider's engine on its own thread, as a server would."""

        def loop() -> None:
            while not self._stop.is_set():
                produced = self.provider.step()
                if not produced:
                    time.sleep(poll_s)
                    continue
                for output in produced:
                    self._send_output(output)

        thread = threading.Thread(target=loop, name="provider", daemon=True)
        thread.start()
        self._threads.append(thread)

    def deliver_output(self, output: OutputMsg) -> None:
        """Put one output on this client's downlink."""
        self._send_output(output)

    def _send_output(self, output: OutputMsg) -> None:
        num_bytes = output.wire_bytes()
        done_serializing, arrival = self.downlink.reserve(num_bytes)
        self.bytes_down += num_bytes
        self._deliver_later(arrival, lambda: self.outputs.put(output))

    def _deliver_later(self, at: float, action) -> None:
        def wait_and_run() -> None:
            PacedChannel.sleep_until(at)
            action()

        thread = threading.Thread(target=wait_and_run, daemon=True)
        thread.start()

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)

    def snapshot(self) -> dict:
        return {
            "bytes_up": self.bytes_up,
            "bytes_down": self.bytes_down,
            "frames_up": self.frames_up,
            "uplink_busy_s": self.uplink.busy_for(),
            "link": self.link.describe(),
        }


class ProviderHub:
    """One provider engine serving several clients in one process.

    A deployment gives each client its own connection, so the gateway knows
    whose output it is carrying. In a single process the engine is shared, so
    the hub keeps the map from client to transport and dispatches each output
    to the right link.
    """

    def __init__(self, provider: "ProviderEngine") -> None:
        self.provider = provider
        self.transports: dict[int, "LoopbackTransport"] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, transport: "LoopbackTransport") -> None:
        self.transports[transport.client_id] = transport

    def start(self, poll_s: float = 0.0005) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                produced = self.provider.step()
                if not produced:
                    time.sleep(poll_s)
                    continue
                for output in produced:
                    transport = self.transports.get(output.client_id)
                    if transport is not None:
                        transport.deliver_output(output)

        self._thread = threading.Thread(target=loop, name="provider", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class DirectTransport:
    """The simplest transport of all: no link, no threads, and the provider
    runs when the client asks for outputs.

    Deterministic, so the correctness tests can compare a streamed round with
    a one-shot round token for token.
    """

    def __init__(self, provider: "ProviderEngine", client_id: int = 0) -> None:
        self.provider = provider
        self.client_id = client_id
        self._pending: list[OutputMsg] = []
        self.bytes_up = 0
        self.bytes_down = 0
        self.frames_up = 0

    def send_hidden_state(self, msg: HiddenStateMsg) -> None:
        msg.client_id = self.client_id
        self.bytes_up += msg.wire_bytes()
        self.frames_up += 1
        self.provider.on_hidden_state(msg)

    def send_verdict(self, msg: VerdictMsg) -> None:
        msg.client_id = self.client_id
        self.provider.on_verdict(msg)

    def poll_outputs(self, timeout_s: float = 0.0) -> list[OutputMsg]:
        if not self._pending:
            self._pending.extend(self.provider.run_until_idle())
        out, self._pending = self._pending, []
        self.bytes_down += sum(o.wire_bytes() for o in out)
        return out

    def start_provider_loop(self, poll_s: float = 0.0) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        return None

    def snapshot(self) -> dict:
        return {
            "bytes_up": self.bytes_up,
            "bytes_down": self.bytes_down,
            "frames_up": self.frames_up,
            "link": {"kind": "none, one process"},
        }
