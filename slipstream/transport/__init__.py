"""How a hidden state gets from one side to the other.

  messages.py  the three messages of the protocol (Appendix C)
  link.py      the link itself, shaped in software for a single-node run
  loopback.py  both engines in one process, for tests and functional runs
  shm.py       the shared-memory ring the Rust gateway reads
  grpc_channel.py  a pure-Python client that speaks the same protocol
"""

from .link import LinkModel, PacedChannel
from .loopback import DirectTransport, LoopbackTransport, ProviderHub
from .messages import HiddenStateMsg, OutputMsg, VerdictMsg

__all__ = [
    "LinkModel",
    "PacedChannel",
    "DirectTransport",
    "LoopbackTransport",
    "ProviderHub",
    "HiddenStateMsg",
    "OutputMsg",
    "VerdictMsg",
]
