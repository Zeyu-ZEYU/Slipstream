"""Slipstream: streaming speculative decoding for split LLM inference over
wide-area networks.

The package mirrors the paper's structure:

  config.py        every knob, with the paper's defaults
  tree.py          the round's draft tree, ranks, trunk, ancestor mask
  alpha.py         acceptance probability by path probability (Section 4.2)
  costs.py         the cost model and the marginal rule (Sections 4.1, 4.4)
  codec.py         stream compression (Section 4.5)
  scheduler.py     the client's stream scheduler (Sections 4.2, 4.4)
  client/          encoder, decoder, drafter, the round loop
  provider/        the middle, two queues, passengers, the window (Section 4.3)
  transport/       shared memory to the gateway, and the protocol
  models/          split shards, the drafter's module, the provider's cache
  methods/         the five comparison methods of Section 6.1
"""

from .config import Config
from .costs import CostModel
from .tree import DraftTree

__version__ = "0.1.0"

__all__ = ["Config", "CostModel", "DraftTree", "__version__"]
