"""The five comparison methods of Section 6.1.

They share the split, the models and the code, and differ in what a round
ships and waits for:

  Baseline        no speculation, one hidden state up and one down per token
  Cunningham      n-gram guesses from the text so far, shipped in one piece
  One-shot chain  the drafter's chain, shipped in one piece
  One-shot tree   the same best-first tree as Slipstream, shipped in one piece
  Slipstream      the tree streamed in order of acceptance probability
"""

from .base import Method, RequestMetrics
from .baseline import Baseline
from .cunningham import Cunningham, prompt_lookup
from .oneshot import BudgetSearch, OneShotChain, OneShotTree
from .slipstream import Slipstream

METHODS = {
    "baseline": Baseline,
    "cunningham": Cunningham,
    "oneshot_chain": OneShotChain,
    "oneshot_tree": OneShotTree,
    "slipstream": Slipstream,
}

NAMES = {
    "baseline": "Baseline",
    "cunningham": "Cunningham",
    "oneshot_chain": "One-shot chain",
    "oneshot_tree": "One-shot tree",
    "slipstream": "Slipstream",
}

__all__ = [
    "Method",
    "RequestMetrics",
    "Baseline",
    "Cunningham",
    "prompt_lookup",
    "BudgetSearch",
    "OneShotChain",
    "OneShotTree",
    "Slipstream",
    "METHODS",
    "NAMES",
]
