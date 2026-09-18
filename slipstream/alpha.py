"""Acceptance probability by path probability.

No model gives alpha_v, the probability that the target's path runs through a
draft token (Section 4.1, "Goal and rule"). Under greedy decoding a token is
simply accepted or not, so alpha_v is how often tokens like v are: the client
keeps a table by q_v, the fraction of past tokens in each bin of q_v that the
target accepted, counting only tokens whose verdict it has seen and keeping
the estimate monotone in q_v (Section 4.2, "Estimating alpha_v").

The table starts at alpha_v = q_v, which is safe because the drafter is
underconfident on our traces but the order is the same, and it settles within
about a hundred rounds. It is per client, persists across the client's
requests, and can be seeded offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .config import AlphaConfig


@dataclass
class _Bin:
    accepted: float = 0.0
    total: float = 0.0


class AlphaTable:
    def __init__(self, cfg: AlphaConfig | None = None) -> None:
        self.cfg = cfg or AlphaConfig()
        self.bins = [_Bin() for _ in range(self.cfg.num_bins)]
        self.rounds_seen = 0
        if self.cfg.state_path and os.path.exists(self.cfg.state_path):
            self.load(self.cfg.state_path)

    # ------------------------------------------------------------- internals

    def _index(self, q: float) -> int:
        q = min(max(q, 0.0), 1.0)
        i = int(q * self.cfg.num_bins)
        return min(i, self.cfg.num_bins - 1)

    def _bin_centre(self, i: int) -> float:
        return (i + 0.5) / self.cfg.num_bins

    def _raw(self, i: int) -> float:
        """Posterior mean of bin `i` with the alpha_v = q_v prior."""
        b = self.bins[i]
        prior = self._bin_centre(i) * self.cfg.prior_weight
        return (b.accepted + prior) / (b.total + self.cfg.prior_weight)

    def _monotone_table(self) -> list[float]:
        """Running maximum, so a higher path probability never yields a lower
        acceptance probability."""
        out, best = [], 0.0
        for i in range(self.cfg.num_bins):
            best = max(best, self._raw(i))
            out.append(best)
        return out

    # ---------------------------------------------------------------- public

    def estimate(self, q: float) -> float:
        """alpha_v for a draft token whose path probability is `q`."""
        table = self._monotone_table() if self.cfg.monotone else [
            self._raw(i) for i in range(self.cfg.num_bins)
        ]
        return float(min(max(table[self._index(q)], 0.0), 1.0))

    def update(self, observations: list[tuple[float, bool]]) -> None:
        """Fold in one round's verdict: `(q_v, accepted)` for every draft
        token whose verdict the client has seen. Tokens under a rejected
        position have no verdict and are not counted."""
        for q, accepted in observations:
            b = self.bins[self._index(q)]
            b.total += 1.0
            if accepted:
                b.accepted += 1.0
        self.rounds_seen += 1
        if self.cfg.state_path:
            self.save(self.cfg.state_path)

    def settled(self) -> bool:
        """The table settles within about a hundred rounds, one request."""
        return self.rounds_seen >= 100

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        return {
            "num_bins": self.cfg.num_bins,
            "rounds_seen": self.rounds_seen,
            "accepted": [b.accepted for b in self.bins],
            "total": [b.total for b in self.bins],
        }

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.to_dict(), fh)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        with open(path) as fh:
            d = json.load(fh)
        if d.get("num_bins") != self.cfg.num_bins:
            return
        self.rounds_seen = int(d.get("rounds_seen", 0))
        for i, (a, t) in enumerate(zip(d["accepted"], d["total"])):
            self.bins[i] = _Bin(accepted=float(a), total=float(t))

    def table(self) -> list[tuple[float, float]]:
        """`(q bin centre, alpha)` pairs, for logging and for the sanity check
        that the estimate is monotone."""
        vals = self._monotone_table()
        return [(self._bin_centre(i), vals[i]) for i in range(self.cfg.num_bins)]
