"""The cost model and the one marginal rule.

Section 4.1 fits a one-shot round to

    T = T0 + m (c_d + c)

with T0 the fixed part of a round including the round trip, c_d the drafting
of one token, and c what its hidden state then costs on the critical path: the
uplink, the middle, the downlink, and the verify. Two of those numbers enter
the rules, c and c_u, and the client measures both online (Section 4.4,
"Inputs"):

  c     slope of the client's round times against the first-segment length s
  c_u   spacing of consecutive arrivals in the provider's timestamps, which
        needs no clock synchronization
  rho   fraction of time the client's uplink carries first-segment traffic,
        read off its own uplink scheduler
  tau,T running averages over the client's requests

The rule itself is Equation 1: adding a draft token v raises goodput exactly
when alpha_v / Delta > tau / T. With Delta = c it decides the first segment;
with Delta = c_u rho it decides admission (Equation 2); with the expected wait
for a continuation's outputs it decides whether a round continues.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import CostConfig


def _ewma(old: float, new: float, weight: float) -> float:
    return (1.0 - weight) * old + weight * new


@dataclass
class StageBreakdown:
    """One round's five stages, in milliseconds (Section 3.3)."""

    draft_encode: float = 0.0
    uplink: float = 0.0
    middle: float = 0.0
    downlink: float = 0.0
    decode_verify: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.draft_encode
            + self.uplink
            + self.middle
            + self.downlink
            + self.decode_verify
        )

    @property
    def network(self) -> float:
        return self.uplink + self.downlink

    def as_dict(self) -> dict:
        return {
            "draft_encode_ms": self.draft_encode,
            "uplink_ms": self.uplink,
            "middle_ms": self.middle,
            "downlink_ms": self.downlink,
            "decode_verify_ms": self.decode_verify,
            "round_ms": self.total,
            "network_share": self.network / self.total if self.total else 0.0,
        }


class CostModel:
    """Online estimates plus the two thresholds the scheduler asks for."""

    def __init__(self, cfg: CostConfig | None = None) -> None:
        self.cfg = cfg or CostConfig()
        self.c = self.cfg.c_init_ms
        self.c_uplink = self.cfg.c_uplink_init_ms
        self.c_draft = self.cfg.c_draft_init_ms
        self.t0 = self.cfg.t0_init_ms
        self.tau = self.cfg.tau_init
        self.round_time = self.cfg.round_time_init_ms
        self.rho = self.cfg.rho_init
        # What a background token's output has cost to wait for, observed.
        # Continuation prices its wait with this: one pass of the middle and
        # the downlink when the provider is idle, and a round trip, the
        # foreground queue ahead, and the pass when it is busy
        # (Section 4.2, "Branch continuation").
        self.background_delay_ms = self.cfg.c_init_ms + 2 * 20.0
        # Samples for the slope of round time against first-segment length.
        self._round_samples: list[tuple[int, float]] = []
        self._rounds = 0

    # ------------------------------------------------------------- the rule

    @property
    def rate(self) -> float:
        """tau / T, the rate the round already achieves, in tokens per ms."""
        return self.tau / self.round_time if self.round_time > 0 else 0.0

    def pays(self, alpha: float, delay_ms: float) -> bool:
        """Equation 1: does a token worth `alpha` pay for `delay_ms` of extra
        waiting?"""
        if delay_ms <= 0:
            return alpha > 0.0
        return (alpha / delay_ms) > self.rate

    @property
    def first_segment_threshold(self) -> float:
        """Equation 1 with Delta = c: a token joins the first segment while
        alpha_v > c tau / T, about 0.2 on Nemotron-3-Super."""
        return self.c * self.rate

    def background_threshold(self, uplink_cost_scale: float = 1.0) -> float:
        """Equation 2: a background token enters the stream while
        alpha_v > c_u rho tau / T.

        `uplink_cost_scale` is b_v / 16 when compression is on, so a token at
        four bits passes admission at close to a quarter of the cost
        (Section 4.5).
        """
        return self.c_uplink * uplink_cost_scale * self.rho * self.rate

    def continuation_pays(self, gain_tokens: float, expected_wait_ms: float) -> bool:
        """Branch continuation, again by Equation 1: the gain is one token
        from the matched token's own output plus alpha_u / alpha_a for each
        streamed token u under the matched token a."""
        return self.pays(gain_tokens, expected_wait_ms)

    # ------------------------------------------------------- online estimates

    def observe_round(
        self,
        first_segment_len: int,
        round_ms: float,
        accepted_tokens: int,
    ) -> None:
        self._rounds += 1
        self.round_time = _ewma(self.round_time, round_ms, self.cfg.ewma_alpha)
        self.tau = _ewma(self.tau, float(accepted_tokens), self.cfg.ewma_alpha)
        self._round_samples.append((first_segment_len, round_ms))
        if len(self._round_samples) > 512:
            self._round_samples = self._round_samples[-512:]
        slope = _slope(self._round_samples)
        if slope is not None and slope > 0:
            self.c = _ewma(self.c, slope, self.cfg.ewma_alpha)

    def observe_arrivals(self, arrival_times_ns: list[int]) -> None:
        """c_u from the spacing of consecutive arrivals in the provider's
        timestamps. Differences cancel any clock offset."""
        if len(arrival_times_ns) < 2:
            return
        gaps = [
            (b - a) / 1e6
            for a, b in zip(arrival_times_ns, arrival_times_ns[1:])
            if b > a
        ]
        if not gaps:
            return
        gaps.sort()
        median = gaps[len(gaps) // 2]
        if median > 0:
            self.c_uplink = _ewma(self.c_uplink, median, self.cfg.ewma_alpha)

    def observe_uplink_share(self, first_segment_bytes: int, total_bytes: int) -> None:
        """rho, read off the client's own uplink scheduler."""
        if total_bytes <= 0:
            return
        self.rho = _ewma(
            self.rho, first_segment_bytes / total_bytes, self.cfg.ewma_alpha
        )

    def observe_draft_step(self, ms: float) -> None:
        self.c_draft = _ewma(self.c_draft, ms, self.cfg.ewma_alpha)

    def observe_background_delay(self, ms: float) -> None:
        """How long a background token's output took to come back, which is
        what a continuation has to wait for when the provider has not run the
        matched token yet."""
        if ms > 0:
            self.background_delay_ms = _ewma(
                self.background_delay_ms, ms, self.cfg.ewma_alpha
            )

    # --------------------------------------------------------------- fitting

    def fit_one_shot(self, samples: list[tuple[int, float]]) -> dict:
        """Least squares of T = T0 + m (c_d + c) over `(budget, round_ms)`
        samples, the fit Section 4.1 reports. Returns the fit and the largest
        residual, which the paper reports as within 6 ms."""
        slope = _slope(samples)
        if slope is None:
            return {}
        mean_m = sum(m for m, _ in samples) / len(samples)
        mean_t = sum(t for _, t in samples) / len(samples)
        t0 = mean_t - slope * mean_m
        residuals = [abs(t - (t0 + slope * m)) for m, t in samples]
        self.t0 = t0
        return {
            "t0_ms": t0,
            "per_token_ms": slope,
            "max_residual_ms": max(residuals),
        }

    def snapshot(self) -> dict:
        return {
            "c_ms": self.c,
            "c_uplink_ms": self.c_uplink,
            "c_draft_ms": self.c_draft,
            "t0_ms": self.t0,
            "tau": self.tau,
            "round_ms": self.round_time,
            "rho": self.rho,
            "rate_tokens_per_ms": self.rate,
            "first_segment_threshold": self.first_segment_threshold,
            "background_threshold": self.background_threshold(),
            "rounds": self._rounds,
        }


def _slope(samples: list[tuple[int, float]]) -> float | None:
    """Least-squares slope of y against x, or None when x never varies."""
    if len(samples) < 3:
        return None
    n = float(len(samples))
    mean_x = sum(x for x, _ in samples) / n
    mean_y = sum(y for _, y in samples) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in samples)
    den = sum((x - mean_x) ** 2 for x, _ in samples)
    if den <= 1e-9:
        return None
    return num / den


class Stopwatch:
    """Millisecond timer used by the stage breakdown and the overhead study."""

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self._start = time.perf_counter()

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        elapsed = (now - self._start) * 1e3
        self.marks[name] = self.marks.get(name, 0.0) + elapsed
        self._start = now
        return elapsed

    def reset(self) -> None:
        self._start = time.perf_counter()


@dataclass
class OverheadCounters:
    """The six decisions Slipstream adds to a round's critical path
    (Section 6.7). Every one is timed separately so the artifact can report
    the same breakdown."""

    end_first_segment_us: list[float] = field(default_factory=list)
    socket_handoff_us: list[float] = field(default_factory=list)
    between_round_estimates_us: list[float] = field(default_factory=list)
    form_foreground_batch_us: list[float] = field(default_factory=list)
    passenger_selection_us: list[float] = field(default_factory=list)
    apply_verdict_us: list[float] = field(default_factory=list)

    def add(self, name: str, microseconds: float) -> None:
        getattr(self, name).append(microseconds)

    def summary(self) -> dict:
        def stats(values: list[float]) -> dict:
            if not values:
                return {"n": 0}
            ordered = sorted(values)
            return {
                "n": len(ordered),
                "mean_us": sum(ordered) / len(ordered),
                "p50_us": ordered[len(ordered) // 2],
                "p99_us": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
            }

        return {
            name: stats(getattr(self, name))
            for name in (
                "end_first_segment_us",
                "socket_handoff_us",
                "between_round_estimates_us",
                "form_foreground_batch_us",
                "passenger_selection_us",
                "apply_verdict_us",
            )
        }
