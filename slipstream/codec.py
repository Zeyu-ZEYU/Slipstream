"""Stream compression: what a hidden state costs on the wire, and where the
bits go.

Compression is optional and is the one place where the system departs from
exact decoding (Section 4.5): off for exact output, on for a deployment that
will trade a controlled loss in score for goodput. What is new is not the
codec but the placement. The stream already orders hidden states by alpha_v,
and precision follows that order: the root, which every later token reads,
stays exact; first-segment tokens keep a high precision; background tokens,
which are rarely accepted, are compressed harder than one width for all would
dare.

Two levels are implemented, the pair the evaluation measures:

  fp8   E4M3 cast, half the bytes of bf16
  fp4   block-scaled 4-bit floating point (E2M1 codes, one scale per block),
        a quarter of the bytes

Wire layouts, little endian throughout:

  bf16  2 bytes per element
  fp8   1 byte per element
  fp4   num_blocks * 2 bytes of float16 scales, then ceil(n / 2) bytes of
        packed codes, low nibble first

Both levels apply to the obfuscated hidden state, so compression is
post-processing and cannot weaken the obfuscation (Section 4.5).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from .config import BITS, CompressionConfig, Precision

# E2M1 magnitudes, the 4-bit floating-point codes, and their sign bit.
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MAX = _FP4_LEVELS[-1]


def bits_per_element(precision: Precision) -> int:
    return BITS[precision]


# --------------------------------------------------------------------- encode


def encode(tensor: torch.Tensor, precision: Precision, block: int = 32) -> bytes:
    """Encode one hidden state. `tensor` is one dimensional."""
    flat = tensor.detach().to(torch.float32).flatten().cpu()
    if precision == "bf16":
        return flat.to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    if precision == "fp8":
        if hasattr(torch, "float8_e4m3fn"):
            return flat.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
        return _fp8_fallback_encode(flat)
    if precision == "fp4":
        return _fp4_encode(flat, block)
    raise ValueError(f"unknown precision {precision!r}")


def decode(
    payload: bytes, precision: Precision, num_elements: int, block: int = 32
) -> torch.Tensor:
    if precision == "bf16":
        raw = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16)
        return raw[:num_elements].to(torch.float32)
    if precision == "fp8":
        if hasattr(torch, "float8_e4m3fn"):
            raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            return raw[:num_elements].view(torch.float8_e4m3fn).to(torch.float32)
        return _fp8_fallback_decode(payload, num_elements)
    if precision == "fp4":
        return _fp4_decode(payload, num_elements, block)
    raise ValueError(f"unknown precision {precision!r}")


def _fp8_fallback_encode(flat: torch.Tensor) -> bytes:
    """E4M3 by hand, for a torch build without the dtype."""
    out = bytearray()
    for value in flat.tolist():
        out.append(_e4m3_pack(value))
    return bytes(out)


def _fp8_fallback_decode(payload: bytes, num_elements: int) -> torch.Tensor:
    values = [_e4m3_unpack(b) for b in payload[:num_elements]]
    return torch.tensor(values, dtype=torch.float32)


def _e4m3_pack(value: float) -> int:
    if value == 0.0 or math.isnan(value):
        return 0
    sign = 0x80 if value < 0 else 0
    magnitude = min(abs(value), 448.0)
    exponent = max(-6, min(8, math.floor(math.log2(magnitude))))
    mantissa = magnitude / (2.0**exponent) - 1.0
    code = int(round(mantissa * 8))
    if code > 7:
        code, exponent = 0, exponent + 1
    biased = max(0, min(15, exponent + 7))
    return sign | (biased << 3) | code


def _e4m3_unpack(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    biased = (byte >> 3) & 0x0F
    code = byte & 0x07
    if biased == 0:
        return sign * code / 8.0 * (2.0**-6)
    return sign * (1.0 + code / 8.0) * (2.0 ** (biased - 7))


def _fp4_encode(flat: torch.Tensor, block: int) -> bytes:
    n = flat.numel()
    num_blocks = (n + block - 1) // block
    padded = torch.zeros(num_blocks * block, dtype=torch.float32)
    padded[:n] = flat
    view = padded.view(num_blocks, block)
    amax = view.abs().amax(dim=1)
    scales = torch.where(
        amax > 0, amax / _FP4_MAX, torch.ones_like(amax)
    ).to(torch.float16)
    scaled = view / scales.to(torch.float32).clamp(min=1e-12).unsqueeze(1)
    levels = torch.tensor(_FP4_LEVELS, dtype=torch.float32)
    magnitude = scaled.abs().clamp(max=_FP4_MAX)
    # Nearest E2M1 magnitude.
    idx = (magnitude.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    codes = idx.to(torch.uint8) | torch.where(
        scaled < 0, torch.tensor(8, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8)
    )
    codes = codes.flatten()[:n]
    if codes.numel() % 2:
        codes = torch.cat([codes, torch.zeros(1, dtype=torch.uint8)])
    low, high = codes[0::2], codes[1::2]
    packed = (low & 0x0F) | ((high & 0x0F) << 4)
    return scales.view(torch.uint8).numpy().tobytes() + packed.numpy().tobytes()


def _fp4_decode(payload: bytes, num_elements: int, block: int) -> torch.Tensor:
    num_blocks = (num_elements + block - 1) // block
    scale_bytes = num_blocks * 2
    scales = torch.frombuffer(bytearray(payload[:scale_bytes]), dtype=torch.float16)
    packed = torch.frombuffer(bytearray(payload[scale_bytes:]), dtype=torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack([low, high], dim=1).flatten()[: num_blocks * block]
    levels = torch.tensor(_FP4_LEVELS, dtype=torch.float32)
    magnitude = levels[(codes & 0x07).long()]
    sign = torch.where(codes & 0x08 > 0, -1.0, 1.0)
    values = (magnitude * sign).view(num_blocks, block)
    values = values * scales.to(torch.float32).unsqueeze(1)
    return values.flatten()[:num_elements]


def payload_bytes(num_elements: int, precision: Precision, block: int = 32) -> int:
    """Bytes a hidden state takes on the wire, which is what admission
    prices."""
    if precision == "bf16":
        return 2 * num_elements
    if precision == "fp8":
        return num_elements
    num_blocks = (num_elements + block - 1) // block
    return num_blocks * 2 + (num_elements + 1) // 2


# --------------------------------------------------------------------- policy


@dataclass
class _RangeStats:
    checks: int = 0
    disagreements: int = 0
    backed_off: bool = False


class PrecisionPolicy:
    """Precision follows alpha_v (Section 4.5).

    The thresholds are calibrated, not modelled: for each range of alpha_v the
    client uses the coarsest level whose outputs still agree with full
    precision, chosen offline per model. Online, a sampled few percent of
    positions go up at both the chosen and full precision, as two positions
    with the same parent, and a range backs off one level when they disagree
    more often than the calibration allows.
    """

    _ORDER: tuple[Precision, ...] = ("fp4", "fp8", "bf16")

    def __init__(
        self, cfg: CompressionConfig | None = None, rng: random.Random | None = None
    ) -> None:
        self.cfg = cfg or CompressionConfig()
        self.rng = rng or random.Random(0)
        # Thresholds on alpha_v, highest first. Empty means the two-level
        # default: the first segment at one precision, background tokens at
        # the coarser one, which is the pair the evaluation reports.
        self.thresholds: list[tuple[float, Precision]] = sorted(
            self.cfg.alpha_thresholds, key=lambda t: -t[0]
        )
        self.stats: dict[int, _RangeStats] = {}

    def _range_index(self, alpha: float) -> int:
        """Which calibrated range of alpha_v a token falls in. Without
        explicit thresholds there are two ranges, split at the first-segment
        boundary the scheduler already applied."""
        if not self.thresholds:
            return 0 if alpha >= 0.2 else 1
        for i, (low, _) in enumerate(self.thresholds):
            if alpha >= low:
                return i
        return len(self.thresholds)

    def uplink_precision(
        self, alpha: float, *, is_root: bool, in_first_segment: bool
    ) -> Precision:
        """Precision of a hidden state on its way up."""
        if not self.cfg.enabled:
            return "bf16"
        if is_root:
            return self.cfg.root_precision
        chosen = (
            self.cfg.first_segment_precision
            if in_first_segment
            else self.cfg.background_precision
        )
        for low, precision in self.thresholds:
            if alpha >= low:
                chosen = precision
                break
        index = self._range_index(alpha)
        stats = self.stats.get(index)
        if stats and stats.backed_off:
            chosen = self.finer(chosen)
        return chosen

    def downlink_precision(self, uplink: Precision) -> Precision:
        """Precision the client asks for on the way back. A corrupted uplink
        state persists in the provider's cache, so the uplink decays more
        slowly than the downlink."""
        if not self.cfg.enabled:
            return "bf16"
        if self.cfg.downlink_one_level_coarser:
            return self.coarser(uplink)
        return uplink

    def uplink_cost_scale(self, precision: Precision) -> float:
        """b_v / 16, the factor admission prices a background token by."""
        return bits_per_element(precision) / 16.0

    @classmethod
    def coarser(cls, precision: Precision) -> Precision:
        i = cls._ORDER.index(precision)
        return cls._ORDER[max(0, i - 1)]

    @classmethod
    def finer(cls, precision: Precision) -> Precision:
        i = cls._ORDER.index(precision)
        return cls._ORDER[min(len(cls._ORDER) - 1, i + 1)]

    # ------------------------------------------------------- online checking

    def should_check(self) -> bool:
        """True for the sampled few percent of positions that go up at both
        the chosen and full precision."""
        return self.cfg.enabled and self.rng.random() < self.cfg.online_check_fraction

    def record_check(self, alpha: float, disagreed: bool) -> None:
        index = self._range_index(alpha)
        stats = self.stats.setdefault(index, _RangeStats())
        stats.checks += 1
        stats.disagreements += int(disagreed)
        if stats.checks >= 50:
            rate = stats.disagreements / stats.checks
            stats.backed_off = rate > self.cfg.backoff_disagreement
            stats.checks, stats.disagreements = 0, 0

    def summary(self) -> dict:
        return {
            "enabled": self.cfg.enabled,
            "root": self.cfg.root_precision,
            "first_segment": self.cfg.first_segment_precision,
            "background": self.cfg.background_precision,
            "backed_off_ranges": [i for i, s in self.stats.items() if s.backed_off],
        }


def quantization_error(tensor: torch.Tensor, precision: Precision, block: int = 32) -> float:
    """Relative element error of one level, the number the calibration reports."""
    payload = encode(tensor, precision, block)
    restored = decode(payload, precision, tensor.numel(), block)
    reference = tensor.detach().to(torch.float32).flatten().cpu()
    denominator = reference.abs().mean().clamp(min=1e-12)
    return float((restored - reference).abs().mean() / denominator)
