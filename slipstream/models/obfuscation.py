"""Obfuscation of the hidden states.

Privacy in this deployment comes from obfuscating the hidden states, for
example with local differential privacy, not from encryption, so the provider
runs an ordinary forward pass (Section 2.1). Slipstream is orthogonal to the
mechanism: it changes neither the model nor the drafter, and the obfuscation of
the hidden states is unchanged.

The hook is here so a deployment can plug its own mechanism in at the one place
where a hidden state leaves the client. Two things matter for the rest of the
system:

  * The obfuscation runs before the codec, so compression is post-processing of
    an already obfuscated hidden state and cannot weaken it (Section 4.5).
  * Whatever it does, it does the same thing for every hidden state, so
    streaming a tree and shipping it in one piece are equally private in the
    values they expose.

`none` is the default, which is what an exact-decoding deployment without an
obfuscation layer uses. `ldp` is a reference mechanism, not a contribution of
the paper: it clips a hidden state to a fixed norm and adds Gaussian noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Obfuscation:
    kind: str = "none"
    epsilon: float = 8.0
    delta: float = 1e-5
    clip_norm: float = 1.0
    generator: torch.Generator | None = None

    def sigma(self) -> float:
        """Gaussian scale for (epsilon, delta) at the configured clip norm."""
        import math

        if self.epsilon <= 0:
            return 0.0
        return (
            self.clip_norm
            * math.sqrt(2.0 * math.log(1.25 / self.delta))
            / self.epsilon
        )

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return hidden
        if self.kind != "ldp":
            raise ValueError(f"unknown obfuscation {self.kind!r}")
        flat = hidden.to(torch.float32)
        norm = flat.norm()
        if norm > self.clip_norm:
            flat = flat * (self.clip_norm / norm)
        noise = torch.randn(
            flat.shape, generator=self.generator, device=flat.device
        ) * self.sigma()
        return (flat + noise).to(hidden.dtype)

    def describe(self) -> dict:
        out = {"kind": self.kind}
        if self.kind == "ldp":
            out.update(
                {"epsilon": self.epsilon, "delta": self.delta, "sigma": self.sigma()}
            )
        return out
