"""Every knob of the system, with the defaults the paper reports.

Section references point at the paper. Anything a deployment may want to
change is here; nothing else in the package reads an environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

# Element precisions the codec understands (Section 4.5).
Precision = Literal["bf16", "fp8", "fp4"]

BITS = {"bf16": 16, "fp8": 8, "fp4": 4}


@dataclass
class DrafterConfig:
    """The client's drafter (Section 4.2, "Streaming from the drafter")."""

    # Children the drafter keeps per expanded token, from one softmax. The
    # sensitivity analysis of Section 6.5 settles on four.
    children_per_step: int = 4
    # Hard cap on tree depth. Slipstream has no fixed budget; this only stops
    # a pathological run from growing without bound.
    max_depth: int = 24
    # Hard cap on nodes per round, for the same reason.
    max_nodes: int = 64
    # Temperature of the drafter's softmax. Greedy verification does not
    # depend on it; it only shapes the path probabilities.
    temperature: float = 1.0


@dataclass
class AlphaConfig:
    """The table that turns a path probability into an acceptance probability
    (Section 4.2, "Estimating alpha_v")."""

    # Bins of q_v over [0, 1].
    num_bins: int = 20
    # The table starts at alpha_v = q_v and is corrected from verdicts.
    prior_weight: float = 8.0
    # Keep the estimate monotone in q_v.
    monotone: bool = True
    # Where a client persists its table across requests. None keeps it in
    # memory only.
    state_path: str | None = None


@dataclass
class CostConfig:
    """Online cost estimates that feed the marginal rule (Section 4.4,
    "Inputs"). The initial values are the fit of Section 4.1 on
    Nemotron-3-Super, and are replaced by measurements after the first
    rounds."""

    # Critical-path cost of one draft token, milliseconds.
    c_init_ms: float = 10.8
    # Uplink time per hidden state, milliseconds.
    c_uplink_init_ms: float = 4.0
    # Drafting cost of one token, milliseconds.
    c_draft_init_ms: float = 3.0
    # Fixed part of a round, milliseconds.
    t0_init_ms: float = 138.0
    # Acceptance length and round time the rules start from.
    tau_init: float = 4.2
    round_time_init_ms: float = 250.0
    # Exponential-average weight of a new observation.
    ewma_alpha: float = 0.1
    # Fraction of time the uplink carries first-segment traffic, rho. Measured
    # from the client's own uplink scheduler.
    rho_init: float = 0.0


@dataclass
class SchedulerConfig:
    """The client's stream scheduler (Sections 4.2 and 4.4)."""

    # Keep the socket's unsent buffer below one hidden state, so a
    # first-segment hidden state waits behind at most one background one.
    one_frame_at_a_time: bool = True
    # Stream order. "alpha" is the design; "level" is the Level order ablation
    # of Section 6.3.
    order: Literal["alpha", "level"] = "alpha"
    # Admission. "rule" is Equation 2; "all" is the Admit all ablation.
    admission: Literal["rule", "all"] = "rule"
    # Class separation on the wire. False is the No priorities ablation.
    priorities: bool = True
    # Send background tokens at all. False is the First segment only ablation.
    background: bool = True
    # Continue a round on a matched streamed token. False is the No
    # continuation ablation.
    continuation: bool = True
    # A round's first segment is never shorter than this, so a round always
    # carries the trunk's first token.
    min_first_segment: int = 1
    # Safety cap on the first segment.
    max_first_segment: int = 12


@dataclass
class ProviderConfig:
    """The provider's half (Section 4.3)."""

    # Background tokens a batch may carry as passengers, and the size of a
    # background batch.
    batch_passengers: int = 4
    # Background tokens the provider holds per client. The provider recomputes
    # this from its free speculative-state budget and announces it with every
    # output; this is the value it starts from and the one the paper reports.
    window: int = 24
    # Execution policy. "opportunistic" is the design; "eager" and "lazy" are
    # the two ablations of Section 6.3.
    policy: Literal["opportunistic", "eager", "lazy"] = "opportunistic"
    # Speculative state per path, bytes. Used to divide the free budget into
    # windows. The default is the per-path SSM state of a hybrid model with
    # the shape of Nemotron-3-Super.
    state_bytes_per_path: int = 170 * 1024 * 1024
    # Free speculative-state budget, bytes. None disables the division and
    # keeps `window` fixed.
    state_budget_bytes: int | None = None
    # A round's state expires after this many round times.
    round_expiry_round_times: float = 1.0


@dataclass
class CompressionConfig:
    """Stream compression (Section 4.5). Off by default: with compression off
    the output is the target's own greedy decoding, token for token."""

    enabled: bool = False
    # The root is every later token's ancestor, so it never leaves exact.
    root_precision: Precision = "bf16"
    # Precision of the first segment and of background tokens, as the paper's
    # "Slipstream FP4" arm uses them.
    first_segment_precision: Precision = "fp8"
    background_precision: Precision = "fp4"
    # Thresholds on alpha_v, highest first: a token takes the precision of the
    # first threshold it clears. Empty falls back to the two settings above.
    alpha_thresholds: list[tuple[float, Precision]] = field(default_factory=list)
    # The downlink decays more slowly than the uplink, since a corrupted
    # uplink state persists in the provider's cache.
    downlink_one_level_coarser: bool = False
    # Block size of the FP4 codec.
    fp4_block: int = 32
    # Share of positions sent at both the chosen and full precision, so a
    # range can back off when the two disagree.
    online_check_fraction: float = 0.02
    # Disagreement rate above which a range backs off one level.
    backoff_disagreement: float = 0.02


@dataclass
class TransportConfig:
    """Where the engine hands hidden states, and how they cross the network
    (Section 5, "Transport")."""

    # "loopback" keeps both engines in one process, for tests and for a
    # single-node functional run. "shm" hands frames to the Rust gateway over
    # shared memory, which is the deployment the paper measures. "grpc" is a
    # pure-Python fallback that speaks the same protocol.
    kind: Literal["loopback", "shm", "grpc"] = "loopback"
    # Gateway endpoint for "grpc", and the address the provider's gateway
    # listens on.
    address: str = "127.0.0.1:50151"
    # Shared-memory ring for "shm".
    shm_dir: str = ".shm"
    ring_slots: int = 1024
    ring_slot_bytes: int = 64 * 1024
    # Mutual TLS. Paths are read by the gateway, not by Python.
    tls_ca: str | None = None
    tls_cert: str | None = None
    tls_key: str | None = None


@dataclass
class ModelConfig:
    """Which model, and how it is split (Section 2.1)."""

    # A Hugging Face checkpoint directory, or "tiny" for the built-in model
    # used by the tests.
    path: str = "tiny"
    # Layers the client holds at the front. The paper's split is one.
    client_layers: int = 1
    # Torch dtype for the shards.
    dtype: str = "bfloat16"
    # Device of each side.
    client_device: str = "cpu"
    provider_device: str = "cpu"
    # Path to the vendor's multi-token-prediction module. None falls back to
    # drafting with the target's own head, which the tests use.
    mtp_path: str | None = None
    # Obfuscation of the hidden states. Slipstream is orthogonal to it and
    # leaves it unchanged; the deployment plugs its own in here.
    obfuscation: Literal["none", "ldp"] = "none"
    ldp_epsilon: float = 8.0


@dataclass
class DecodeConfig:
    """Decoding (Section 3.1, "Decoding")."""

    greedy: bool = True
    max_output_tokens: int = 256
    # Draft budget for the one-shot methods. Slipstream ignores it.
    draft_budget: int = 8
    # The range the one-shot methods search online (Section 6.1).
    budget_search: tuple[int, int] = (3, 22)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    drafter: DrafterConfig = field(default_factory=DrafterConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        def build(kind, value):
            return kind(**value) if isinstance(value, dict) else value

        return cls(
            model=build(ModelConfig, d.get("model", {})),
            decode=build(DecodeConfig, d.get("decode", {})),
            drafter=build(DrafterConfig, d.get("drafter", {})),
            alpha=build(AlphaConfig, d.get("alpha", {})),
            cost=build(CostConfig, d.get("cost", {})),
            scheduler=build(SchedulerConfig, d.get("scheduler", {})),
            provider=build(ProviderConfig, d.get("provider", {})),
            compression=build(CompressionConfig, d.get("compression", {})),
            transport=build(TransportConfig, d.get("transport", {})),
            seed=d.get("seed", 0),
        )

    @classmethod
    def load(cls, path: str) -> "Config":
        import yaml

        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})
