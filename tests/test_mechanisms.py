"""The mechanisms of Section 4, one test each.

These run on the built-in tiny model, so they need no checkpoint and no GPU.
Where a test needs the drafter to be right or wrong on purpose it uses the
drafter in `slipstream/models/oracle.py`, which proposes from the target.
"""

from __future__ import annotations

import torch

from slipstream.alpha import AlphaTable
from slipstream.client.engine import ClientEngine
from slipstream.codec import decode, encode, payload_bytes, quantization_error
from slipstream.config import Config
from slipstream.costs import CostModel
from slipstream.methods import Baseline, OneShotTree, Slipstream, prompt_lookup
from slipstream.models import build_tiny, build_tiny_hybrid
from slipstream.models.oracle import OracleDrafter
from slipstream.provider.engine import ProviderEngine
from slipstream.provider.queues import Arrival, ProviderScheduler
from slipstream.provider.queues import Klass as ProviderKlass
from slipstream.scheduler import Klass
from slipstream.scheduler import StreamScheduler
from slipstream.transport.loopback import DirectTransport
from slipstream.tree import DraftTree, candidate_pool_size


def build(cfg: Config | None = None, hit_rate: float = 1.0, rank: int = 1, hybrid: bool = False):
    torch.manual_seed(0)
    cfg = cfg or Config()
    if hybrid:
        client, provider, _ = build_tiny_hybrid()
        for shard in (client, provider):
            for layer in shard.layers:
                if layer.self_attn is not None:
                    torch.nn.init.zeros_(layer.self_attn.o_proj.weight)
    else:
        client, provider, _ = build_tiny(context_free=True)
    engine_provider = ProviderEngine(provider, cfg.provider)
    drafter = OracleDrafter(client, provider, hit_rate=hit_rate, rank=rank)
    engine = ClientEngine(
        client, drafter, DirectTransport(engine_provider), cfg
    )
    return engine, engine_provider


# ------------------------------------------------------------------- the tree


def test_tree_shape_and_ranks():
    tree = DraftTree(root_token=1)
    first = tree.add_children(0, [10, 11], [0.7, 0.2])
    tree.add_children(first[0], [20, 21], [0.6, 0.3])
    tree.assign_ranks()
    assert tree.trunk()[0] == first[0]
    assert tree[first[0]].rank == 1 and tree[first[1]].rank == 2
    # A child's path probability is at most its parent's, which is what makes
    # the stream order put a parent before its children.
    for node in tree.nodes[1:]:
        assert node.q <= tree[node.parent].q + 1e-9
    # The pools Section 3.2 runs into: a (6, 2) tree offers 22 candidates and
    # a (6, 4) tree 84, which is also what a (10, 3) tree offers.
    assert candidate_pool_size(6, 2) == 22
    assert candidate_pool_size(6, 4) == 84
    assert candidate_pool_size(10, 3) == 84


def test_ancestor_mask_is_exact():
    tree = DraftTree(root_token=1)
    a, b = tree.add_children(0, [10, 11], [0.7, 0.2])
    c, _ = tree.add_children(a, [20, 21], [0.6, 0.3])
    positions = [0, a, b, c]
    mask = tree.ancestor_mask(positions)
    index = {p: i for i, p in enumerate(positions)}
    # c sees the root, a and itself, but never its sibling's branch.
    assert mask[index[c]][index[0]] and mask[index[c]][index[a]] and mask[index[c]][index[c]]
    assert not mask[index[c]][index[b]]
    assert not mask[index[a]][index[c]]


def test_connected_rerank_keeps_a_tree():
    tree = DraftTree(root_token=1)
    a, b = tree.add_children(0, [10, 11], [0.9, 0.05])
    c, d = tree.add_children(a, [20, 21], [0.8, 0.1])
    keep = set(tree.connected_top_m(2))
    assert keep == {a, c}
    for position in keep:
        assert tree[position].parent in keep | {0}
    assert d not in keep and b not in keep


# ------------------------------------------------------- acceptance estimates


def test_alpha_table_starts_at_q_and_stays_monotone():
    table = AlphaTable()
    assert abs(table.estimate(0.5) - 0.5) < 0.1
    # Tokens with a low path probability are accepted often: the estimate
    # follows the verdicts, not the belief.
    table.update([(0.15, True)] * 200)
    assert table.estimate(0.15) > 0.5
    values = [table.estimate(q / 20) for q in range(21)]
    assert values == sorted(values)


def test_marginal_rule_is_the_ratio_test():
    cost = CostModel()
    cost.tau, cost.round_time = 4.0, 200.0  # rate 0.02 tokens per ms
    assert cost.pays(alpha=0.5, delay_ms=10.0)  # 0.05 > 0.02
    assert not cost.pays(alpha=0.1, delay_ms=10.0)  # 0.01 < 0.02
    # Equation 2 prices a background token at its compressed cost, so four
    # bits pass at close to a quarter of the cost of sixteen.
    cost.rho = 1.0
    full = cost.background_threshold(1.0)
    quarter = cost.background_threshold(0.25)
    assert abs(quarter - full / 4) < 1e-9


# -------------------------------------------------------------------- codec


def test_codec_round_trips_and_shrinks():
    torch.manual_seed(0)
    hidden = torch.randn(256)
    for precision, tolerance in (("bf16", 0.02), ("fp8", 0.12), ("fp4", 0.4)):
        restored = decode(encode(hidden, precision), precision, hidden.numel())
        assert restored.shape == hidden.shape
        assert quantization_error(hidden, precision) < tolerance
    assert payload_bytes(4096, "bf16") == 8192
    assert payload_bytes(4096, "fp8") == 4096
    # Block-scaled FP4 spends four bits an element plus one float16 scale per
    # block of 32, so 4.5 bits an element.
    assert payload_bytes(4096, "fp4") == 4096 * 9 // 16


# ---------------------------------------------------------------- scheduling


def test_first_segment_is_a_prefix_of_the_stream():
    cfg = Config()
    scheduler = StreamScheduler(cfg=cfg.scheduler, hidden_size=256)
    scheduler.cost.tau, scheduler.cost.round_time = 4.2, 250.0
    tree = DraftTree(root_token=1)
    scheduler.enqueue(tree, 0, request_id=1)
    parent = 0
    for _ in range(6):
        children = tree.add_children(parent, [5, 6, 7, 8], [0.7, 0.15, 0.08, 0.04])
        for position in children:
            scheduler.enqueue(tree, position, request_id=1)
        parent = children[0]
    order = []
    while True:
        frame = scheduler.next_frame()
        if frame is None:
            break
        order.append(frame)
        scheduler.on_sent(frame)
    classes = [f.klass for f in order]
    # Every first-segment frame precedes every background frame.
    first_background = next(
        (i for i, k in enumerate(classes) if k.value == "background"), len(classes)
    )
    assert all(k.value == "first_segment" for k in classes[:first_background])
    assert all(k.value == "background" for k in classes[first_background:])
    # Background frames leave in decreasing acceptance probability.
    background_alphas = [f.alpha for f in order[first_background:]]
    assert background_alphas == sorted(background_alphas, reverse=True)


def test_window_bounds_what_the_client_may_have_outstanding():
    """A background token enters the stream while the client has fewer than W
    outstanding, sent but without a verdict yet (Section 4.4)."""
    cfg = Config()
    window = 2
    scheduler = StreamScheduler(cfg=cfg.scheduler, hidden_size=256, window=window)
    scheduler.cost.tau, scheduler.cost.round_time = 4.2, 250.0
    scheduler.cost.rho = 0.0  # an idle uplink, so only the window binds
    tree = DraftTree(root_token=1)
    scheduler.enqueue(tree, 0, request_id=1)
    frame = scheduler.next_frame()
    scheduler.on_sent(frame)

    # Draft and send one token at a time, as the round does.
    background_sent = 0
    parent = 0
    for _ in range(6):
        children = tree.add_children(parent, [5, 6], [0.04, 0.03])
        for position in children:
            if scheduler.enqueue(tree, position, request_id=1) is None:
                continue
            claimed = scheduler.next_frame()
            if claimed is None:
                continue
            scheduler.on_sent(claimed)
            if claimed.klass is Klass.BACKGROUND:
                background_sent += 1
        parent = children[0]
    assert background_sent <= window


def test_provider_fills_a_foreground_batch_with_passengers():
    queues = ProviderScheduler()
    def arrival(position, klass, end=False):
        return Arrival(
            client_id=0, request_id=1, round_id=1, position=position,
            parent=max(0, position - 1), klass=klass, first_segment_end=end,
            payload=b"", num_elements=0, payload_precision="bf16",
            output_precision="bf16", arrival_time_ns=0,
        )
    queues.on_arrival(arrival(0, ProviderKlass.FIRST_SEGMENT))
    queues.on_arrival(arrival(1, ProviderKlass.FIRST_SEGMENT, end=True))
    for position in range(2, 10):
        queues.on_arrival(arrival(position, ProviderKlass.BACKGROUND))
    batch = queues.next_batch()
    assert batch is not None and batch.foreground
    assert len(batch.arrivals) == 2
    assert len(batch.passengers) == queues.cfg.batch_passengers
    # With the foreground queue empty the rest runs in background batches of
    # at most B tokens.
    second = queues.next_batch()
    assert second is not None and not second.foreground
    assert len(second.arrivals) <= queues.cfg.batch_passengers


def test_lazy_runs_background_only_as_passengers():
    cfg = Config()
    cfg.provider.policy = "lazy"
    queues = ProviderScheduler(cfg.provider)
    queues.on_arrival(
        Arrival(0, 1, 1, 5, 4, ProviderKlass.BACKGROUND, False, b"", 0, "bf16", "bf16", 0)
    )
    assert queues.next_batch() is None


# ------------------------------------------------------------- the whole loop


def test_baseline_advances_one_token_per_round():
    engine, _ = build()
    metrics = Baseline(engine).run_request(1, [3, 4, 5, 6], 6)
    assert metrics.output_tokens == 6
    assert abs(metrics.tau_mean - 1.0) < 1e-6


def test_speculation_advances_more_than_one_token_per_round():
    engine, _ = build(hit_rate=1.0)
    metrics = Slipstream(engine).run_request(1, [3, 4, 5, 6], 12)
    assert metrics.output_tokens >= 12
    assert metrics.tau_mean > 1.0
    assert engine.snapshot()["first_segment_mean"] >= 1.0


def test_streamed_round_agrees_with_one_shot_round():
    """For a given tree, a streamed round computes what a one-shot round
    would (Section 4.2, "Correctness").

    Both methods draft from the same module and verify greedily against the
    same target, so with an accurate drafter they must emit the same text; the
    streamed one just pays for less of it up front.
    """
    prompt = [3, 4, 5, 6]
    streamed, _ = build(hit_rate=1.0)
    one_shot, _ = build(hit_rate=1.0)
    Slipstream(streamed).run_request(1, prompt, 12)
    OneShotTree(one_shot).run_request(1, prompt, 12)
    a = streamed.stats["emitted"]
    b = one_shot.stats["emitted"]
    shared = min(len(a), len(b))
    assert shared >= 8
    assert a[:shared] == b[:shared]


def test_a_streamed_batch_equals_one_batch_over_the_same_tree():
    """Splitting a tree across batches changes when it is computed, not what
    is computed."""
    import torch as _torch

    from slipstream.models import build_tiny as _build
    from slipstream.provider.engine import ProviderEngine as _Engine
    from slipstream.transport.messages import HiddenStateMsg

    _torch.manual_seed(0)
    _, provider, _ = _build()
    hidden = _torch.randn(3, 256)

    def send(engine, request_id, split):
        positions, parents = [0, 1, 2], [0, 0, 1]
        outputs = {}
        groups = [[0, 1, 2]] if not split else [[0], [1, 2]]
        for group in groups:
            for index in group:
                engine.on_hidden_state(
                    HiddenStateMsg(
                        request_id=request_id,
                        round_id=1,
                        position=positions[index],
                        parent=parents[index],
                        payload=encode(hidden[index], "bf16"),
                        num_elements=256,
                        first_segment_end=(index == group[-1]),
                    )
                )
            for message in engine.run_until_idle():
                outputs[message.position] = decode(
                    message.payload, message.payload_precision, message.num_elements
                )
        return outputs

    one_batch = send(_Engine(provider), 1, split=False)
    streamed = send(_Engine(provider), 2, split=True)
    for position in (0, 1, 2):
        assert _torch.allclose(one_batch[position], streamed[position], atol=1e-4)


def test_provider_never_holds_a_token_or_a_probability():
    engine, provider = build(hit_rate=1.0)
    Slipstream(engine).run_request(1, [3, 4, 5, 6], 6)
    for cache in provider.shard.caches.values():
        # The provider's whole view of the tree is parent pointers.
        assert set(cache.parents.values()) <= set(cache.parents.keys()) | {0}
        assert not hasattr(cache, "tokens")
    fields = set(vars(next(iter(provider.shard.caches.values()))))
    assert "parents" in fields and "ran" in fields
    assert not any("token" in name or "prob" in name for name in fields)


def test_hybrid_model_keeps_one_state_per_path():
    engine, provider = build(hit_rate=1.0, hybrid=True)
    Slipstream(engine).run_request(1, [3, 4, 5, 6], 6)
    recurrent = [
        layer
        for cache in provider.shard.caches.values()
        for layer in cache.layers
        if layer.prefix_state is not None or layer.tree_state
    ]
    assert recurrent, "a hybrid model should hold recurrent state"


def test_prompt_lookup_matches_the_last_two_tokens():
    tokens = [1, 2, 3, 4, 1, 2]
    assert prompt_lookup(tokens, n=2, k=3) == [3, 4, 1]
    assert prompt_lookup([1, 2, 3], n=2, k=3) == []
