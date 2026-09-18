# The vLLM backend

The paper's system is built into vLLM 0.24, which keeps the model loader, the
cache managers, the attention kernels, and the API; both sides run the same
fork (Section 5). This directory says where the fork's changes go, so the same
design runs on the paper's two models at the paper's speed.

The package in `slipstream/` is backend independent. Everything that is the
design lives there and is exercised by the tests:

| Part of the design | Module | Needs from the engine |
| --- | --- | --- |
| Draft tree, ranks, ancestor mask | `slipstream/tree.py` | nothing |
| Acceptance probability by path probability | `slipstream/alpha.py` | the drafter's probabilities |
| The marginal rule and the cost model | `slipstream/costs.py` | round times |
| Stream scheduler, first segment, admission, priorities | `slipstream/scheduler.py` | one hook after each drafting step |
| Two queues, passengers, the window | `slipstream/provider/queues.py` | a scheduler hook per batch |
| Verdicts and round state | `slipstream/provider/engine.py`, `slipstream/models/cache.py` | the cache manager |
| Codec and precision by acceptance | `slipstream/codec.py` | the engine's own FP8 and FP4 paths |
| Wire protocol | `proto/slipstream.proto`, `slipstream/transport/` | shared memory to the gateway |

## The five hook points

1. **Client model runner.** A runner that holds the embedding, the first
   layer, the final norm, the language-model head and the vendor's
   multi-token-prediction module, and nothing else. `slipstream/models/split_model.py`
   is the reference implementation of that split; in the fork it is a vLLM
   model runner with the middle layers left out.

2. **Drafting, one top-k step per level.** vLLM's speculative proposer already
   proposes a chain. The fork adds one top-k step per level and keeps the tree
   by path probability, which is `DraftTree.add_children` plus
   `best_first_candidate` here. The scheduler runs beside the proposer, where
   the path probabilities are, and updates the table of acceptance rates by
   path probability after every round.

3. **Provider scheduler.** vLLM's scheduler gains the foreground and
   background queues, the passengers and the window of Section 4.3. The policy
   is `ProviderScheduler.next_batch`; what the fork supplies is the batch
   builder and the cache manager underneath it.

4. **Tree attention.** A foreground batch attends to the prefix with the
   regular kernel and to the tree under an exact ancestor mask, merged by
   log-sum-exp. `slipstream/models/layers.py:attend_split` is that merge
   written out and checked against attention over the concatenation; in the
   fork it is the paged-attention kernel called twice and merged, which is
   what a partitioned attention kernel does anyway. Mamba layers keep one
   recurrent state per path, which is `LayerCache.tree_state` here.

5. **Codec.** The fork uses vLLM's own FP8 cast and its block-scaled FP4
   format rather than the reference codec in `slipstream/codec.py`; the
   placement rule, precision by acceptance probability, is unchanged.

## Applying the changes

`patches/` holds the diffs against vLLM 0.24 as they were applied, one file
per hook point, with the paths they touch. They are ordered:

```
0001-client-model-runner.patch
0002-tree-proposer-topk-per-level.patch
0003-provider-queues-window-passengers.patch
0004-tree-attention-ancestor-mask-lse.patch
0005-shared-memory-transport.patch
```

To build the fork:

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
git checkout v0.24.0
for p in ../vllm_integration/patches/*.patch; do git apply "$p"; done
pip install -e .
```

Then point the deployment scripts at the fork's Python environment; the
gateway, the protocol and every rule above are unchanged.

## Running without the fork

Everything in this repository runs without vLLM, on the reference shards in
`slipstream/models/`: the same scheduler, the same queues, the same protocol
and the same codec, over plain torch modules that load a Llama-style
checkpoint or the small built-in model. That path is what the tests use and
what makes each mechanism readable on its own. The fork is what serves the
mixture-of-experts models of Section 3.1 with the engine's own kernels, cache
managers and quantized formats.
