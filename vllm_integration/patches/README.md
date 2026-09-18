# Patches against vLLM 0.24

Each patch is one hook point of `../README.md`. They are kept separate so a
reader can see exactly what the engine had to give the design, and so a
different engine can be approached the same way.

| Patch | Touches | What it adds |
| --- | --- | --- |
| `0001-client-model-runner.patch` | `vllm/worker/`, `vllm/model_executor/` | a runner that loads the embedding, the first layer, the norm, the head and the vendor's multi-token-prediction module, and a provider runner that loads the middle layers only |
| `0002-tree-proposer-topk-per-level.patch` | `vllm/spec_decode/` | one top-k step per level in the speculative proposer, a tree kept by path probability, and the hook the stream scheduler attaches to |
| `0003-provider-queues-window-passengers.patch` | `vllm/core/scheduler.py` | the foreground and background queues, the passenger budget `B`, the window `W`, and verdict handling |
| `0004-tree-attention-ancestor-mask-lse.patch` | `vllm/attention/` | the exact ancestor mask, the log-sum-exp merge of the prefix and tree partitions, and one recurrent state per path for hybrid models |
| `0005-shared-memory-transport.patch` | `vllm/engine/` | handing hidden states to the gateway's ring in stream order, at the precision the acceptance probability calls for |

The diffs are not included in this snapshot: they carry the fork's own history
and would go stale against any other vLLM revision. The table above and
`../README.md` name every file and every entry point they touch, and the
reference implementation in `slipstream/` is the specification they were
written against.
