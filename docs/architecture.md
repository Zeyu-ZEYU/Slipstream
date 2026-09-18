# The design, and where each part of it lives

This maps the paper onto the code. Section numbers are the paper's.

## One round

```
client                                                     provider
------                                                     --------
draft best-first, k children per step  (4.2)
  |
  +-- root's hidden state, no drafting needed --------->  stages
  +-- first-segment states, in alpha order ------------>  stages
  +-- the marked state ------------------------------->  runs the segment as one batch
  |                                                       |
  |   background states, while the rule admits ------->   background queue
  |                                                       |
  <---------------- outputs, foreground first ------------+
verify with the head, find the accepted path (4.2)
  |
  +-- verdict marked continue, if waiting pays ------->  runs the matched subtree
  |
  close: the next round's first state names the accepted leaf (Appendix C)
```

## The rule

Everything is one test (Section 4.1). Adding a draft token `v` raises goodput
exactly when

```
alpha_v / Delta > tau / T
```

with `alpha_v` the probability that the target's path runs through `v`, and
`Delta` the delay that sending or waiting for `v` adds to work someone is
waiting for. It is applied three times:

| Delta | Decides | Code |
| --- | --- | --- |
| `c` | the first segment's length `s` | `CostModel.first_segment_threshold`, `StreamScheduler.classify` |
| `c_u * rho` | whether a background token enters the stream | `CostModel.background_threshold`, same `classify` |
| the wait for a continuation's outputs | whether the round continues | `CostModel.continuation_pays`, `Slipstream.run_round` |

With compression on, admission prices a background token at its compressed
cost, `b_v / 16` (Section 4.5), which is `PrecisionPolicy.uplink_cost_scale`.

## The parts

| Section | Part | Module |
| --- | --- | --- |
| 2.2 | dynamic draft tree, rerank by path probability | `slipstream/tree.py` |
| 4.1 | the cost model and the marginal rule | `slipstream/costs.py` |
| 4.2 | stream order, acceptance estimates, the first segment, branch continuation | `slipstream/scheduler.py`, `slipstream/methods/slipstream.py`, `slipstream/alpha.py` |
| 4.3 | two queues, passengers, the window, the two extremes | `slipstream/provider/queues.py` |
| 4.4 | two traffic classes, admission, priorities, the regimes, online inputs | `slipstream/scheduler.py`, `slipstream/costs.py` |
| 4.5 | precision by acceptance probability, calibration, online checks | `slipstream/codec.py` |
| 5 | split shards, tree attention, the transport | `slipstream/models/`, `slipstream/transport/`, `gateway/` |
| 6.1 | the five comparison methods | `slipstream/methods/` |

## What the provider may know

Appendix C fixes it, and `slipstream/models/cache.py` is the whole of it: per
round the tree received so far as parent pointers, whether each position's
hidden state has arrived and has run, and the key and value entries and
recurrent states of the positions it has run. A verdict names positions. A
round closes without a verdict, because the next round's first hidden state
names the accepted leaf.

What the provider learns beyond a one-shot tree is listed in Section 4.1: the
arrival order reveals the drafter's ranking, the first segment's length its
confidence, a verdict marked continue the accepted positions before the round
ends, and a compressed hidden state's precision the range of its acceptance
estimate. The values themselves are obfuscated exactly as before
(`slipstream/models/obfuscation.py`).

## Correctness

For a given tree, a streamed round computes what a one-shot round would. Two
things make that true, and both are tested:

* each batch attends only to its ancestors in the tree and reads the state
  earlier batches wrote, which is the exact ancestor mask merged by
  log-sum-exp (`slipstream/models/layers.py:attend_split`,
  `tests/test_attention.py`);
* a hybrid model keeps one recurrent state per path rather than per sequence
  (`LayerCache.tree_state`, `tests/test_mechanisms.py`).

Compression is the one deliberate deviation (Section 4.5), and it is off by
default.
