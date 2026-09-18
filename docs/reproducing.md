# Figure by figure

What each figure is, what draws it, and what measures it. Figure numbers are
the paper's.

| Figure | What it shows | Draw it | Measure it |
| --- | --- | --- | --- |
| 1 | the split | hand-drawn | — |
| 2 | a dynamic draft tree | `figures/tree_example.py` | — |
| 3 | chain drafting saturates early | `figures/acceptance_vs_draft.py` | `experiments/motivation/chain_sweep.py` |
| 4 | budgets to 22, chain against trees of width 2 and 4 | `figures/tree_budget_ext.py` | `experiments/motivation/budget_sweep.py` |
| 5 | one round in five stages, per budget | `figures/stage_stack_vs_budget.py` | `experiments/motivation/stage_breakdown.py` |
| 6 | drafted trees and accepted paths, by rank | `figures/tree_overlap_paper.py` | `experiments/motivation/rank_overlap.py` |
| 7 | the architecture and the eight steps of a round | `figures/design_arch.py --regenerate` | — |
| 8 | one round's tree and the stream it becomes | `figures/design_tree.py` | — |
| 9 | one round, one-shot against streamed | `figures/design_timeline.py` | — |
| 10 | overall performance | `figures/overall_perf.py` | `experiments/evaluation/overall.py` |
| 11 | the ablation | `figures/ablation.py` | `experiments/evaluation/ablation.py` |
| 12 | at scale | `figures/overall_scale.py` | `experiments/evaluation/at_scale.py` |
| Table 1 | the symbols of the design | in the paper | — |
| Table 2 | stream compression | in the paper | `experiments/evaluation/stream_compression.py` |
| A1 | request-set lengths | `figures/dataset_lengths.py` | `experiments/datasets.py` |
| A2 | tree shape at a fixed budget | `figures/tree_shape_tau.py` | `experiments/motivation/tree_shape_sweep.py` |
| A3 | the (6,2) budget sweep per workload | `figures/tree_budget_tau.py` | `experiments/motivation/budget_sweep.py` |
| A4 | sensitivity to the window | `figures/sensitivity_w.py` | `experiments/evaluation/sensitivity.py` |

Every figure script takes `--save` to write the file and, without it, opens a
window. The data each one plots sits in `figures/data/` or in a `DATA` block at
the top of the script, and `analysis/collect.py` reshapes a fresh run into
that form.

## Checking the paper's numbers

```bash
python3 analysis/check_paper_numbers.py
```

This recomputes, from the data behind the figures, the numbers the paper states
in prose: the acceptance length at a budget of eight and the gain from six to
eight, every tree's lead over the chain and the spread across shapes, the
(6,2) tree's lead at a budget of ten, the network's share of a round and the
round time, and the candidate pools of the three shapes. It is a consistency
check between the figures and the sentences, and it needs no hardware.

## What a run records

Every driver writes the two metrics of Section 6.1, goodput and the median
time per output token, and the counters that say what the mechanisms did: the
acceptance length, the first segment's length, continuations, passengers the
provider carried, dropped arrivals, and uplink bytes per output token.
`analysis/stats.py` prints them, and each run's json also records the settings
it used, so two runs can be compared knob by knob.

`README.md` lists the settings that point the same code at a checkpoint, a
GPU, the vendor's drafter and a shaped link.
