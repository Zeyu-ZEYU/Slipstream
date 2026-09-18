#!/usr/bin/env python3
"""Figure 5: where the budget goes, and where acceptance lands.

Section 3.4 overlays every verified round of a (6,2) tree at m=8 and counts,
per rank, the share of drafted tokens and the share of accepted tokens. Ranks
1 and 2 take almost all of the budget, and acceptance is more concentrated
still.

Two ways to run it:

  * over a fresh run of this artifact, which is what `--live` does;
  * over the verified-round dumps that ship in
    `figures/data/tree_dumps/`, which is what the paper's figure is drawn
    from and what `figures/tree_overlap_paper.py --save` redraws.

    python experiments/motivation/rank_overlap.py --live
    python experiments/motivation/rank_overlap.py --dumps figures/data/tree_dumps/nemotron
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import datasets
from experiments.acceptance import build_harness
from experiments.common import RunSpec, write_result


def from_dumps(directory: Path) -> dict:
    """Rank shares from a dump of verified rounds."""

    def read(name: str) -> list[dict]:
        plain = directory / name
        if plain.exists():
            return [json.loads(line) for line in open(plain)]
        with gzip.open(directory / (name + ".gz"), "rt") as fh:
            return [json.loads(line) for line in fh]

    trees = read("tree_dump_math_n6k2_m8.jsonl")
    verdicts = read("verify_dump_math_n6k2_m8.jsonl")
    drafted, accepted = Counter(), Counter()
    for tree, verdict in zip(trees, verdicts):
        parents = tree["par"]
        scores = tree.get("score") or []
        keep = tree.get("keep") or list(range(len(parents)))
        # Rank within a level, by path probability, as Section 3.4 defines it.
        depth = {}
        for index, parent in enumerate(parents):
            depth[index] = 0 if parent in (index, -1) else depth.get(parent, 0) + 1
        by_level: dict[int, list[int]] = {}
        for index in keep:
            by_level.setdefault(depth.get(index, 0), []).append(index)
        rank_of = {}
        for _, members in by_level.items():
            ordered = sorted(members, key=lambda i: -(scores[i] if i < len(scores) else 0.0))
            for rank, index in enumerate(ordered, start=1):
                rank_of[index] = rank
        for index in keep:
            if depth.get(index, 0) > 0:
                drafted[rank_of.get(index, 1)] += 1
        for index in verdict.get("path", []):
            if index in rank_of and depth.get(index, 0) > 0:
                accepted[rank_of[index]] += 1
    return {"drafted_by_rank": dict(drafted), "accepted_by_rank": dict(accepted), "rounds": len(verdicts)}


def shares(counts: dict) -> dict:
    total = sum(counts.values()) or 1
    return {str(rank): round(100 * value / total, 2) for rank, value in sorted(counts.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="", help="a directory of verified-round dumps")
    parser.add_argument("--live", action="store_true", help="measure from a fresh run instead")
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--drafter", default="oracle", choices=["oracle", "mtp"])
    parser.add_argument("--hit-rate", type=float, default=0.6)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--shape", default="6,2")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/motivation/rank_overlap.json")
    args = parser.parse_args()

    if args.dumps:
        counts = from_dumps(Path(args.dumps))
        out = {
            "source": f"dumps at {args.dumps}",
            "rounds": counts["rounds"],
            "drafted_share_by_rank": shares(counts["drafted_by_rank"]),
            "accepted_share_by_rank": shares(counts["accepted_by_rank"]),
        }
    else:
        depth, width = (int(part) for part in args.shape.split(","))
        spec = RunSpec(model=args.model, drafter=args.drafter, hit_rate=args.hit_rate, seed=args.seed)
        harness = build_harness(spec)
        prompts = datasets.load("math", args.requests, seed=args.seed)
        result = harness.measure(prompts, (depth, width), args.budget, args.max_output_tokens)
        out = {
            "source": f"live run, {args.model} model, {args.drafter} drafter",
            "rounds": result.rounds,
            "tau": round(result.tau_mean, 3),
            "drafted_share_by_rank": shares(result.drafted_by_rank),
            "accepted_share_by_rank": shares(result.accepted_by_rank),
            "rank1_accept_rate": round(result.rank1_accept_rate, 3),
            "rank2_rescue_rate": round(result.rank2_rescues, 3),
        }
    for key in ("drafted_share_by_rank", "accepted_share_by_rank"):
        print(f"{key:26s} {out[key]}")
    if "rank2_rescue_rate" in out:
        print(f"{'rank-2 rescues':26s} {out['rank2_rescue_rate']:.1%} of rounds whose first choice missed")
    print("wrote", write_result(args.out, out))


if __name__ == "__main__":
    main()
