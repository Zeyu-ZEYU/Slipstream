#!/usr/bin/env python3
"""Recompute the paper's own figures from the data that ships with them.

The numbers in Sections 3 and 6 are read off the figures, so they can be
checked against the data behind those figures without running anything:

    python analysis/check_paper_numbers.py

It reads the data behind each figure and recomputes the quantities the text
states, so a reader can see the sentences and the figures come from the same
numbers. It needs no hardware.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "figures" / "data"


def load(name: str) -> dict:
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else {}


def chain_saturation() -> None:
    print("Section 3.2, the chain saturates early")
    for name, model in (
        ("acceptance_vs_draft.json", "Nemotron-3-Super"),
        ("acceptance_vs_draft_glm.json", "GLM-4.5-Air"),
    ):
        data = load(name)
        taus = data.get("tau", {})
        if not taus:
            continue
        gains, at_eight = [], []
        for workload, points in taus.items():
            six, eight = points.get("6"), points.get("8")
            if six and eight:
                gains.append(eight - six)
                at_eight.append(eight)
        if gains:
            print(
                f"  {model:18s} tau at m=8: {min(at_eight):.2f} to {max(at_eight):.2f}; "
                f"largest gain from m=6 to m=8: {max(gains):.2f}"
            )


def tree_shapes() -> None:
    print("Section 3.2, trees beat chains and shape is secondary")
    for name, model in (
        ("tree_shape_tau.json", "Nemotron-3-Super"),
        ("tree_shape_tau_glm.json", "GLM-4.5-Air"),
    ):
        data = load(name)
        chain, shapes = data.get("chain", {}), data.get("shapes", {})
        if not chain or not shapes:
            continue
        leads, spreads = [], []
        for workload, chain_tau in chain.items():
            values = [shapes[shape][workload] for shape in shapes if workload in shapes[shape]]
            if not values:
                continue
            leads.append(min(values) - chain_tau)
            spreads.append(max(values) - min(values))
        if leads:
            print(
                f"  {model:18s} every tree leads the chain by at least "
                f"{min(leads):.2f}; across shapes tau moves by at most {max(spreads):.2f}"
            )


def budget_sweep() -> None:
    print("Section 3.2, only trees turn budget into tokens")
    for name, model in (
        ("tree_budget_tau.json", "Nemotron-3-Super"),
        ("tree_budget_tau_glm.json", "GLM-4.5-Air"),
    ):
        data = load(name)
        tree = data.get("tree") or data.get("tau", {}).get("tree")
        chain = data.get("chain") or data.get("tau", {}).get("chain")
        if not isinstance(tree, dict) or not isinstance(chain, dict):
            continue
        leads = []
        for workload, points in tree.items():
            if not isinstance(points, dict):
                continue
            ten = points.get("10")
            chain_ten = (chain.get(workload) or {}).get("10") if isinstance(chain.get(workload), dict) else None
            if ten and chain_ten:
                leads.append(ten - chain_ten)
        if leads:
            print(
                f"  {model:18s} at m=10 the (6,2) tree leads the chain by "
                f"{min(leads):.2f} to {max(leads):.2f}"
            )


def stage_breakdown() -> None:
    """Section 3.3 reads its numbers off Figure 4, whose five series live in
    the figure script, so the check reads them from there."""
    import re

    print("Section 3.3, the network is the largest cost of a round")
    script = (
        Path(__file__).resolve().parents[1] / "figures" / "stage_stack_vs_budget.py"
    )
    if not script.exists():
        return
    source = script.read_text()

    def series(name: str) -> list[float]:
        match = re.search(rf"^{name}\s*=\s*(\[[^\]]*\])", source, re.M)
        return [float(v) for v in eval(match.group(1))] if match else []

    for prefix, model in (("NEM", "Nemotron-3-Super"), ("GLM", "GLM-4.5-Air")):
        stages = [
            series(f"{prefix}_{name}")
            for name in (
                "DRAFTER_ENCODER",
                "UPLINK",
                "PROVIDER_MIDDLE",
                "DOWNLINK",
                "DECODER_VERIFY",
            )
        ]
        if not all(stages):
            continue
        totals = [sum(column) for column in zip(*stages)]
        uplink, downlink = stages[1], stages[3]
        shares = [(u + d) / t for u, d, t in zip(uplink, downlink, totals)]
        print(
            f"  {model:18s} network is {min(shares):.0%} to {max(shares):.0%} of a "
            f"round; the round takes {min(totals):.0f} to {max(totals):.0f} ms"
        )


def candidate_pools() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from slipstream.tree import candidate_pool_size

    print("Section 3.2, the candidate pools")
    for depth, width in ((6, 2), (6, 4), (10, 3)):
        print(f"  a ({depth},{width}) tree offers {candidate_pool_size(depth, width)} candidates")


def main() -> None:
    chain_saturation()
    tree_shapes()
    budget_sweep()
    stage_breakdown()
    candidate_pools()


if __name__ == "__main__":
    main()
