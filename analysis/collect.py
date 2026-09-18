#!/usr/bin/env python3
"""Turn a run's json into the data a figure script reads.

Every driver in `experiments/` writes one json file. The figure scripts in
`figures/` read a `DATA` block or a json next to them. `collect.py` bridges the
two, so a fresh run can redraw a figure:

    python experiments/evaluation/overall.py --out results/evaluation/overall.json
    python analysis/collect.py overall results/evaluation/overall.json
    python figures/overall_perf.py --save

Nothing here interprets the numbers; it only reshapes them, and it prints the
block to paste when a figure keeps its data inline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DATA = ROOT / "figures" / "data"

METHOD_NAMES = {
    "baseline": "Baseline",
    "cunningham": "Cunningham",
    "oneshot_chain": "OS-Chain",
    "oneshot_tree": "OS-Tree",
    "slipstream": "Slipstream",
}


def overall(path: Path) -> str:
    """The (goodput, TPOT) pairs `figures/overall_perf.py` plots."""
    runs = json.loads(path.read_text())["runs"]
    by_method: dict[str, list[tuple[int, float, float]]] = {}
    for run in runs:
        method = METHOD_NAMES.get(run["spec"]["method"], run["spec"]["method"])
        by_method.setdefault(method, []).append(
            (
                run["spec"]["requests_in_flight"],
                round(run["goodput_tokens_per_s"], 1),
                round(run["tpot_p50_ms"], 1),
            )
        )
    lines = ["# One line per method: (goodput tokens/s, TPOT P50 ms) per R."]
    loads = sorted({point[0] for points in by_method.values() for point in points})
    lines.append(f"R = {loads}")
    for method, points in by_method.items():
        points.sort()
        pairs = ", ".join(f"({g:6.1f}, {t:6.1f})" for _, g, t in points)
        lines.append(f"# {method:12s} [{pairs}]")
    return "\n".join(lines)


def ablation(path: Path) -> str:
    """The per-variant loss `figures/ablation.py` plots."""
    losses = json.loads(path.read_text())["losses"]
    lines = ["# Goodput lost when one mechanism is removed, per R."]
    for name, per_load in losses.items():
        ordered = sorted(per_load.items(), key=lambda kv: int(kv[0]))
        lines.append(f"# {name:20s} " + ", ".join(f"R={k}: {v:5.1f}%" for k, v in ordered))
    return "\n".join(lines)


def sensitivity(path: Path) -> str:
    data = json.loads(path.read_text())
    lines = ["# Goodput against the window W, per R."]
    for load, per_window in data.get("window", {}).items():
        ordered = sorted(per_window.items(), key=lambda kv: int(kv[0]))
        lines.append(f"# R={load:>3s} " + ", ".join(f"W={k}: {v:7.1f}" for k, v in ordered))
    return "\n".join(lines)


def acceptance(path: Path) -> str:
    """A chain or budget sweep, in the shape the motivation figures read."""
    data = json.loads(path.read_text())
    target = FIGURE_DATA / "acceptance_from_run.json"
    target.write_text(json.dumps(data, indent=2))
    return f"wrote {target}"


def stages(path: Path) -> str:
    data = json.loads(path.read_text())
    target = FIGURE_DATA / "stage_stack_from_run.json"
    target.write_text(json.dumps(data, indent=2))
    return f"wrote {target}"


SHAPERS = {
    "overall": overall,
    "ablation": ablation,
    "sensitivity": sensitivity,
    "acceptance": acceptance,
    "stages": stages,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(SHAPERS))
    parser.add_argument("path")
    args = parser.parse_args()
    print(SHAPERS[args.kind](Path(args.path)))


if __name__ == "__main__":
    main()
