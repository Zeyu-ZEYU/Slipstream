#!/usr/bin/env python3
"""The two metrics, and the mechanism counters behind them.

Section 6.1 defines goodput as output tokens per second summed over the
clients and the time per output token as the median over requests. This module
computes both from a run's json and prints what the mechanisms did, which is
what a functional run should be read for.

    python analysis/stats.py results/evaluation/overall.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize_run(run: dict) -> dict:
    spec = run["spec"]
    client = run.get("client", {})
    provider = run.get("provider", {})
    queues = provider.get("queues", {})
    return {
        "method": spec["method"],
        "requests_in_flight": spec["requests_in_flight"],
        "goodput_tokens_per_s": round(run["goodput_tokens_per_s"], 2),
        "tpot_p50_ms": round(run["tpot_p50_ms"], 2),
        "tpot_p99_ms": round(run["tpot_p99_ms"], 2),
        "tau_mean": round(run["tau_mean"], 3),
        "first_segment_mean": round(run["first_segment_mean"], 2),
        "continuations": client.get("continuations", 0),
        "hidden_states_sent": client.get("sent", 0),
        "bytes_up_per_output_token": (
            round(run["bytes_up"] / run["output_tokens"], 1) if run["output_tokens"] else 0
        ),
        "provider_batches": provider.get("batches_run", 0),
        "provider_tokens": provider.get("tokens_run", 0),
        "passengers": provider.get("passengers_run", 0),
        "dropped_arrivals": queues.get("dropped", 0),
        "note": run.get("note", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", action="store_true", help="print json instead of a table")
    args = parser.parse_args()

    rows = []
    for path in args.paths:
        payload = json.loads(Path(path).read_text())
        runs = payload.get("runs") if isinstance(payload, dict) else None
        if isinstance(runs, list):
            rows.extend(summarize_run(run) for run in runs)
        elif isinstance(runs, dict):
            rows.extend(
                {"method": "slipstream", "requests_in_flight": int(k), **v}
                for k, v in runs.items()
                if isinstance(v, dict)
            )
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print("nothing to summarize")
        return
    header = (
        f"{'method':16s} {'R':>4s} {'goodput':>9s} {'TPOT P50':>9s} {'tau':>6s} "
        f"{'s':>5s} {'cont':>5s} {'pass':>5s} {'B/token':>8s}"
    )
    print(header)
    for row in rows:
        print(
            f"{row.get('method',''):16s} {row.get('requests_in_flight',0):4d} "
            f"{row.get('goodput_tokens_per_s',0):9.1f} {row.get('tpot_p50_ms',0):9.1f} "
            f"{row.get('tau_mean',0):6.2f} {row.get('first_segment_mean',0):5.1f} "
            f"{row.get('continuations',0):5d} {row.get('passengers',0):5d} "
            f"{row.get('bytes_up_per_output_token',0):8.1f}"
        )
    notes = {row.get("note", "") for row in rows if row.get("note")}
    for note in notes:
        print(f"\nnote: {note}")
    taus = [row.get("tau_mean", 0) for row in rows if row.get("tau_mean")]
    if taus:
        print(f"acceptance length across runs: median {statistics.median(taus):.2f}")


if __name__ == "__main__":
    main()
