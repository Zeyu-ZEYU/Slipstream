#!/usr/bin/env python3
"""
2x2 input/output token-length distributions for the four frozen workloads
(datasets/*.jsonl.gz), one panel per request set:
log-x histogram PDFs drawn as step lines with a light fill.

Data: data/dataset_lengths.json — token counts under the
deployed Nemotron tokenizer, precomputed on the server (see
see experiments/datasets.py for where each set comes from). Rebuilding the
json needs the model's tokenizer; redrawing the figure needs only this script
and the json.

Output: dataset_lengths.pdf, written next to this file, exactly one column
wide so every font prints at its set size.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

# =============================================================
# CONFIG — edit anything below to retune the figure.
# =============================================================

# ---- Panels (order = reading order in the 2x2 grid) ----
PANELS = [
    ("sharegpt",      "ShareGPT"),
    ("livecodebench", "LiveCodeBench"),
    ("math",          "MATH"),
    ("longbench",     "LongBench"),
]

# ---- Canvas size (inches): exactly one USENIX column (3.335in) ----
FIG_WIDTH_IN  = 3.335
FIG_HEIGHT_IN = 1.04   # the limit: the one-line y label is 0.99in long (plot areas 0.23in)

# ---- Font sizes (pt), as printed: nothing below 8, labels at 9 ----
FONT_BASE       = 8.0
FONT_AXES_LABEL = 9.0
FONT_TICK       = 8.0
FONT_LEGEND     = 8.0
FONT_TITLE      = 9.0

# STIX is Times-compatible and embeds as TrueType (no Type 3 fonts).
FONT_FAMILY = "serif"
FONT_SERIF  = ["STIXGeneral", "DejaVu Serif"]

# ---- Line styling: output dashed, so the two series also differ in black and white ----
LINE_WIDTH      = 1.1
OUTPUT_LINESTYLE = (0, (2.2, 1.2))
AXES_LINE_WIDTH = 0.6
TICK_LINE_WIDTH = 0.5
GRID_LINE_WIDTH = 0.4
GRID_ALPHA      = 0.45

# ---- Colors: input drives prefill -> red; output drives decode -> blue ----
COLOR_INPUT  = "#c0392b"
COLOR_OUTPUT = "#1f3a93"
FILL_ALPHA   = 0.18

# ---- Ticks / grid ----
TICK_DIRECTION = "in"
TICK_LENGTH    = 2.0

# ---- Axis range & binning (shared log-x across panels) ----
X_MIN, X_MAX = 1.0, 1e5
NUM_BINS     = 36
X_TICKS      = [1e0, 1e2, 1e4]  # every other decade; minors stay unlabeled
X_TICK_LABELS = ["1", "100", "10K"]  # plain text: 10^k superscripts print too small

# ---- Legend (drawn once, where no curve comes near it) ----
# LiveCodeBench has one peak past ~250 tokens, so its upper left (1-100) is
# empty; in MATH the legend handle ran into the output curve's plateau.
LEGEND_PANEL         = None   # no room in a 0.23in panel; the caption names the styles
LEGEND_LOC           = "upper left"
LEGEND_BORDERAXESPAD = 0.1
LEGEND_HANDLELENGTH  = 0.9
LEGEND_BORDERPAD     = 0.05
LEGEND_HANDLETEXTPAD = 0.35
LEGEND_LABELSPACING  = 0.15
LEGEND_FRAMEON       = False

# ---- Spacing (constrained layout; wspace/hspace are inter-panel gaps) ----
TITLE_PAD   = 1.5
# hspace leaves ~3.4 pt above the lower titles (1.5 pt below them), so each
# title reads with its own panel without the rows looking detached.
LAYOUT_PADS = dict(w_pad=0.01, h_pad=0.0, wspace=0.03, hspace=0.06)

# =============================================================
# Implementation below.
# =============================================================


def _apply_rcparams() -> None:
    plt.rcParams.update({
        "font.family":     FONT_FAMILY,
        "font.serif":      FONT_SERIF,
        "mathtext.fontset": "stix",
        "font.size":       FONT_BASE,
        "axes.titlesize":  FONT_TITLE,
        "axes.labelsize":  FONT_AXES_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.linewidth":  AXES_LINE_WIDTH,
        "xtick.major.width": TICK_LINE_WIDTH,
        "ytick.major.width": TICK_LINE_WIDTH,
        "lines.linewidth": LINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })


def _pdf_bins(values: np.ndarray, bins: np.ndarray):
    counts, _ = np.histogram(values, bins=bins)
    centers = np.sqrt(bins[:-1] * bins[1:])  # geometric centers for log-x
    return centers, counts / counts.sum()


def main() -> None:
    _apply_rcparams()
    data = json.load(open(DATA / "dataset_lengths.json"))
    bins = np.geomspace(X_MIN, X_MAX, NUM_BINS + 1)

    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=True,
        layout="constrained",
    )
    fig.get_layout_engine().set(**LAYOUT_PADS)

    for i, (key, title) in enumerate(PANELS):
        ax = axes.flat[i]
        series = [("Input", np.asarray(data[key]["input"]), COLOR_INPUT, "-")]
        if "output" in data[key]:
            series.append(
                ("Output", np.asarray(data[key]["output"]), COLOR_OUTPUT, OUTPUT_LINESTYLE))
        for label, arr, color, ls in series:
            x, y = _pdf_bins(np.clip(arr, X_MIN, X_MAX), bins)
            ax.fill_between(x, y, step="mid", color=color,
                            alpha=FILL_ALPHA, linewidth=0)
            ax.step(x, y, where="mid", color=color, label=label, ls=ls)

        ax.set_xscale("log")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_xticks(X_TICKS, X_TICK_LABELS)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(title, pad=TITLE_PAD)
        ax.grid(True, which="major", linewidth=GRID_LINE_WIDTH,
                alpha=GRID_ALPHA)
        ax.grid(True, which="minor", linewidth=GRID_LINE_WIDTH * 0.6,
                alpha=GRID_ALPHA * 0.5)
        ax.tick_params(direction=TICK_DIRECTION, length=TICK_LENGTH,
                       which="both")
        if LEGEND_PANEL is not None and i == LEGEND_PANEL:
            # Explicit handles: the legend's panel may lack the Output series.
            handles = [
                matplotlib.lines.Line2D([], [], color=c, lw=LINE_WIDTH, label=lb)
                for lb, c in (("Input", COLOR_INPUT), ("Output", COLOR_OUTPUT))
            ]
            ax.legend(handles=handles, loc=LEGEND_LOC, frameon=LEGEND_FRAMEON,
                      handlelength=LEGEND_HANDLELENGTH,
                      borderpad=LEGEND_BORDERPAD,
                      handletextpad=LEGEND_HANDLETEXTPAD,
                      labelspacing=LEGEND_LABELSPACING,
                      borderaxespad=LEGEND_BORDERAXESPAD)

    fig.supxlabel("Length (tokens)", fontsize=FONT_AXES_LABEL)
    fig.supylabel("Fraction of requests", fontsize=FONT_AXES_LABEL)

    # The top row's "0" tick label is centered on the axis bottom, so half of
    # it hangs below and constrained layout would open a gap between the rows
    # for it. Keep only that label out of the layout; the lower row's titles
    # are centered, so it cannot collide with them.
    fig.canvas.draw()
    for ax in axes[0]:
        for lab in ax.get_yticklabels():
            if lab.get_text() == "0":
                lab.set_in_layout(False)

    # No tight bbox: the canvas is the printed size, so fonts print as set.
    out = Path(__file__).with_suffix(".pdf")
    fig.savefig(out)
    print(f"wrote {out}")
    print("panel heights (in):", [round(ax.get_position().height * FIG_HEIGHT_IN, 3) for ax in axes.flat])

    for key, _ in PANELS:
        for kind in ("input", "output"):
            if kind in data[key]:
                a = np.array(data[key][kind])
                print(f"{key:14s} {kind:6s} n={a.size:5d} p50={np.percentile(a,50):7.0f} "
                      f"p90={np.percentile(a,90):7.0f} max={a.max():6d}")


if __name__ == "__main__":
    main()
