#!/usr/bin/env python3
"""Per-round latency of one verify round, split into five pipeline stages
and stacked per draft budget m. Two panels, one legend: Nemotron-3-Super on
top, GLM-4.5-Air below.

    1. Fill the DATA block below with your measured medians (ms).
    2. python3 stage_stack_vs_budget.py          # preview window, writes nothing
       python3 stage_stack_vs_budget.py --save   # write pdf + png next to this file

The figure is exactly one column wide, so fonts print at their set size
(STIX, 8/9 pt). Both panels share one y scale, so bar heights compare
directly. This file runs on its own and needs only matplotlib.
"""

import argparse
from pathlib import Path

# ####################################################################
# ##                                                                ##
# ##   DATA  --  FILL IN YOUR MEASURED NUMBERS HERE                 ##
# ##                                                                ##
# ####################################################################
#
# One line per stage, one block per model (NEM_ = top panel, GLM_ =
# bottom panel). The i-th value of every line belongs to the bar at M[i],
# so each column below is one bar. Every value is the median time per
# verify round, in milliseconds. To add or drop a budget, edit M and all
# ten stage lines together.
#
# Stages, in the timeline order of one round (bottom to top):
#   DRAFTER_ENCODER  decoder step end -> encoder step end
#                    (the drafter proposes the tree, the encoder runs on it)
#   UPLINK           first uplink byte leaves the client -> the provider
#                    holds the whole payload
#   PROVIDER_MIDDLE  the provider holds the payload -> it sends the first
#                    downlink byte (staging + middle forward)
#   DOWNLINK         first downlink byte leaves the provider -> the last
#                    byte reaches the client
#   DECODER_VERIFY   decoder step (LM head + acceptance walk)

M                   = [    3,     4,     5,     6,     7,     8,     9,    10,    11,    12]

# Nemotron-3-Super: measured medians, filled in 2026-09-11.
NEM_LABEL           = "Nemotron-3-Super"
NEM_DRAFTER_ENCODER = [   21,    20,    23,    25,    26,    31,    33,    39,    43,    45]
NEM_UPLINK          = [59.11, 63.38, 66.66, 70.94, 74.21, 78.49, 81.77, 85.04, 91.32, 96.60]
NEM_PROVIDER_MIDDLE = [ 37.7,  49.6,  61.5,  58.9,  63.6,  73.9,  74.8,  75.5,  80.5,  78.8]
NEM_DOWNLINK        = [46.62, 49.28, 49.93, 50.59, 53.24, 51.90, 54.55, 54.21, 58.86, 62.52]
NEM_DECODER_VERIFY  = [  9.4,  12.6,  11.5,  14.6,  15.7,  13.6,  16.7,  17.5,  18.5,  19.6]

# GLM-4.5-Air: measured medians, filled in 2026-09-13.
GLM_LABEL           = "GLM-4.5-Air"
GLM_DRAFTER_ENCODER = [20.39, 19.33, 22.52,  24.5, 25.48, 30.07, 31.98,    37, 41.16, 43.18]
GLM_UPLINK          = [56.11,    63, 65.66, 69.94, 72.21, 78.49, 77.77, 83.04, 97.32, 99.60]
GLM_PROVIDER_MIDDLE = [ 22.3, 32.23,  43.3, 39.22, 41.22,  50.6, 49.21,  49.1, 52.23, 49.44]
GLM_DOWNLINK        = [42.62, 51.28, 48.93, 52.59, 55.24, 53.90, 57.55, 55.21, 59.86, 66.52]
GLM_DECODER_VERIFY  = [10.51, 14.09, 12.83, 16.27, 17.53, 15.16,  18.6, 19.35, 20.57, 21.50]

# ####################################################################
# STYLE
# ####################################################################

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

FIG_WIDTH_IN  = 3.335   # one USENIX column, so fonts print at their set size
FIG_HEIGHT_IN = 1.42    # each panel's plot area prints 0.39 in tall (0.45 in
                        # at the 1.55 in height before the page cut)

# Printed sizes: nothing below 8 pt, labels at 9 pt. STIX is Times-compatible
# and embeds as TrueType (no Type 3 fonts).
FONT_TICK   = 8.0
FONT_LEGEND = 8.0
FONT_LABEL  = 9.0
FONT_PANEL  = 8.0   # model name inside each panel
FONT_SERIF  = ["STIXGeneral", "DejaVu Serif"]

AXES_LINE_WIDTH = 0.6
TICK_LINE_WIDTH = 0.5
TICK_DIRECTION  = "in"
TICK_LENGTH     = 2.0
BAR_WIDTH       = 0.72
Y_HEADROOM      = 1.07  # y top = tallest bar of either model x this; puts
                        # the top tick label a few pt below each panel's top
Y_TICK_STEP     = 100   # ms

# Stack order = timeline order of one round, bottom to top.
# Red and yellow = network legs. Every pair of the five colors stays
# apart, including under red-green color blindness.
STAGES = [
    ("Drafter + encoder", "#1BAF7A"),   # aqua
    ("Uplink",            "#C0392B"),   # red
    ("Provider middle",   "#2A78D6"),   # blue
    ("Downlink",          "#EDA100"),   # yellow
    ("Decoder verify",    "#E377C2"),   # pink
]

LEGEND_KW = dict(loc="outside upper center", ncol=3, frameon=False,
                 handlelength=1.1, handletextpad=0.4, labelspacing=0.2,
                 columnspacing=0.8, borderpad=0.0, borderaxespad=0.0)
# hspace leaves about 6 pt between the panels, so the upper panel's "0"
# stays clear of the lower panel's top tick label.
LAYOUT_PADS = dict(w_pad=0.01, h_pad=0.01, hspace=0.08)

# ####################################################################

PANELS = [  # (model name, stage lists in STAGES order), top to bottom
    (NEM_LABEL, [NEM_DRAFTER_ENCODER, NEM_UPLINK, NEM_PROVIDER_MIDDLE,
                 NEM_DOWNLINK, NEM_DECODER_VERIFY]),
    (GLM_LABEL, [GLM_DRAFTER_ENCODER, GLM_UPLINK, GLM_PROVIDER_MIDDLE,
                 GLM_DOWNLINK, GLM_DECODER_VERIFY]),
]


def build_figure(m, panels):
    """One panel per (name, stages) pair, stacked top to bottom. `stages`
    holds five value lists in STAGES order, each aligned with `m`."""
    for name, stages in panels:
        for (stage, _), vals in zip(STAGES, stages):
            if len(vals) != len(m):
                raise SystemExit(f"{name}, {stage}: {len(vals)} values "
                                 f"for {len(m)} budgets")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        FONT_SERIF,
        "mathtext.fontset":  "stix",
        "font.size":         FONT_TICK,
        "axes.labelsize":    FONT_LABEL,
        "xtick.labelsize":   FONT_TICK,
        "ytick.labelsize":   FONT_TICK,
        "legend.fontsize":   FONT_LEGEND,
        "axes.linewidth":    AXES_LINE_WIDTH,
        "xtick.major.width": TICK_LINE_WIDTH,
        "ytick.major.width": TICK_LINE_WIDTH,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })

    fig, axes = plt.subplots(len(panels), 1, sharex=True, sharey=True,
                             figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                             layout="constrained")
    fig.get_layout_engine().set(**LAYOUT_PADS)

    tallest = 0.0
    for ax, (name, stages) in zip(axes, panels):
        bottom = [0.0] * len(m)
        for (stage, color), vals in zip(STAGES, stages):
            ax.bar(m, vals, bottom=bottom, width=BAR_WIDTH, color=color,
                   edgecolor="white", linewidth=0.4, label=stage)
            bottom = [b + v for b, v in zip(bottom, vals)]
        tallest = max(tallest, max(bottom))
        # The shortest bars sit at small m, so the top left stays empty.
        ax.text(0.015, 0.97, name, transform=ax.transAxes, ha="left",
                va="top", fontsize=FONT_PANEL)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction=TICK_DIRECTION, length=TICK_LENGTH,
                       width=TICK_LINE_WIDTH)

    low = axes[-1]
    low.set_xticks(m)
    low.set_xlim(min(m) - 0.6, max(m) + 0.6)
    low.set_ylim(0, tallest * Y_HEADROOM)
    low.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
    low.set_xlabel("Draft budget $m$", labelpad=0)
    fig.supylabel("Time per round (ms)", fontsize=FONT_LABEL)

    # Legends fill column by column; reorder so the rows read in timeline
    # order (drafter, uplink, provider / downlink, decoder), like the bars.
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = LEGEND_KW["ncol"]
    rows = -(-len(handles) // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(rows)
             if r * ncol + c < len(handles)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               **LEGEND_KW)

    # Each upper panel's "0" hangs below its frame. Keep it out of the
    # layout, so the gap between panels stays as set.
    fig.canvas.draw()
    for ax in axes[:-1]:
        for lab in ax.get_yticklabels():
            if lab.get_text() == "0":
                lab.set_in_layout(False)
    return fig


def save_figure(fig, stem):
    """Write <stem>.pdf and .png next to this file (no tight bbox: the canvas
    is the printed size)."""
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png", dpi=220)
    print("wrote", HERE / f"{stem}.pdf")
    print("panel heights (in):", [round(ax.get_position().height * fig.get_figheight(), 3) for ax in fig.axes])


def run(m, panels, stem):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true",
                    help="write pdf + png next to this file instead of "
                         "opening a preview window")
    args = ap.parse_args()
    if args.save:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe when only writing files
        save_figure(build_figure(m, panels), stem)
    else:
        import matplotlib.pyplot as plt
        build_figure(m, panels)
        plt.show()


if __name__ == "__main__":
    run(M, PANELS, "stage_stack_vs_budget")
