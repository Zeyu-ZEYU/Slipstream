#!/usr/bin/env python3
"""Tau across draft-tree shapes at a fixed budget m=8, vs the chain, one
panel per model: Nemotron-3-Super on top, GLM-4.5-Air below.

Data: tree_shape_tau.json (Nemotron) and tree_shape_tau_glm.json (GLM,
written by analysis/collect.py), both next to this file. Output:
tree_shape_tau.pdf, exactly one column wide so every font prints at its
set size. Same layout as acceptance_vs_draft.py: the workload legend runs
across the top and each panel names its model.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# =============================================================
# CONFIG
# =============================================================

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = [  # (data file, panel label), top to bottom
    ("tree_shape_tau.json",     "Nemotron-3-Super"),
    ("tree_shape_tau_glm.json", "GLM-4.5-Air"),
]

FIG_WIDTH_IN  = 3.335   # one USENIX column, so fonts print at their set size
FIG_HEIGHT_IN = 1.230   # solved so each plot area = its y span x IN_PER_TAU

# Printed sizes: nothing below 8 pt, labels at 9 pt. STIX is Times-compatible
# and embeds as TrueType (no Type 3 fonts).
FONT_TICK   = 8.0
FONT_LEGEND = 8.0
FONT_LABEL  = 9.0
FONT_PANEL  = 8.0   # model name inside each panel
FONT_SERIF  = ["STIXGeneral", "DejaVu Serif"]

LINE_WIDTH      = 1.2
AXES_LINE_WIDTH = 0.6
TICK_LINE_WIDTH = 0.5
GRID_LINE_WIDTH = 0.4
GRID_ALPHA      = 0.45
MARKER_SIZE     = 3.2

SERIES = [  # key, label, color, marker -- same encoding in every figure
    ("math",          "MATH",          "#27ae60", "^"),
    ("longbench",     "LongBench",  "#DAA520", "d"),
    ("sharegpt",      "ShareGPT",      "#c0392b", "o"),
    ("livecodebench", "LiveCodeBench", "#1f3a93", "s"),
]
SHAPE_ORDER = ["chain", "(6,2)", "(6,3)", "(6,4)", "(8,2)", "(8,3)", "(8,4)"]
X_LABELS = ["Chain", "6,2", "6,3", "6,4", "8,2", "8,3", "8,4"]

# Each panel's y range hugs its own model's data: the models sit at
# different levels, so a shared range would leave an empty band in each.
# Both panels keep IN_PER_TAU inches per tau unit, so the heights follow
# the spans and a tau gap looks the same in both.
IN_PER_TAU  = 0.2000 # 0.235 before the page cut; 0.5 tau steps stay 0.10 in apart
Y_PAD_IN    = 0.05   # in added below/above the data, then rounded to 0.1 tau;
                     # the bottom pad also clears the in-panel model name
Y_PAD       = Y_PAD_IN / IN_PER_TAU
BOTTOM_EXTRA = 0.3   # tau; room under the lowest line for the model name at the bottom right
Y_TICK_STEP = 0.5    # minor ticks and grid lines, unlabeled
Y_LABEL_STEP = 1.0   # labeled ticks; 0.5 tau labels would touch at this panel height
Y_TICK_EDGE = 0.06   # in; no tick label closer than this to a panel edge

TICK_DIRECTION = "in"
TICK_LENGTH    = 2.0

LEGEND_KW = dict(loc="outside upper center", ncol=len(SERIES), frameon=False,
                 handlelength=1.2, handletextpad=0.3, columnspacing=0.8,
                 borderpad=0.0, borderaxespad=0.0)
LAYOUT_PADS = dict(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.06)

# =============================================================


def _apply_rcparams() -> None:
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
        "lines.linewidth":   LINE_WIDTH,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def _series(data, key):
    return [data["chain"][key] if s == "chain" else data["shapes"][s][key]
            for s in SHAPE_ORDER]


def _y_range(data):
    vals = [v for key, *_ in SERIES for v in _series(data, key)]
    return (math.floor((min(vals) - Y_PAD - BOTTOM_EXTRA) * 10) / 10,
            math.ceil((max(vals) + Y_PAD) * 10) / 10)


def _y_ticks(lo, hi, step):
    edge = Y_TICK_EDGE / IN_PER_TAU
    k0 = math.ceil(lo / step)
    ticks = [k * step for k in range(k0, int(hi / step) + 1)]
    return [t for t in ticks if lo + edge <= t <= hi - edge]


def main() -> None:
    _apply_rcparams()
    datas = [json.loads((DATA / f).read_text()) for f, _ in MODELS]
    ranges = [_y_range(d) for d in datas]

    fig, axes = plt.subplots(
        len(MODELS), 1, sharex=True, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        layout="constrained",
        gridspec_kw={"height_ratios": [hi - lo for lo, hi in ranges]})
    fig.get_layout_engine().set(**LAYOUT_PADS)

    xs = range(len(SHAPE_ORDER))
    for ax, data, (lo, hi), (_, name) in zip(axes, datas, ranges, MODELS):
        for key, label, color, marker in SERIES:
            ax.plot(xs, _series(data, key), color=color, marker=marker,
                    ms=MARKER_SIZE, lw=LINE_WIDTH, label=label)
        # Visual cue: everything right of the divider is a tree shape.
        ax.axvline(0.5, color="#999999", lw=0.6, ls=":")
        ax.set_ylim(lo, hi)
        major = _y_ticks(lo, hi, Y_LABEL_STEP)
        ax.set_yticks(major)
        ax.set_yticks([t for t in _y_ticks(lo, hi, Y_TICK_STEP) if t not in major],
                      minor=True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.text(0.985, 0.06, name, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=FONT_PANEL)
        ax.grid(True, axis="y", which="major", lw=GRID_LINE_WIDTH, alpha=GRID_ALPHA)
        ax.grid(True, axis="y", which="minor", lw=GRID_LINE_WIDTH * 0.6,
                alpha=GRID_ALPHA * 0.6)
        ax.tick_params(direction=TICK_DIRECTION, length=TICK_LENGTH,
                       width=TICK_LINE_WIDTH)
        ax.tick_params(which="minor", direction=TICK_DIRECTION,
                       length=TICK_LENGTH * 0.6, width=TICK_LINE_WIDTH)

    bottom = axes[-1]
    bottom.set_xticks(list(xs))
    bottom.set_xticklabels(X_LABELS)
    bottom.set_xlabel(r"Draft shape at budget $m{=}8$: chain or tree $(n, k)$", labelpad=0)
    fig.supylabel(r"Acceptance length $\tau$", fontsize=FONT_LABEL)

    handles = [Line2D([], [], color=c, marker=mk, markersize=MARKER_SIZE,
                      lw=LINE_WIDTH, label=lb) for _, lb, c, mk in SERIES]
    fig.legend(handles=handles, **LEGEND_KW)

    # No tight bbox: the canvas is the printed size.
    out = HERE / "tree_shape_tau.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    fig.canvas.draw()
    print("panel heights (in):", [round(ax.get_position().height * FIG_HEIGHT_IN, 3) for ax in axes])


if __name__ == "__main__":
    main()
