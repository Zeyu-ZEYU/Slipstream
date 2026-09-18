#!/usr/bin/env python3
"""Mean acceptance length (tau) vs draft budget m for chain MTP drafting,
one panel per model: Nemotron-3-Super on top, GLM-4.5-Air below.

Data: acceptance_vs_draft.json (Nemotron, written by analysis/collect.py) and acceptance_vs_draft_glm.json (GLM,
written by analysis/collect.py), both next to this file.
Output: acceptance_vs_draft.pdf, exactly one column wide so every font
prints at its set size. The workload legend runs across the top; the
dashed line is the ideal tau = m + 1.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================
# CONFIG
# =============================================================

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = [  # (data file, panel label), top to bottom
    ("acceptance_vs_draft.json",     "Nemotron-3-Super"),
    ("acceptance_vs_draft_glm.json", "GLM-4.5-Air"),
]

FIG_WIDTH_IN  = 3.335   # one USENIX column, so fonts print at their set size
FIG_HEIGHT_IN = 1.240   # plot areas = y span x IN_PER_TAU (0.357in for 1.2-4.6)

# Printed sizes: nothing below 8 pt, labels at 9 pt. STIX is Times-compatible
# and embeds as TrueType (no Type 3 fonts).
FONT_TICK   = 8.0
FONT_LEGEND = 8.0
FONT_LABEL  = 9.0
FONT_PANEL  = 8.0   # model name inside each panel
FONT_IDEAL  = 8.0
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
IDEAL_COLOR = "#666666"
# The ideal label runs along the dashed line, just above it, in the top
# panel only; horizontal text would cross the line in a panel this short.
IDEAL_TEXT_XY = (1.02, 2.12)   # data coords; lower-left so the label stays inside the frame
IDEAL_ROTATION = 45            # degrees in data space = the slope-1 line

# The y range hugs the data: min/max tau at the plotted m, padded by Y_PAD
# and rounded out to 0.1. Panels keep IN_PER_TAU inches per tau unit, so
# trimming empty range shortens them without squeezing the curves.
Y_PAD      = 0.2   # ~3 pt between the top markers and the frame
Y_PAD_LOW  = 0.55  # more room below: the panel name sits under the lowest curve
IN_PER_TAU = 0.105   # 0.125 before the page cut; the earlier 0.703in panels over 1.0-5.3
TICK_DIRECTION = "in"
TICK_LENGTH    = 2.0

LEGEND_KW = dict(loc="outside upper center", ncol=len(SERIES), frameon=False,
                 handlelength=1.2, handletextpad=0.3, columnspacing=0.8,
                 borderpad=0.0, borderaxespad=0.0)
# hspace keeps a visible 4 pt gap between the two panels; touching
# panels read as one block.
LAYOUT_PADS = dict(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.08)

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


def main() -> None:
    _apply_rcparams()
    datas = [json.loads((DATA / f).read_text()) for f, _ in MODELS]
    drafts = datas[0]["drafts"]
    assert all(d["drafts"] == drafts for d in datas), "models must share m"
    vals = [t for d in datas for taus in d["tau"].values()
            for m, t in taus.items() if int(m) in drafts]
    y_lo = math.floor((min(vals) - Y_PAD_LOW) * 10) / 10
    y_hi = math.ceil((max(vals) + Y_PAD) * 10) / 10

    fig, axes = plt.subplots(len(MODELS), 1, sharex=True, sharey=True,
                             figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                             layout="constrained")
    fig.get_layout_engine().set(**LAYOUT_PADS)

    for ax, data, (_, name) in zip(axes, datas, MODELS):
        ax.plot(drafts, [m + 1 for m in drafts], ls="--", color=IDEAL_COLOR,
                zorder=1)
        for key, label, color, marker in SERIES:
            taus = data["tau"][key]
            xs = [m for m in drafts if str(m) in taus]
            ax.plot(xs, [taus[str(m)] for m in xs], marker=marker,
                    markersize=MARKER_SIZE, color=color, label=label, zorder=2)
        ax.text(0.985, 0.02, name, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=FONT_PANEL)
        ax.grid(True, which="major", linewidth=GRID_LINE_WIDTH,
                alpha=GRID_ALPHA)
        ax.tick_params(direction=TICK_DIRECTION, length=TICK_LENGTH,
                       which="both")

    axes[0].text(*IDEAL_TEXT_XY, "Ideal ($m{+}1$)", color=IDEAL_COLOR,
                 fontsize=FONT_IDEAL, ha="left", va="bottom",
                 rotation=IDEAL_ROTATION, rotation_mode="anchor",
                 transform_rotates_text=True)

    bottom = axes[-1]
    bottom.set_xticks(drafts)
    bottom.set_xlim(min(drafts) - 0.2, max(drafts) + 0.2)
    bottom.set_ylim(y_lo, y_hi)
    bottom.set_yticks(range(math.ceil(y_lo), math.floor(y_hi) + 1))
    bottom.set_xlabel("Draft budget $m$", labelpad=0)
    fig.supylabel(r"Acceptance length $\tau$", fontsize=FONT_LABEL)

    handles = [Line2D([], [], color=c, marker=mk, markersize=MARKER_SIZE,
                      lw=LINE_WIDTH, label=lb) for _, lb, c, mk in SERIES]
    fig.legend(handles=handles, **LEGEND_KW)

    # No tight bbox: the canvas is the printed size.
    out = HERE / "acceptance_vs_draft.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    fig.canvas.draw()
    print("panel heights (in):", [round(ax.get_position().height * FIG_HEIGHT_IN, 3) for ax in axes])


if __name__ == "__main__":
    main()
