#!/usr/bin/env python3
"""Tau vs draft budget m, tree (6,2) (solid) against the chain (dashed), one
panel per model: Nemotron-3-Super on top, GLM-4.5-Air below.

Data: tree_budget_tau.json (Nemotron) and tree_budget_tau_glm.json (GLM,
written by analysis/collect.py), both next to this file. Output:
tree_budget_tau.pdf, exactly one column wide so every font prints at its
set size. Same layout as acceptance_vs_draft.py and tree_shape_tau.py; the
top legend adds a second row for the two line styles.
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
    ("tree_budget_tau.json",     "Nemotron-3-Super"),
    ("tree_budget_tau_glm.json", "GLM-4.5-Air"),
]

FIG_WIDTH_IN  = 3.335   # one USENIX column, so fonts print at their set size
FIG_HEIGHT_IN = 1.520   # solved so each plot area = its y span x IN_PER_TAU

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
MARKER_SIZE     = 3.0

SERIES = [  # key, label, color, marker -- same encoding in every figure
    ("math",          "MATH",          "#27ae60", "^"),
    ("longbench",     "LongBench",  "#DAA520", "d"),
    ("sharegpt",      "ShareGPT",      "#c0392b", "o"),
    ("livecodebench", "LiveCodeBench", "#1f3a93", "s"),
]
CHAIN_ALPHA = 0.55       # chain: dashed, lighter, hollow markers
STYLE_COLOR = "#444444"  # line-style legend entries

# Each panel's y range hugs its own model's data (the models top out about
# 0.6 apart); both keep IN_PER_TAU inches per tau unit, so the heights
# follow the spans and a tau gap looks the same in both.
IN_PER_TAU  = 0.1650 # 0.205 before the page cut; tick labels then need 1.0 tau steps
Y_PAD_IN    = 0.05   # in added below/above the data, then rounded to 0.1 tau
Y_PAD       = Y_PAD_IN / IN_PER_TAU
TOP_EXTRA   = 0.25   # tau; room above the highest line for the model name at the top left
Y_TICK_STEP = 1.0
Y_TICK_EDGE = 0.06   # in; no tick label closer than this to a panel edge

# The model name goes top left: in GLM the bottom right holds the lowest
# (ShareGPT chain) line, while the top left is empty in both panels.
PANEL_LABEL_XY = (0.015, 0.95)

TICK_DIRECTION = "in"
TICK_LENGTH    = 2.0

LEGEND_KW = dict(loc="outside upper center", ncol=len(SERIES), frameon=False,
                 handlelength=1.3, handletextpad=0.3, columnspacing=0.8,
                 labelspacing=0.15, borderpad=0.0, borderaxespad=0.0)
LAYOUT_PADS = dict(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.07)

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


def _curves(data, key, ms):
    return ([data["tree"][key][str(m)] for m in ms],
            [data["chain"][key][str(m)] for m in ms])


def _y_range(data, ms):
    vals = [v for key, *_ in SERIES for c in _curves(data, key, ms) for v in c]
    return (math.floor((min(vals) - Y_PAD) * 10) / 10,
            math.ceil((max(vals) + Y_PAD + TOP_EXTRA) * 10) / 10)


def _y_ticks(lo, hi):
    edge = Y_TICK_EDGE / IN_PER_TAU
    k0 = math.ceil(lo / Y_TICK_STEP)
    ticks = [k * Y_TICK_STEP for k in range(k0, int(hi / Y_TICK_STEP) + 1)]
    return [t for t in ticks if lo + edge <= t <= hi - edge]


def _add_legends(fig):
    """Row 1: the workloads. Row 2: the two line styles, centered under row 1.
    A frame around row 2 would need about 4 pt more height than the reserved
    row (it touched row 1 and the panel), so there is none.

    The layout legend keeps row 2 as invisible stand-ins with the same text,
    so constrained layout reserves the same room as a plain two-row legend.
    The framed style legend is then drawn over that row, outside the layout.
    Legend columns fill top to bottom, hence the interleaving.
    """
    ds = [Line2D([], [], color=c, marker=mk, markersize=MARKER_SIZE,
                 lw=LINE_WIDTH, label=lb) for _, lb, c, mk in SERIES]
    styles = [Line2D([], [], color=STYLE_COLOR, lw=LINE_WIDTH,
                     label="Tree (6,2)"),
              Line2D([], [], color=STYLE_COLOR, lw=LINE_WIDTH, ls="--",
                     alpha=CHAIN_ALPHA, label="Chain")]
    ghosts = [Line2D([], [], alpha=0, label=h.get_label()) for h in styles]
    blank = Line2D([], [], alpha=0, label="")
    handles = [ds[0], blank, ds[1], ghosts[0], ds[2], ghosts[1], ds[3], blank]
    layout_leg = fig.legend(handles=handles,
                            labels=[h.get_label() for h in handles],
                            **LEGEND_KW)
    names = {h.get_label() for h in styles}
    ghost_texts = [t for t in layout_leg.get_texts() if t.get_text() in names]
    for t in ghost_texts:
        t.set_alpha(0)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    box = layout_leg.get_window_extent(r)
    row2 = ghost_texts[0].get_window_extent(r)
    x_c = to_fig.transform(((box.x0 + box.x1) / 2, 0))[0]
    y_c = to_fig.transform((0, (row2.y0 + row2.y1) / 2))[1]
    style_leg = fig.legend(
        handles=styles, loc="center", bbox_to_anchor=(x_c, y_c),
        bbox_transform=fig.transFigure, ncol=2, frameon=False,
        handlelength=LEGEND_KW["handlelength"],
        handletextpad=LEGEND_KW["handletextpad"],
        columnspacing=LEGEND_KW["columnspacing"])
    style_leg.set_in_layout(False)


def main() -> None:
    _apply_rcparams()
    datas = [json.loads((DATA / f).read_text()) for f, _ in MODELS]
    ms = datas[0]["budgets"]
    assert all(d["budgets"] == ms for d in datas), "models must share m"
    ranges = [_y_range(d, ms) for d in datas]

    fig, axes = plt.subplots(
        len(MODELS), 1, sharex=True, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        layout="constrained",
        gridspec_kw={"height_ratios": [hi - lo for lo, hi in ranges]})
    fig.get_layout_engine().set(**LAYOUT_PADS)

    for ax, data, (lo, hi), (_, name) in zip(axes, datas, ranges, MODELS):
        for key, label, color, marker in SERIES:
            tree, chain = _curves(data, key, ms)
            ax.plot(ms, tree, color=color, marker=marker, ms=MARKER_SIZE,
                    lw=LINE_WIDTH, label=label)
            ax.plot(ms, chain, color=color, marker=marker, ms=MARKER_SIZE,
                    lw=LINE_WIDTH, ls="--", alpha=CHAIN_ALPHA,
                    markerfacecolor="none")
        ax.set_ylim(lo, hi)
        ax.set_yticks(_y_ticks(lo, hi))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.text(*PANEL_LABEL_XY, name, transform=ax.transAxes, ha="left",
                va="top", fontsize=FONT_PANEL)
        ax.grid(True, axis="y", lw=GRID_LINE_WIDTH, alpha=GRID_ALPHA)
        ax.tick_params(direction=TICK_DIRECTION, length=TICK_LENGTH,
                       width=TICK_LINE_WIDTH)

    bottom = axes[-1]
    bottom.set_xticks(ms)
    bottom.set_xlabel(r"Draft budget $m$ ($m{+}1$ hidden states per round)", labelpad=0)
    fig.supylabel(r"Acceptance length $\tau$", fontsize=FONT_LABEL)

    _add_legends(fig)

    # No tight bbox: the canvas is the printed size.
    out = HERE / "tree_budget_tau.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    fig.canvas.draw()
    print("panel heights (in):", [round(ax.get_position().height * FIG_HEIGHT_IN, 3) for ax in axes])


if __name__ == "__main__":
    main()
