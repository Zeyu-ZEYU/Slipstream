"""Sensitivity to the window W: provider goodput against the background tokens the provider
holds per client, on Nemotron-3-Super + LiveCodeBench, one line per R (requests in flight at
the provider, R/4 per client). The star on each line marks W_DEFAULT, the window every other
experiment ran with; it is one of the swept values, so it needs no number of its own.

    python3 sensitivity_w.py           # opens a window
    python3 sensitivity_w.py --save    # writes sensitivity_w.pdf and sensitivity_w.png next to this file

DATA holds the plotted values themselves: for each R, the goodput at W = 0, 2, 4, 8, 16, 24, 32, in
that order. A point that was not run is None and the line skips it. Fill it with measured numbers
and set SOURCE to "measured"; while SOURCE is anything else the figure carries an ESTIMATE watermark.
`sensitivity_w_estimate.py --write` fills the block with estimates and refuses to touch a
measured block.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FONT_TICK, FONT_LABEL = 8.0, 9.0
FONT_SERIF = ["STIXGeneral", "DejaVu Serif"]
MODEL, WORKLOAD = "Nemotron-3-Super", "LiveCodeBench"
R_STYLE = [   # R, color (light to dark with load), marker
    (4,  "#8FB3DE", "o"),
    (8,  "#5B8FD0", "s"),
    (16, "#2A62A8", "^"),
    (32, "#0F3A6E", "D"),
]

SOURCE = "measured"
R = [4, 8, 16, 32]
W = [0, 2, 4, 8, 16, 24, 32]   # x positions; W = 0 is the first segment alone
W_DEFAULT = 24             # the window the other experiments ran with (the star); None draws no star

DATA = {
    # provider goodput, tokens/s over all four clients, at W = [0, 2, 4, 8, 16, 24, 32]; one row per R
    # (the W = 24 column is the default window, the Slipstream point of the overall figure)
    ('W',  4): [  53.5,   68.4,   75.5,   84.1,   91.0,   91.0,   92.4],
    ('W',  8): [ 111.6,  123.9,  130.7,  136.4,  142.4,  142.4,  143.6],
    ('W', 16): [ 145.2,  151.4,  153.1,  157.0,  160.0,  162.0,  162.2],
    ('W', 32): [ 159.5,  163.5,  164.9,  166.5,  167.5,  167.5,  167.5],
}


def _wpos(w):
    """x position of a window value: 0 at the left, then log2 spacing."""
    import math
    return 0.0 if w <= 1 else math.log2(w)


def build_figure():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": FONT_SERIF,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.labelsize": FONT_TICK,
                         "ytick.labelsize": FONT_TICK})
    fig, ax = plt.subplots(figsize=(3.335, 0.95), layout="constrained")   # 1.16 before the page cut
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.0)
    handles = []
    for r, color, marker in R_STYLE:
        ys = DATA.get(("W", r))
        if not ys:
            continue
        pts = [(w, y) for w, y in zip(W, ys) if y is not None]
        h, = ax.plot([_wpos(w) for w, _ in pts], [y for _, y in pts], color=color, marker=marker,
                     ms=3.6, lw=1.0, markeredgewidth=0.6, label=f"$R{{=}}{r}$")
        handles.append(h)
        if W_DEFAULT in W and ys[W.index(W_DEFAULT)] is not None:
            ax.plot([_wpos(W_DEFAULT)], [ys[W.index(W_DEFAULT)]], marker="*", ms=7.5, color=color,
                    markeredgecolor="black", markeredgewidth=0.5, ls="none", zorder=5)
    from matplotlib.lines import Line2D
    if W_DEFAULT is not None:
        handles.append(Line2D([], [], marker="*", ms=7.5, color="white", markeredgecolor="black",
                              markeredgewidth=0.5, ls="none", label="Default"))
    ax.set_xticks([_wpos(w) for w in W]); ax.set_xticklabels([str(w) for w in W])
    ax.set_xlabel("Window $W$ (background tokens per client)", fontsize=FONT_LABEL, labelpad=0)
    ax.set_ylabel("Goodput\n(tokens/s)", fontsize=FONT_LABEL, labelpad=2, ha="center",
                  multialignment="center")
    ax.set_ylim(50, 175); ax.set_yticks([50, 100, 150])   # three labels fit the shorter panel
    ax.grid(True, lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.5, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(handles=handles, loc="outside upper center", ncol=5, fontsize=FONT_TICK,
               frameon=False, handlelength=1.3, handletextpad=0.35, columnspacing=0.7,
               borderaxespad=0.0, borderpad=0.0)
    if SOURCE != "measured":
        fig.text(0.5, 0.45, "ESTIMATE", fontsize=26, color="#BBBBBB", alpha=0.35,
                 ha="center", va="center", rotation=15, zorder=0)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.save:
        import matplotlib; matplotlib.use("Agg")
        fig = build_figure()
        out = HERE / "sensitivity_w.pdf"
        fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300); print("wrote", out)
    else:
        import matplotlib.pyplot as plt
        build_figure(); plt.show()


if __name__ == "__main__":
    main()
