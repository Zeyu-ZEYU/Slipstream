"""Overall performance at scale: 8 providers (one node of H20 GPUs, one model replica each) behind a
router that sends each request to the provider with the fewest requests in flight, and 24 clients
(three nodes of H20 GPUs), each on its own emulated residential link. Same form as overall_perf.py:
TPOT P50 against goodput over all 24 clients, one curve per system, the five points at K = 1, 2,
4, 8, 16 requests in flight per client (R = 24, 48, 96, 192, 384 at the providers, 3K each).

    python3 overall_scale.py           # opens a window
    python3 overall_scale.py --save    # writes overall_scale.pdf / .png next to this file

DATA holds the plotted points: per (model, workload, system) the five (goodput, TPOT) pairs in
figure reading order. Fill it with measured numbers and set SOURCE = "measured"; while SOURCE is
anything else the figure carries an ESTIMATE watermark. `scale_estimate.py --write` fills the block
from the measured overall figure through the closed-loop model in its docstring.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG_W = 7.0                # full text width
PANEL_W, PANEL_H = 1.42, 0.48   # a panel is shorter than its rotated row label, so the label wraps
FONT_TICK, FONT_LABEL = 8.0, 9.0
FONT_SERIF = ["STIXGeneral", "DejaVu Serif"]

# (label, color, marker, linestyle); the palette is the paper's validated set
SYSTEMS_DRAWN = [   # the OS- arms ship the draft in one piece and pick its budget online
    ("Baseline",    "#555555", "o", (0, (3, 2))),
    ("Cunningham",  "#EDA100", "v", (0, (3, 2))),
    ("OS-Chain",    "#17A05A", "s", "-"),    # drawn as "One-shot chain"
    ("OS-Tree",     "#3358C8", "^", "-"),    # drawn as "One-shot tree"
    ("Slipstream",  "#C0392B", "D", "-"),
]
LABELS = {"OS-Chain": "One-shot chain", "OS-Tree": "One-shot tree"}   # legend names for the DATA keys
MODELS = ["Nemotron-3-Super", "GLM-4.5-Air"]
ROW_LABELS = {"Nemotron-3-Super": "Nemotron-\n3-Super", "GLM-4.5-Air": "GLM-4.5-\nAir"}   # two lines each: a name is longer than a panel is tall
WORKLOADS = ["ShareGPT", "LiveCodeBench", "MATH", "LongBench"]

# ####################################################################
# DATA  (provider goodput in tokens/s over all clients; TPOT P50 in ms)
# ####################################################################
# BEGIN DATA
SOURCE = "measured"   # set to "measured" once the numbers below are measurements (drops the watermark)
R = [24, 48, 96, 192, 384]   # requests in flight over the 8 providers at the five points of every curve; each of the 24 clients keeps R/24
# Order follows the figure: top row (Nemotron) left to right, then the bottom row (GLM).
# Each entry: (model, workload, system): [(x, y) at R=24, 48, 96, 192, 384], x = goodput, y = TPOT P50
#   x = goodput, delivered tokens/s summed over the four clients;  y = TPOT P50, ms per delivered token
#   Use None for a point that was not measured.
DATA = {
    # Nemotron-3-Super, ShareGPT: (goodput over 24 clients, TPOT P50) at K = 1, 2, 4, 8, 16 per client
    ('Nemotron-3-Super', 'ShareGPT', 'Baseline'   ): [(  193.5,  124.0), (  386.9,  124.1), (  741.3,  129.5), (  864.0,  222.2), (  865.0,  443.9)],
    ('Nemotron-3-Super', 'ShareGPT', 'Cunningham' ): [(  215.2,  111.5), (  430.0,  111.6), (  742.9,  129.2), (  777.7,  246.9), (  777.9,  493.7)],
    ('Nemotron-3-Super', 'ShareGPT', 'OS-Chain'   ): [(  331.0,   72.5), (  656.5,   73.1), (  917.3,  104.7), (  923.5,  207.9), (  923.5,  415.8)],
    ('Nemotron-3-Super', 'ShareGPT', 'OS-Tree'    ): [(  358.8,   66.9), (  708.8,   67.7), (  948.4,  101.2), (  952.8,  201.5), (  952.8,  403.0)],
    ('Nemotron-3-Super', 'ShareGPT', 'Slipstream' ): [(  530.4,   45.3), (  884.7,   54.3), ( 1034.9,   92.8), ( 1036.1,  185.3), ( 1036.1,  370.6)],

    # Nemotron-3-Super, LiveCodeBench: (goodput over 24 clients, TPOT P50) at K = 1, 2, 4, 8, 16 per client
    ('Nemotron-3-Super', 'LiveCodeBench', 'Baseline'   ): [(  193.5,  124.0), (  386.9,  124.1), (  741.3,  129.5), (  864.0,  222.2), (  865.0,  443.9)],
    ('Nemotron-3-Super', 'LiveCodeBench', 'Cunningham' ): [(  219.3,  109.4), (  436.8,  109.9), (  657.2,  146.1), (  665.8,  288.4), (  665.9,  576.7)],
    ('Nemotron-3-Super', 'LiveCodeBench', 'OS-Chain'   ): [(  357.8,   67.1), (  708.5,   67.8), (  969.9,   99.0), (  975.5,  196.8), (  975.5,  393.7)],
    ('Nemotron-3-Super', 'LiveCodeBench', 'OS-Tree'    ): [(  395.9,   60.6), (  781.8,   61.4), ( 1040.6,   92.3), ( 1045.2,  183.7), ( 1045.3,  367.4)],
    ('Nemotron-3-Super', 'LiveCodeBench', 'Slipstream' ): [(  588.1,   40.8), (  949.4,   50.6), ( 1061.6,   90.4), ( 1062.8,  180.7), ( 1062.8,  361.3)],

    # Nemotron-3-Super, MATH: (goodput over 24 clients, TPOT P50) at K = 1, 2, 4, 8, 16 per client
    ('Nemotron-3-Super', 'MATH', 'Baseline'   ): [(  193.5,  124.0), (  386.9,  124.1), (  741.3,  129.5), (  864.0,  222.2), (  865.0,  443.9)],
    ('Nemotron-3-Super', 'MATH', 'Cunningham' ): [(  242.2,   99.1), (  481.1,   99.8), (  690.7,  139.0), (  696.7,  275.6), (  696.7,  551.1)],
    ('Nemotron-3-Super', 'MATH', 'OS-Chain'   ): [(  368.3,   65.2), (  731.0,   65.7), ( 1031.9,   93.0), ( 1039.6,  184.7), ( 1039.7,  369.4)],
    ('Nemotron-3-Super', 'MATH', 'OS-Tree'    ): [(  409.7,   58.6), (  808.2,   59.4), ( 1067.6,   89.9), ( 1072.0,  179.1), ( 1072.0,  358.2)],
    ('Nemotron-3-Super', 'MATH', 'Slipstream' ): [(  680.0,   35.3), ( 1157.2,   41.5), ( 1262.3,   76.0), ( 1262.3,  152.1), ( 1262.3,  304.2)],

    # Nemotron-3-Super, LongBench: (goodput over 24 clients, TPOT P50) at K = 1, 2, 4, 8, 16 per client
    ('Nemotron-3-Super', 'LongBench', 'Baseline'   ): [(  193.5,  124.0), (  386.9,  124.1), (  741.3,  129.5), (  864.0,  222.2), (  865.0,  443.9)],
    ('Nemotron-3-Super', 'LongBench', 'Cunningham' ): [(  293.2,   81.9), (  577.3,   83.1), (  751.9,  127.7), (  754.6,  254.4), (  754.6,  508.9)],
    ('Nemotron-3-Super', 'LongBench', 'OS-Chain'   ): [(  426.9,   56.2), (  845.3,   56.8), ( 1157.3,   83.0), ( 1163.8,  165.0), ( 1163.9,  329.9)],
    ('Nemotron-3-Super', 'LongBench', 'OS-Tree'    ): [(  455.5,   52.7), (  902.2,   53.2), ( 1240.2,   77.4), ( 1247.6,  153.9), ( 1247.6,  307.8)],
    ('Nemotron-3-Super', 'LongBench', 'Slipstream' ): [(  676.2,   35.5), ( 1135.5,   42.3), ( 1272.4,   75.4), ( 1272.1,  150.9), ( 1272.0,  301.9)],

    # GLM-4.5-Air, ShareGPT: (goodput over 24 clients, TPOT P50) at K = 1, 2, 4, 8, 16 per client
    ('GLM-4.5-Air', 'ShareGPT', 'Baseline'   ): [(  232.5,  103.2), (  465.0,  103.2), (  929.1,  103.3), ( 1609.7,  119.3), ( 1687.2,  227.6)],
    ('GLM-4.5-Air', 'ShareGPT', 'Cunningham' ): [(  258.8,   92.7), (  517.6,   92.7), ( 1031.1,   93.1), ( 1572.2,  122.1), ( 1595.3,  240.7)],
    ('GLM-4.5-Air', 'ShareGPT', 'OS-Chain'   ): [(  399.6,   60.1), (  799.3,   60.1), ( 1593.8,   60.2), ( 2499.9,   76.8), ( 2547.2,  150.8)],
    ('GLM-4.5-Air', 'ShareGPT', 'OS-Tree'    ): [(  441.6,   54.4), (  883.1,   54.4), ( 1759.9,   54.5), ( 2707.6,   70.9), ( 2750.7,  139.6)],
    ('GLM-4.5-Air', 'ShareGPT', 'Slipstream' ): [(  609.4,   39.4), ( 1175.9,   40.8), ( 2018.2,   47.6), ( 2892.3,   66.4), ( 2962.8,  129.6)],

    # GLM-4.5-Air, LiveCodeBench: (goodput over 24 clients, TPOT P50) at K = 1, 2, 4, 8, 16 per client
    ('GLM-4.5-Air', 'LiveCodeBench', 'Baseline'   ): [(  232.5,  103.2), (  465.0,  103.2), (  929.1,  103.3), ( 1609.7,  119.3), ( 1687.2,  227.6)],
    ('GLM-4.5-Air', 'LiveCodeBench', 'Cunningham' ): [(  268.3,   89.4), (  536.7,   89.4), ( 1068.7,   89.8), ( 1610.1,  119.2), ( 1631.4,  235.4)],
    ('GLM-4.5-Air', 'LiveCodeBench', 'OS-Chain'   ): [(  461.7,   52.0), (  923.4,   52.0), ( 1841.8,   52.1), ( 2911.7,   65.9), ( 2970.8,  129.3)],
    ('GLM-4.5-Air', 'LiveCodeBench', 'OS-Tree'    ): [(  506.4,   47.4), ( 1012.7,   47.4), ( 2014.7,   47.7), ( 2961.8,   64.8), ( 2993.3,  128.3)],
    ('GLM-4.5-Air', 'LiveCodeBench', 'Slipstream' ): [(  712.2,   33.7), ( 1431.5,   33.5), ( 2860.2,   33.6), ( 3242.6,   59.2), ( 3243.7,  118.4)],

    # GLM-4.5-Air, MATH: (goodput over 24 clients, TPOT P50) at K = 1, 2, 4, 8, 16 per client
    ('GLM-4.5-Air', 'MATH', 'Baseline'   ): [(  232.5,  103.2), (  465.0,  103.2), (  929.1,  103.3), ( 1609.7,  119.3), ( 1687.2,  227.6)],
    ('GLM-4.5-Air', 'MATH', 'Cunningham' ): [(  290.4,   82.6), (  580.8,   82.6), ( 1156.4,   83.0), ( 1737.2,  110.5), ( 1759.6,  218.2)],
    ('GLM-4.5-Air', 'MATH', 'OS-Chain'   ): [(  489.6,   49.0), (  979.1,   49.0), ( 1950.6,   49.2), ( 2973.5,   64.6), ( 3017.1,  127.3)],
    ('GLM-4.5-Air', 'MATH', 'OS-Tree'    ): [(  557.0,   43.1), ( 1113.9,   43.1), ( 2215.6,   43.3), ( 3249.2,   59.1), ( 3283.0,  117.0)],
    ('GLM-4.5-Air', 'MATH', 'Slipstream' ): [(  795.1,   30.2), ( 1595.6,   30.1), ( 3164.9,   30.3), ( 3563.8,   53.9), ( 3565.3,  107.7)],

    # GLM-4.5-Air, LongBench: (goodput over 24 clients, TPOT P50) at K = 1, 2, 4, 8, 16 per client
    ('GLM-4.5-Air', 'LongBench', 'Baseline'   ): [(  232.5,  103.2), (  465.0,  103.2), (  929.1,  103.3), ( 1609.7,  119.3), ( 1687.2,  227.6)],
    ('GLM-4.5-Air', 'LongBench', 'Cunningham' ): [(  272.2,   88.2), (  544.4,   88.2), ( 1083.3,   88.6), ( 1604.0,  119.7), ( 1622.2,  236.7)],
    ('GLM-4.5-Air', 'LongBench', 'OS-Chain'   ): [(  456.0,   52.6), (  912.0,   52.6), ( 1817.9,   52.8), ( 2818.3,   68.1), ( 2866.3,  134.0)],
    ('GLM-4.5-Air', 'LongBench', 'OS-Tree'    ): [(  499.0,   48.1), (  998.1,   48.1), ( 1985.6,   48.3), ( 2923.1,   65.7), ( 2954.6,  130.0)],
    ('GLM-4.5-Air', 'LongBench', 'Slipstream' ): [(  705.8,   34.0), ( 1348.2,   35.6), ( 2361.0,   40.7), ( 3154.0,   60.9), ( 3179.3,  120.8)],
}
# END DATA
# ####################################################################


def build_figure():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": FONT_SERIF,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.labelsize": FONT_TICK,
                         "ytick.labelsize": FONT_TICK})
    fig, axes = plt.subplots(len(MODELS), len(WORKLOADS),
                             figsize=(FIG_W, len(MODELS) * PANEL_H + 0.66),
                             layout="constrained", sharey="row")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.0, hspace=0.04, wspace=0.03)
    handles = []
    for i, model in enumerate(MODELS):
        for j, wl in enumerate(WORKLOADS):
            ax = axes[i][j]
            for name, color, marker, ls in SYSTEMS_DRAWN:
                key = (model, wl, name)
                if key not in DATA:
                    continue
                pts = [pt for pt in DATA[key] if pt is not None and pt[0] is not None and pt[1] is not None]
                if not pts:
                    continue
                h, = ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker,
                             ms=4.0, lw=1.0, ls=ls, label=LABELS.get(name, name), markeredgewidth=0.7,
                             markerfacecolor="white" if name in ("Baseline", "Cunningham") else color)
                if i == 0 and j == 0:
                    handles.append(h)
            ax.grid(True, lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)
            ax.tick_params(length=2.5, width=0.5, pad=1.5)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4)); ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(wl, fontsize=FONT_LABEL, pad=2)
            if j == 0:
                ax.set_ylabel(ROW_LABELS.get(model, model), fontsize=FONT_TICK, labelpad=2,
                              multialignment="center")
        # shared limits per model row; the y cap excludes off-scale points, which are clipped
        ys = [pt[1] for wl in WORKLOADS for name, *_ in SYSTEMS_DRAWN if name != "Cunningham"
              for pt in DATA.get((model, wl, name), []) if pt is not None and pt[1] is not None]
        xs = [pt[0] for wl in WORKLOADS for name, *_ in SYSTEMS_DRAWN
              for pt in DATA.get((model, wl, name), []) if pt is not None and pt[0] is not None]
        if ys and xs:
            for ax in axes[i]:
                ax.set_ylim(0, 1.15 * max(ys))
                ax.set_xlim(0, 1.05 * max(xs))
    fig.supxlabel("Goodput (output tokens/s); the points of a curve are $K$ = 1, 2, 4, 8, 16 per client from left to right", fontsize=FONT_LABEL)
    fig.supylabel("P50 TPOT (ms)", fontsize=FONT_LABEL)
    fig.legend(handles=handles, loc="outside upper center", ncol=5, fontsize=FONT_LABEL,
               frameon=False, handlelength=2.0, columnspacing=1.2, borderaxespad=0.0,
               borderpad=0.3, handletextpad=0.5)
    if SOURCE != "measured":
        fig.text(0.5, 0.5, "ESTIMATE", fontsize=40, color="#BBBBBB", alpha=0.35,
                 ha="center", va="center", rotation=20, zorder=0)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.save:
        import matplotlib; matplotlib.use("Agg")
        fig = build_figure()
        out = HERE / "overall_scale.pdf"
        fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300); print("wrote", out)
    else:
        import matplotlib.pyplot as plt
        build_figure(); plt.show()


if __name__ == "__main__":
    main()
