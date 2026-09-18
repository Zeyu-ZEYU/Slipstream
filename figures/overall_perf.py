"""Overall performance: TPOT versus provider goodput, one curve per system,
points at the provider's concurrency R = 4, 8, 16, 32 requests in flight (R/4 per client); one panel per model
and workload (Design §6, the only overall-performance figure).

    python3 overall_perf.py          # opens the figure in a window (needs only matplotlib)
    python3 overall_perf.py --save   # writes overall_perf.pdf / .png next to this file

To fill in measurements: edit the DATA block below (one entry per model,
workload, and system; four (goodput, TPOT) pairs, one per R), then set
SOURCE = "measured". Nothing else needs to change.

The numbers live in the DATA block, in the figure's reading order (top row
left to right, then the bottom row): one entry per (model, workload,
system) holding four (x, y) points, one per R, where x is the provider
goodput (delivered tokens per second over all four clients) and y is the
TPOT P50 (ms per delivered token).
SOURCE says where they come from. While SOURCE is "estimate", the values
are produced by overall_perf_estimate.py from the cost model of §4.1 and
the §3 acceptance data, and the figure carries an ESTIMATE watermark;
replace the block with measurements and set SOURCE = "measured".
The one-shot arms choose their budget online; their fixed-budget
versions belong to the sensitivity figure.
The DATA block now holds the user's measurements (09-17); the estimator
refuses to overwrite it while SOURCE is "measured".
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
R = [4, 8, 16, 32]   # requests in flight at the provider at the four points of every curve; each of the four clients keeps R/4
# Order follows the figure: top row (Nemotron) left to right, then the bottom row (GLM).
# Each entry: (model, workload, system): [(x, y) at R=4, (x, y) at R=8, (x, y) at R=16, (x, y) at R=32]
#   x = goodput, delivered tokens/s summed over the four clients;  y = TPOT P50, ms per delivered token
#   Use None for a point that was not measured.
DATA = {
    # ---- Nemotron-3-Super / ShareGPT ----
    ('Nemotron-3-Super', 'ShareGPT', 'Baseline'    ): [(  30.8,  129.8), (  61.2,  130.7), ( 107.4,  149.0), ( 129.7,  246.8)],
    ('Nemotron-3-Super', 'ShareGPT', 'Cunningham'  ): [(  32.4,  123.5), (  63.8,  125.5), ( 103.7,  154.3), ( 116.6,  274.3)],
    ('Nemotron-3-Super', 'ShareGPT', 'OS-Chain'    ): [(  36.8,   77.9), ( 94.6,   89.7), ( 124.7,  124.1), ( 145.7,  231.0)],
    ('Nemotron-3-Super', 'ShareGPT', 'OS-Tree'     ): [(  48,   72.7), ( 102.3,   79.0), ( 132.4,  122.4), ( 149.4,  223.9)],
    ('Nemotron-3-Super', 'ShareGPT', 'Slipstream'  ): [(  81.7,   48.9), ( 130.0,   61.5), ( 150.4,  106.4), ( 155.4,  205.9)],
    # ---- Nemotron-3-Super / LiveCodeBench ----
    ('Nemotron-3-Super', 'LiveCodeBench', 'Baseline'    ): [(  30.8,  129.8), (  61.2,  130.7), ( 107.4,  149.0), ( 129.7,  246.8)],
    ('Nemotron-3-Super', 'LiveCodeBench', 'Cunningham'  ): [(  33.2,  120.4), (  64.2,  124.7), (  93.8,  170.6), (  99.9,  320.4)],
    ('Nemotron-3-Super', 'LiveCodeBench', 'OS-Chain'    ): [(  49.1,   72.5), ( 101.8,   82.1), ( 137.5,   117.1), ( 156.0,  218.7)],
    ('Nemotron-3-Super', 'LiveCodeBench', 'OS-Tree'     ): [(  53.3,   66.4), ( 108.2,   73.9), ( 146.0,   110.8), ( 159.5,  204.1)],
    ('Nemotron-3-Super', 'LiveCodeBench', 'Slipstream'  ): [(  91.0,   43.9), ( 142.4,   56.2), ( 162.0,   98.8), ( 167.5,  191.1)],
    # ---- Nemotron-3-Super / MATH ----
    ('Nemotron-3-Super', 'MATH', 'Baseline'    ): [(  30.8,  129.8), (  61.2,  130.7), ( 107.4,  149.0), ( 129.7,  246.8)],
    ('Nemotron-3-Super', 'MATH', 'Cunningham'  ): [(  36.7,  108.8), (  70.3,  113.7), (  99.3,  161.2), ( 104.5,  306.2)],
    ('Nemotron-3-Super', 'MATH', 'OS-Chain'    ): [(  51.2,   69.8), ( 123.9,   77.2), ( 160.0,   104.4), ( 177.2,  205.2)],
    ('Nemotron-3-Super', 'MATH', 'OS-Tree'     ): [(  62.1,   63.5), ( 132.5,   69.1), ( 165.2,   99.3), ( 180,  199.0)],
    ('Nemotron-3-Super', 'MATH', 'Slipstream'  ): [( 105.0,   38.1), ( 162.5,   49.2), ( 183.2,   87.3), ( 189.4,  169.0)],
    # ---- Nemotron-3-Super / LongBench ----
    ('Nemotron-3-Super', 'LongBench', 'Baseline'    ): [(  30.8,  129.8), (  61.2,  130.7), ( 107.4,  149.0), ( 129.7,  246.8)],
    ('Nemotron-3-Super', 'LongBench', 'Cunningham'  ): [(  35.7,   89.4), (  83.5,   95.8), ( 109.5,  146.1), ( 113.2,  282.7)],
    ('Nemotron-3-Super', 'LongBench', 'OS-Chain'    ): [(  54.1,   61.2), ( 127.4,   69.2), ( 161.3,   103.4), ( 177.6,  183.3)],
    ('Nemotron-3-Super', 'LongBench', 'OS-Tree'     ): [(  60.9,   57.5), ( 131.3,   61.4), ( 165.8,   93.7), ( 180,  171)],
    ('Nemotron-3-Super', 'LongBench', 'Slipstream'  ): [( 104.2,   38.4), ( 163.5,   48.9), ( 186.8,   85.7), ( 193.1,  165.7)],
    # ---- GLM-4.5-Air / ShareGPT ----
    ('GLM-4.5-Air', 'ShareGPT', 'Baseline'    ): [(  35.4,  112.9), (  70.9,  112.9), ( 140.0,  114.3), ( 242.6,  131.9)],
    ('GLM-4.5-Air', 'ShareGPT', 'Cunningham'  ): [(  37.1,  108.0), (  74.1,  108.0), ( 145.9,  109.7), ( 246.2,  130.0)],
    ('GLM-4.5-Air', 'ShareGPT', 'OS-Chain'    ): [(  52.1,   66.4), ( 115.0,   66.5), ( 234.8,   68.1), ( 383.1,   85.2)],
    ('GLM-4.5-Air', 'ShareGPT', 'OS-Tree'     ): [(  55.9,   60.7), ( 111.7,   60.8), ( 246.7,   62.3), ( 401.5,   78.7)],
    ('GLM-4.5-Air', 'ShareGPT', 'Slipstream'  ): [(  90.6,   44.1), ( 181.1,   44.2), ( 320.3,   50.0), ( 432.4,   74.0)],
    # ---- GLM-4.5-Air / LiveCodeBench ----
    ('GLM-4.5-Air', 'LiveCodeBench', 'Baseline'    ): [(  35.4,  112.9), (  70.9,  112.9), ( 140.0,  114.3), ( 242.6,  131.9)],
    ('GLM-4.5-Air', 'LiveCodeBench', 'Cunningham'  ): [(  38.8,  103.1), (  77.5,  103.2), ( 151.7,  105.4), ( 245.9,  130.1)],
    ('GLM-4.5-Air', 'LiveCodeBench', 'OS-Chain'    ): [(  53.3,   58.3), ( 132.4,   58.4), ( 270.6,   58.8), ( 439.8,   72.8)],
    ('GLM-4.5-Air', 'LiveCodeBench', 'OS-Tree'     ): [(  64.1,   54.0), ( 137.9,   54.1), ( 277.2,   55.7), ( 449.8,   70.4)],
    ('GLM-4.5-Air', 'LiveCodeBench', 'Slipstream'  ): [( 105.8,   37.8), ( 211.5,   37.8), ( 369.8,   43.3), ( 488.4,   65.5)],
    # ---- GLM-4.5-Air / MATH ----
    ('GLM-4.5-Air', 'MATH', 'Baseline'    ): [(  35.4,  112.9), (  70.9,  112.9), ( 140.0,  114.3), ( 242.6,  131.9)],
    ('GLM-4.5-Air', 'MATH', 'Cunningham'  ): [(  42.1,   95.1), (  84.1,   95.1), ( 164.4,   97.3), ( 264.7,  120.9)],
    ('GLM-4.5-Air', 'MATH', 'OS-Chain'    ): [(  63.5,   54.7), ( 145.8,   55), ( 298.6,   55.7), ( 486.5,   71.5)],
    ('GLM-4.5-Air', 'MATH', 'OS-Tree'     ): [(  71.4,   49.1), ( 152.6,   49.2), ( 304.7,   50.8), ( 493.3,   64.2)],
    ('GLM-4.5-Air', 'MATH', 'Slipstream'  ): [( 116.9,   34.2), ( 233.5,   34.3), ( 407.5,   39.3), ( 536.1,   59.7)],
    # ---- GLM-4.5-Air / LongBench ----
    ('GLM-4.5-Air', 'LongBench', 'Baseline'    ): [(  35.4,  112.9), (  70.9,  112.9), ( 140.0,  114.3), ( 242.6,  131.9)],
    ('GLM-4.5-Air', 'LongBench', 'Cunningham'  ): [(  39.6,  101.1), (  79.1,  101.2), ( 154.1,  103.8), ( 243.2,  131.6)],
    ('GLM-4.5-Air', 'LongBench', 'OS-Chain'    ): [(  47.8,   59), ( 133.4,   59.8), ( 269.6,   61.2), ( 438.6,   75.3)],
    ('GLM-4.5-Air', 'LongBench', 'OS-Tree'     ): [(  69.9,   54.8), ( 135.6,   54.9), ( 273.5,   56.4), ( 443.8,   71.3)],
    ('GLM-4.5-Air', 'LongBench', 'Slipstream'  ): [( 103.1,   38.8), ( 206.0,   38.8), ( 361.5,   44.3), ( 480.6,   66.6)],
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
    fig.supxlabel("Goodput (output tokens/s); the points of a curve are $R$ = 4, 8, 16, 32 from left to right", fontsize=FONT_LABEL)
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
        out = HERE / "overall_perf.pdf"
        fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300); print("wrote", out)
    else:
        import matplotlib.pyplot as plt
        build_figure(); plt.show()


if __name__ == "__main__":
    main()
