"""Ablation: goodput lost when one mechanism is removed from Slipstream,
against the load R (requests in flight at the provider, R/4 per client);
one line per variant, one panel per model and workload present in DATA.

    python3 ablation.py          # opens the figure in a window (needs only matplotlib)
    python3 ablation.py --save   # writes ablation.pdf / .png next to this file

DATA holds exactly what is drawn: one entry per (model, workload, variant)
with four numbers, the goodput lost relative to the full system in
percent, 100 * (1 - variant / Slipstream), at R = 4, 8, 16, 32.
Panels follow the models and workloads
that appear in DATA, in the order of MODELS and WORKLOADS; drop a workload
from DATA to drop its panel. While SOURCE is "estimate", the figure carries
an ESTIMATE watermark.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FONT_TICK, FONT_LABEL = 8.0, 9.0
FONT_SERIF = ["STIXGeneral", "DejaVu Serif"]
MODELS = ["Nemotron-3-Super", "GLM-4.5-Air"]
ROW_LABELS = {"Nemotron-3-Super": "Nemotron-\n3-Super", "GLM-4.5-Air": "GLM-4.5-\nAir"}   # two lines each: a name is longer than a panel is tall
WORKLOADS = ["ShareGPT", "LiveCodeBench", "MATH", "LongBench"]
VARIANTS = [  # (key in DATA = legend label, color, marker)
    ("First segment only", "#C0392B", "D"),
    ("Level order",        "#EDA100", "v"),
    ("No continuation",    "#CC79A7", "s"),
    ("Always eager",       "#3358C8", "^"),
    ("Always lazy",        "#56B4E9", "o"),
    ("Admit all",          "#17A05A", "P"),
    ("No priorities",      "#555555", "X"),
]

# ####################################################################
# DATA  (goodput lost vs. Slipstream, in percent, at R = 4, 8, 16, 32; exactly what is drawn)
# ####################################################################
# BEGIN DATA
SOURCE = "measured"   # the user's measurements (09-17); the estimator refuses to overwrite this block
R = [4, 8, 16, 32]   # requests in flight at the provider; each of the four clients keeps R/4
# Order follows the figure: top row (Nemotron) left to right, then the bottom row (GLM).
# Each entry: (model, workload, variant): [loss at R=4, R=8, R=16, R=32]
#   loss = goodput lost relative to full Slipstream, in percent: 100 * (1 - variant / Slipstream).
#   Use None for a point that was not measured.
DATA = {
    ('Nemotron-3-Super', 'ShareGPT', 'First segment only' ): [ 36.9,  17.0,   6.7,   0.0],
    ('Nemotron-3-Super', 'ShareGPT', 'Level order'        ): [  8.5,   9.5,   8.9,   8.3],
    ('Nemotron-3-Super', 'ShareGPT', 'No continuation'    ): [ 38.4,  21.6,  10.8,   4.5],
    ('Nemotron-3-Super', 'ShareGPT', 'Always eager'       ): [  0.0,  15.9,  21.5,  24.1],
    ('Nemotron-3-Super', 'ShareGPT', 'Always lazy'        ): [ 11.6,   4.5,   2.5,   0.8],
    ('Nemotron-3-Super', 'ShareGPT', 'Admit all'          ): [  0.0,   0.0,   5.3,   8.3],
    ('Nemotron-3-Super', 'ShareGPT', 'No priorities'      ): [  2.0,   3.2,   4.2,   4.5],
    ('Nemotron-3-Super', 'LiveCodeBench', 'First segment only' ): [ 41.2,  21.6,  10.4,   4.8],
    ('Nemotron-3-Super', 'LiveCodeBench', 'Level order'        ): [  9.8,   8.0,   8.1,   8.5],
    ('Nemotron-3-Super', 'LiveCodeBench', 'No continuation'    ): [ 37.4,  21.8,   6.5,   1.7],
    ('Nemotron-3-Super', 'LiveCodeBench', 'Always eager'       ): [  0.0,  14.3,  24.1,  25.8],
    ('Nemotron-3-Super', 'LiveCodeBench', 'Always lazy'        ): [ 10.8,   5.8,   1.9,   0.9],
    ('Nemotron-3-Super', 'LiveCodeBench', 'Admit all'          ): [  0.0,   1.1,   4.4,  10.7],
    ('Nemotron-3-Super', 'LiveCodeBench', 'No priorities'      ): [  2.0,   2.9,   4.3,   5.2],
    ('Nemotron-3-Super', 'MATH', 'First segment only' ): [ 38.5,  12.2,   4.7,   0.0],
    ('Nemotron-3-Super', 'MATH', 'Level order'        ): [  8.8,  11.8,  10.9,  10.9],
    ('Nemotron-3-Super', 'MATH', 'No continuation'    ): [ 41.0,  15.8,   5.9,   2.0],
    ('Nemotron-3-Super', 'MATH', 'Always eager'       ): [  0.0,  17.9,  21.4,  23.8],
    ('Nemotron-3-Super', 'MATH', 'Always lazy'        ): [ 11.2,   3.8,   1.8,   1.0],
    ('Nemotron-3-Super', 'MATH', 'Admit all'          ): [  0.0,   0.7,   2.6,   9.1],
    ('Nemotron-3-Super', 'MATH', 'No priorities'      ): [  2.0,   2.8,   4.2,   4.9],
    ('Nemotron-3-Super', 'LongBench', 'First segment only' ): [ 34.9,  16.2,   5.3,   1.2],
    ('Nemotron-3-Super', 'LongBench', 'Level order'        ): [  9.8,   9.8,  10.4,  11.8],
    ('Nemotron-3-Super', 'LongBench', 'No continuation'    ): [ 37.6,  17.2,   7.1,   3.3],
    ('Nemotron-3-Super', 'LongBench', 'Always eager'       ): [  0.0,  19.2,  24.7,  25.9],
    ('Nemotron-3-Super', 'LongBench', 'Always lazy'        ): [ 12.8,   3.9,   1.9,   1.3],
    ('Nemotron-3-Super', 'LongBench', 'Admit all'          ): [  0.0,   0.3,   5.3,  11.4],
    ('Nemotron-3-Super', 'LongBench', 'No priorities'      ): [  1.9,   3.2,   3.8,   5.0],
    ('GLM-4.5-Air', 'ShareGPT', 'First segment only' ): [ 37.5,  37.6,  20.1,   1.5],
    ('GLM-4.5-Air', 'ShareGPT', 'Level order'        ): [  9.1,   9.0,   8.1,   7.5],
    ('GLM-4.5-Air', 'ShareGPT', 'No continuation'    ): [ 36.9,  39.5,  24.2,   6.6],
    ('GLM-4.5-Air', 'ShareGPT', 'Always eager'       ): [  0.0,   0.0,   2.2,   3.4],
    ('GLM-4.5-Air', 'ShareGPT', 'Always lazy'        ): [ 11.4,  13.1,   6.3,   1.5],
    ('GLM-4.5-Air', 'ShareGPT', 'Admit all'          ): [  0.0,   0.0,   4.1,   8.6],
    ('GLM-4.5-Air', 'ShareGPT', 'No priorities'      ): [  2.0,   5.8,  12.8,  13.7],
    ('GLM-4.5-Air', 'LiveCodeBench', 'First segment only' ): [ 38.0,  30.4,  21.4,   2.5],
    ('GLM-4.5-Air', 'LiveCodeBench', 'Level order'        ): [  8.4,   9.6,   9.0,   9.1],
    ('GLM-4.5-Air', 'LiveCodeBench', 'No continuation'    ): [ 39.2,  34.8,  23.1,   6.2],
    ('GLM-4.5-Air', 'LiveCodeBench', 'Always eager'       ): [  0.0,   0.7,   1.9,   3.4],
    ('GLM-4.5-Air', 'LiveCodeBench', 'Always lazy'        ): [ 12.2,   9.5,   6.8,   1.5],
    ('GLM-4.5-Air', 'LiveCodeBench', 'Admit all'          ): [  0.0,   1.3,   4.0,   7.6],
    ('GLM-4.5-Air', 'LiveCodeBench', 'No priorities'      ): [  2.1,   5.7,  14.4,  13.6],
    ('GLM-4.5-Air', 'MATH', 'First segment only' ): [ 33.0,  28.3,  19.9,   2.3],
    ('GLM-4.5-Air', 'MATH', 'Level order'        ): [  9.0,   8.1,   9.0,   8.7],
    ('GLM-4.5-Air', 'MATH', 'No continuation'    ): [ 39.7,  35.5,  22.6,   4.9],
    ('GLM-4.5-Air', 'MATH', 'Always eager'       ): [  0.0,   0.7,   1.8,   3.2],
    ('GLM-4.5-Air', 'MATH', 'Always lazy'        ): [ 13.1,   9.6,   7.0,   1.7],
    ('GLM-4.5-Air', 'MATH', 'Admit all'          ): [  0.0,   0.5,   4.3,   7.4],
    ('GLM-4.5-Air', 'MATH', 'No priorities'      ): [  1.9,   6.7,  13.6,  15.9],
    ('GLM-4.5-Air', 'LongBench', 'First segment only' ): [ 28.1,  32.5,  21.6,   2.3],
    ('GLM-4.5-Air', 'LongBench', 'Level order'        ): [  9.5,   9.2,   7.9,   7.9],
    ('GLM-4.5-Air', 'LongBench', 'No continuation'    ): [ 29.6,  30.9,  22.6,   5.6],
    ('GLM-4.5-Air', 'LongBench', 'Always eager'       ): [  0.0,   0.0,   1.2,   3.2],
    ('GLM-4.5-Air', 'LongBench', 'Always lazy'        ): [ 11.4,  12.1,   8.2,   2.3],
    ('GLM-4.5-Air', 'LongBench', 'Admit all'          ): [  0.0,   0.4,   4.1,   8.8],
    ('GLM-4.5-Air', 'LongBench', 'No priorities'      ): [  2.1,   6.1,  13.1,  13.6],
}
# END DATA
# ####################################################################


def build_figure():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": FONT_SERIF,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.labelsize": FONT_TICK,
                         "ytick.labelsize": FONT_TICK})
    models = [m for m in MODELS if any(k[0] == m for k in DATA)]
    wls = [w for w in WORKLOADS if any(k[1] == w for k in DATA)]
    ncol, nrow = len(wls), len(models)
    fig_w = 7.0 if ncol > 1 else 3.335
    panel_h = 0.45 if ncol > 1 else 1.35   # shorter than a rotated row label, so the labels wrap
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, nrow * panel_h + 0.78), squeeze=False,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.0, hspace=0.04, wspace=0.03)
    handles = []
    for i, model in enumerate(models):
        for j, wl in enumerate(wls):
            ax = axes[i][j]
            for name, color, marker in VARIANTS:
                v = DATA.get((model, wl, name))
                if not v:
                    continue
                xs, ys = [], []
                for r, loss in zip(R, v):
                    if loss is None:
                        continue
                    xs.append(r); ys.append(loss)
                h, = ax.plot(xs, ys, color=color, marker=marker, ms=4.0, lw=1.0, label=name,
                             markeredgewidth=0.6)
                if i == 0 and j == 0:
                    handles.append(h)
            ax.set_xscale("log", base=2); ax.set_xticks(R); ax.set_xticklabels([str(r) for r in R])
            ax.minorticks_off()
            ax.set_ylim(-2, 45); ax.set_yticks([0, 20, 40])   # three labels fit a 0.45 in panel
            ax.grid(True, lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)
            ax.tick_params(length=2.5, width=0.5, pad=1.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0 and ncol > 1:
                ax.set_title(wl, fontsize=FONT_LABEL, pad=2)
            if j == 0:
                ax.set_ylabel(ROW_LABELS.get(model, model) if ncol > 1 else f"{model}, {wl}", fontsize=FONT_TICK, labelpad=2)
            if i < nrow - 1:
                ax.set_xticklabels([])
    fig.supxlabel("Requests in flight at the provider, $R$", fontsize=FONT_LABEL)
    fig.supylabel("Goodput loss\n(% of Slipstream)", fontsize=FONT_LABEL,
                  multialignment="center")   # two lines, centered on each other; one line is longer than the figure is tall
    fig.legend(handles=handles, loc="outside upper center", ncol=7 if ncol > 1 else 4,
               fontsize=FONT_LABEL, frameon=False, handlelength=1.3, handletextpad=0.3,
               columnspacing=0.7, borderaxespad=0.0, borderpad=0.3)
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
        out = HERE / "ablation.pdf"
        fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300); print("wrote", out)
    else:
        import matplotlib.pyplot as plt
        build_figure(); plt.show()


if __name__ == "__main__":
    main()
