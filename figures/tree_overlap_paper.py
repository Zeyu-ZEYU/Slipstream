#!/usr/bin/env python3
"""Where the tree's budget goes, and where acceptance lands (paper figure).

Rank-aligned overlays of every verified (6,2), m=8 MATH draft round, one
row per model (Nemotron-3-Super on top, GLM-4.5-Air below). Left: the trees
sent for verification. Right: the paths the target accepted. Within a
panel, columns are tree levels and rows rank a level's nodes by cumulative
path probability. Node area is the share of rounds that use a slot, line
width grows with the rounds on an edge, and the row-end percentages give
each rank's share of the column's tokens: of all drafted tokens on the
left, of all accepted tokens on the right. Only rank 1 carries a strong
color.

    python3 tree_overlap_paper.py           # preview window, writes nothing
    python3 tree_overlap_paper.py --save    # write pdf + png next to this file

Data: the verified-round dumps of Section 3.4. Both columns count verified rounds
only. Nemotron's tree dump also holds each request's final propose, which
was never verified, so those lines are skipped. Slot ranking reuses
tree_overlap.py.
"""

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))
from tree_overlap import selected_slots  # noqa: E402

DUMPS = HERE / "data" / "tree_dumps"
MODELS = [  # (row name, dump directory, tree/verify lines pair 1:1), top to bottom
    ("Nemotron-3-Super", DUMPS / "nemotron", False),
    ("GLM-4.5-Air",      DUMPS / "glm", True),
]
TREE_DUMP = "tree_dump_math_n6k2_m8.jsonl"
VERIFY_DUMP = "verify_dump_math_n6k2_m8.jsonl"
COLUMNS = ["Drafted tokens", "Accepted tokens"]
SHARE_LABEL = "Share of tokens"    # of the column's tokens: drafted or accepted
LEVELS = list(range(1, 7))   # the root sits at x = 0, labeled in each panel

# Layout in inches. FIG_W is the USENIX column width, so fonts print at size.
# Top to bottom: column titles, then per model a name line and a row of
# panels, then the level labels. Left to right: "Rank", two panels, the
# share label.
FIG_W = 3.335
FIG_H = 1.670             # panels 0.45 in (0.52 in at the 1.81 in height before the page cut)
LEFT, COL_GAP, RIGHT = 0.29, 0.07, 0.16
TITLES, HEADER, ROW_GAP, BOTTOM = 0.19, 0.13, 0.02, 0.30
PANEL_H = (FIG_H - TITLES - len(MODELS) * HEADER
           - (len(MODELS) - 1) * ROW_GAP - BOTTOM) / len(MODELS)
# Root at x = 0 and levels 1..6, share labels past 6. The room left of the
# root and above rank 1 holds each panel's "Root" label.
X_LIM = (-1.0, 8.6)
Y_LIM = (-3.35, 0.40)     # rank r sits at y = -r; the root square at y = 0 is the top
ROOT_DX = 7.0             # pt from the root square to its label's center

# STIX is Times-compatible and embeds as TrueType (no Type 3 fonts).
# Printed sizes: nothing below 8 pt, labels at 9 pt (the ASPLOS '27 rule).
FONT_SERIF = ["STIXGeneral", "DejaVu Serif"]
FONT_TICK, FONT_LABEL, FONT_TITLE, FONT_MODEL = 8.0, 9.0, 9.0, 8.0

S_MAX = 66.0              # marker area (pt^2) of a slot used in every round; 78 at the taller panels
S_FLOOR = 4.0             # keeps one-off slots visible
LW_MIN, LW_MAX = 0.3, 2.4

C_RANK = ["#184F95", "#5598E7", "#BBBBBB", "#BBBBBB"]  # greedy, runner-up, rest
C_EDGE = "#CACACA"
C_ROOT = "#4A4A4A"
INK, MUTED = "#222222", "#555555"


def read_jsonl(name, dumps):
    """A dump's records, from the plain file if present, else the .gz."""
    plain = dumps / name
    if plain.exists():
        with open(plain) as f:
            return [json.loads(line) for line in f]
    with gzip.open(dumps / (name + ".gz"), "rt") as f:
        return [json.loads(line) for line in f]


def verified_pairs(dumps, paired):
    """(tree, verify) records, one pair per verified round. Nemotron's serial
    dump holds one unverified final propose per request, skipped here; GLM's
    lines pair one-to-one."""
    trees = read_jsonl(TREE_DUMP, dumps)
    vers = read_jsonl(VERIFY_DUMP, dumps)
    pairs, ti = [], 0
    for vi, ver in enumerate(vers):
        tree = trees[ti]
        # Join tripwire: the verified tree must be the proposed one.
        assert [tree["tok"][g] for g in tree["keep"]] == ver["tok"], vi
        pairs.append((tree, ver))
        ti += 1
        if (not paired and vi + 1 < len(vers)
                and vers[vi + 1]["rid"] != ver["rid"]):
            ti += 1
    return pairs


def overlays(pairs):
    """Slot overlays of the sent trees and of the accepted paths, each as
    (node counts, edge counts, rounds)."""
    root = (-1, 0)
    sent_n, sent_e, acc_n, acc_e = Counter(), Counter(), Counter(), Counter()
    for tree, ver in pairs:
        slots = selected_slots(tree)            # kept index -> (level, rank)
        kept = {g: j for j, g in enumerate(tree["keep"])}
        sent_n[root] += 1
        for j, g in enumerate(tree["keep"]):
            p = tree["par"][g]
            sent_n[slots[j]] += 1
            sent_e[(root if p < 0 else slots[kept[p]], slots[j])] += 1
        acc_n[root] += 1
        src = root
        for j in ver["path"]:
            acc_n[slots[j]] += 1
            acc_e[(src, slots[j])] += 1
            src = slots[j]
    return (sent_n, sent_e, len(pairs)), (acc_n, acc_e, len(pairs))


def rank_shares(nodes):
    by_rank = Counter()
    for (lvl, rank), c in nodes.items():
        if lvl >= 0:
            by_rank[rank] += c
    total = sum(by_rank.values())
    return {r: c / total for r, c in by_rank.items()}


def draw_panel(ax, nodes, edges, rounds, root_shift):
    pos = lambda slot: (slot[0] + 1, -slot[1])  # root (-1, 0) -> x = 0
    for (src, dst), c in sorted(edges.items(), key=lambda kv: kv[1]):
        (x0, y0), (x1, y1) = pos(src), pos(dst)
        greedy = src[1] == 0 and dst[1] == 0
        ax.plot([x0, x1], [y0, y1], color=C_RANK[0] if greedy else C_EDGE,
                alpha=0.55 if greedy else 1.0,
                lw=LW_MIN + (LW_MAX - LW_MIN) * c / rounds,
                solid_capstyle="round", zorder=1)
    for slot, c in nodes.items():
        if slot[0] < 0:
            continue
        x, y = pos(slot)
        ax.scatter([x], [y], s=max(S_FLOOR, S_MAX * c / rounds),
                   color=C_RANK[min(slot[1], 3)], edgecolors="white",
                   linewidths=0.6, zorder=2)
    ax.scatter([0], [0], s=22, marker="s", color=C_ROOT, linewidths=0,
               zorder=3)
    # The root's name stands just left of its square.
    ax.text(0, -0.3, "Root", rotation=90, ha="center", va="top",
            fontsize=FONT_TICK, color=MUTED, transform=ax.transData + root_shift)
    for r, share in sorted(rank_shares(nodes).items()):
        ax.text(6.55, -r, f"{share * 100:.1f}%", ha="left", va="center",
                fontsize=FONT_TICK, color=INK)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0, pad=1.5, labelsize=FONT_TICK, labelcolor=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure():
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    root_shift = ScaledTranslation(-ROOT_DX / 72, 0, fig.dpi_scale_trans)
    pw = (FIG_W - LEFT - COL_GAP - RIGHT) / 2
    panels_mid_x = (LEFT + (FIG_W - LEFT - RIGHT) / 2) / FIG_W
    for row, (name, dumps, paired) in enumerate(MODELS):
        pairs = verified_pairs(dumps, paired)
        print(f"{name}: {len(pairs)} verified rounds")
        y0 = BOTTOM + (len(MODELS) - 1 - row) * (HEADER + PANEL_H + ROW_GAP)
        for col, (nodes, edges, rounds) in enumerate(overlays(pairs)):
            x0 = LEFT + col * (pw + COL_GAP)
            ax = fig.add_axes([x0 / FIG_W, y0 / FIG_H, pw / FIG_W,
                               PANEL_H / FIG_H])
            draw_panel(ax, nodes, edges, rounds, root_shift)
            if row == len(MODELS) - 1:
                ax.set_xticks(LEVELS, [str(lvl) for lvl in LEVELS])
            if col == 0:
                ax.set_yticks([0, -1, -2, -3], ["1", "2", "3", "4"])
        # The model name heads its row, centered across both panels, 1 pt above them.
        fig.text(panels_mid_x, (y0 + PANEL_H + 1 / 72) / FIG_H, name,
                 ha="center", va="bottom", fontsize=FONT_MODEL, color=INK)

    titles_y = (FIG_H - TITLES + 2 / 72) / FIG_H   # 2 pt above the name line
    for col, title in enumerate(COLUMNS):
        fig.text((LEFT + col * (pw + COL_GAP) + pw / 2) / FIG_W, titles_y,
                 title, ha="center", va="baseline", fontsize=FONT_TITLE,
                 fontweight="bold")
    rows_span = len(MODELS) * PANEL_H + (len(MODELS) - 1) * (HEADER + ROW_GAP)
    rows_mid_y = (BOTTOM + rows_span / 2) / FIG_H
    fig.text(0.004, rows_mid_y, "Rank", rotation=90, ha="left", va="center",
             fontsize=FONT_LABEL)
    fig.text(1 - 0.004, rows_mid_y, SHARE_LABEL, rotation=90, ha="right",
             va="center", fontsize=FONT_LABEL)
    fig.text(panels_mid_x, 0.012, "Tree level", ha="center", va="bottom",
             fontsize=FONT_LABEL)
    return fig


def main() -> None:
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
        fig = build_figure()
        out = HERE / "tree_overlap_paper.pdf"
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"), dpi=300)
        print("wrote", out)
    else:
        import matplotlib.pyplot as plt
        build_figure()
        plt.show()


if __name__ == "__main__":
    main()
