#!/usr/bin/env python3
"""Rank-aligned overlap diagrams of draft trees from an MTP_TREE_DUMP file.

Each dump line is one draft round: the full (n,k) candidate pool
(``par``/``score``, level derivable) plus the indices the top-m rerank kept
(``keep``). Two figure kinds per dataset:

  A "structure": the beam-survivor tree the (n,k) construction grew
    (per level, the k candidates that survived to expand the next level;
    the last level's survivors are its top-k by score).
  B "selected": the final top-m tree that was sent for verification.

Layout: one row per level (root on top), nodes within a
level ordered left-to-right by cumulative path probability (rank), trees
overlaid by (level, rank) slot. Edge darkness/width scale with how many
rounds contain that slot edge; counts are printed on the edges.

Usage:
  python tree_overlap.py DUMP.jsonl --kind selected --tag math_n6k2_m8
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

FONT_FAMILY = "serif"
NODE_MAX_PT = 320.0   # marker area of the most frequent slot
EDGE_MAX_LW = 4.0
LABEL_MIN_FRAC = 0.0  # label every edge; raise to hide rare-edge labels
CMAP = "inferno_r"


def levels_from_parents(par):
    lvl = [0] * len(par)
    for i, p in enumerate(par):
        lvl[i] = 0 if p < 0 else lvl[p] + 1
    return lvl


def survivor_nodes(par, score, lvl):
    """Beam-survivor set: nodes with children, plus the last level's top-k.

    Every beam node expands exactly k children, so 'has a child' equals
    'survived the beam' on all levels but the last; the last level's
    survivors are its k best by score (k inferred from level-0 size).
    """
    k = sum(1 for p in par if p < 0)
    has_child = [False] * len(par)
    for i, p in enumerate(par):
        if p >= 0:
            has_child[p] = True
    last = max(lvl)
    last_nodes = [i for i in range(len(par)) if lvl[i] == last]
    last_keep = set(sorted(last_nodes, key=lambda i: -score[i])[:k])
    return [i for i in range(len(par)) if has_child[i] or i in last_keep]


def selected_slots(rec):
    """(level, rank) slot per m-node index of the kept tree."""
    par, score, keep = rec["par"], rec["score"], rec["keep"]
    lvl = levels_from_parents(par)
    by_level = {}
    for g in sorted(keep, key=lambda g: -score[g]):
        by_level.setdefault(lvl[g], []).append(g)
    rank = {}
    for level, members in by_level.items():
        for r, g in enumerate(members):
            rank[g] = r
    return {j: (lvl[g], rank[g]) for j, g in enumerate(keep)}


def accumulate_accepted(tree_path, verify_path, paired=False):
    """Overlay of the accepted root paths, in the selected tree's slots.

    Serial-run positional join: per rid, the request's final propose is
    never verified, so advance the tree cursor once more on rid change.
    paired=True is for dumps where every tree line was verified (GLM on
    SGLang): tree and verify lines then pair one-to-one.
    """
    trees = [json.loads(l) for l in open(tree_path)]
    vers = [json.loads(l) for l in open(verify_path)]
    node_cnt = Counter()
    edge_cnt = Counter()
    ti = 0
    for vi, v in enumerate(vers):
        slots = selected_slots(trees[ti])
        root = (-1, 0)
        node_cnt[root] += 1
        src = root
        for j in v["path"]:
            dst = slots[j]
            node_cnt[dst] += 1
            edge_cnt[(src, dst)] += 1
            src = dst
        ti += 1
        if (not paired and vi + 1 < len(vers)
                and vers[vi + 1]["rid"] != v["rid"]):
            ti += 1
    return node_cnt, edge_cnt, len(vers)


def accumulate(path, kind):
    node_cnt = Counter()   # (level, rank) -> rounds containing that slot
    edge_cnt = Counter()   # ((lvl_p, rank_p), (lvl_c, rank_c)) -> rounds
    rounds = 0
    for line in open(path):
        rec = json.loads(line)
        par, score = rec["par"], rec["score"]
        lvl = levels_from_parents(par)
        if kind == "selected":
            nodes = rec["keep"]
        elif kind == "pool":
            nodes = list(range(len(par)))
        else:
            nodes = survivor_nodes(par, score, lvl)
        nodes = set(nodes)
        # Rank within each level by cumulative path probability, descending.
        by_level = {}
        for i in sorted(nodes, key=lambda i: -score[i]):
            by_level.setdefault(lvl[i], []).append(i)
        rank = {}
        for level, members in by_level.items():
            for r, i in enumerate(members):
                rank[i] = r
        root = (-1, 0)  # committed token, one row above level 0
        node_cnt[root] += 1
        for i in nodes:
            slot = (lvl[i], rank[i])
            node_cnt[slot] += 1
            p = par[i]
            src = root if p < 0 else (lvl[p], rank[p])
            if p >= 0 and p not in nodes:
                # Selected trees are root-connected, so this cannot happen;
                # guard stays as a data-integrity tripwire.
                raise AssertionError("parent outside node set")
            edge_cnt[(src, (lvl[i], rank[i]))] += 1
        rounds += 1
    return node_cnt, edge_cnt, rounds


def draw(node_cnt, edge_cnt, rounds, kind, tag, out_pdf):
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 9})
    levels = sorted({l for (l, _) in node_cnt})
    width = max(r for (_, r) in node_cnt) + 1
    fig_w = max(4.2, 0.95 * width + 1.2)
    fig_h = max(3.2, 0.72 * len(levels) + 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    norm = LogNorm(vmin=1, vmax=max(edge_cnt.values()))
    cmap = plt.get_cmap(CMAP)
    xy = lambda slot: (slot[1], -slot[0])

    for (src, dst), c in sorted(edge_cnt.items(), key=lambda kv: kv[1]):
        (x0, y0), (x1, y1) = xy(src), xy(dst)
        color = cmap(norm(c))
        lw = 0.6 + (EDGE_MAX_LW - 0.6) * (c / max(edge_cnt.values()))
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=1,
                solid_capstyle="round")
        if c / rounds >= LABEL_MIN_FRAC:
            mx, my = (x0 + 2 * x1) / 3, (y0 + 2 * y1) / 3
            ax.text(mx, my, str(c), fontsize=6.5, color="#1a1a1a",
                    ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.75))
    max_node = max(node_cnt.values())
    for slot, c in node_cnt.items():
        x, y = xy(slot)
        ax.scatter([x], [y], s=40 + NODE_MAX_PT * c / max_node,
                   color=cmap(norm(max(c, 1))), edgecolors="black",
                   linewidths=0.5, zorder=2)
    ax.scatter(*xy((-1, 0)), s=120, marker="s", color="#444444",
               edgecolors="black", zorder=2)
    ax.text(xy((-1, 0))[0] - 0.35, xy((-1, 0))[1], "root",
            ha="right", va="center", fontsize=8)

    for l in levels:
        if l >= 0:
            ax.text(-0.75, -l, f"level {l + 1}", ha="right", va="center",
                    fontsize=8, color="#333333")
    ax.set_xlim(-1.6, width - 0.4)
    ax.set_ylim(-max(levels) - 0.6, 1.6)
    ax.axis("off")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("rounds containing the edge", fontsize=8)
    kind_name = {"selected": "final top-$m$ tree",
                 "structure": "beam-survivor structure",
                 "pool": "grown candidate pool",
                 "accepted": "accepted paths"}[kind]
    ax.set_title(f"{tag}: {kind_name}, {rounds} draft rounds overlaid",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=170, bbox_inches="tight",
                pad_inches=0.03)
    print(f"wrote {out_pdf} ({rounds} rounds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--kind",
                    choices=["selected", "structure", "pool", "accepted"],
                    default="selected")
    ap.add_argument("--verify", help="verify dump (required for accepted)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.kind == "accepted":
        node_cnt, edge_cnt, rounds = accumulate_accepted(
            args.dump, args.verify)
    else:
        node_cnt, edge_cnt, rounds = accumulate(args.dump, args.kind)
    out = Path(__file__).parent / f"tree_overlap_{args.tag}_{args.kind}.pdf"
    draw(node_cnt, edge_cnt, rounds, args.kind, args.tag, out)


if __name__ == "__main__":
    main()
