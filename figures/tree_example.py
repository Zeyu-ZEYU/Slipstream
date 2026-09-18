#!/usr/bin/env python3
"""Figure: a dynamic draft tree (EAGLE-2 style) with depth 3, width 2, budget 6 (Background 2.2).

Numbers are path probabilities. At each level the two nodes with the highest
path probability are expanded (two children each); the rerank keeps the six
best nodes overall (solid) and drops the rest (gray). The target accepts the
path root -> 0.6 -> 0.42 -> 0.13 and adds its own token, the bonus token.

    python3 tree_example.py --save   # writes tree_example.pdf / .png / .pptx
"""
from pathlib import Path
from diagram import Canvas

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
W, H = 3.33, 0.77
FS = 7.5
KEPT, KEPT_LINE = "#FFFFFF", "#333333"        # kept by the rerank
DROP, DROP_LINE = "#F0F0F0", "#AAAAAA"        # drafted, dropped by the rerank
ACCEPT = "#1F6FE0"

c = Canvas(W, H)
NW, NH = 0.42, 0.12
XC = lambda col: 0.05 + 0.60 * col
YR = lambda row: 0.135 + 0.165 * row        # row may be fractional

# column headers
for col, name in enumerate(["", "Level 1", "Level 2", "Level 3"]):
    if name:
        c.text(XC(col), 0.025, NW, 0.11, name, fs=FS, bold=True, color="#333333")

nodes = {  # name: (col, row, label, kept)
    "root": (0, 1.5, "Root", True),
    "a": (1, 0.5, "0.6", True), "b": (1, 2.5, "0.3", True),
    "aa": (2, 0, "0.42", True), "ab": (2, 1, "0.12", False),
    "ba": (2, 2, "0.15", True), "bb": (2, 3, "0.08", False),
    "aaa": (3, 0, "0.25", True), "aab": (3, 1, "0.13", True),
    "baa": (3, 2, "0.09", False), "bab": (3, 3, "0.03", False),
}
edges = [("root", "a"), ("root", "b"), ("a", "aa"), ("a", "ab"), ("b", "ba"), ("b", "bb"),
         ("aa", "aaa"), ("aa", "aab"), ("ba", "baa"), ("ba", "bab")]
accepted = [("root", "a"), ("a", "aa"), ("aa", "aab")]


def anchor(name, side):
    col, row, _, _ = nodes[name]
    x, y = XC(col), YR(row)
    return (x + NW, y + NH / 2) if side == "out" else (x, y + NH / 2)


c.group_begin()
for a, b in edges:
    (x1, y1), (x2, y2) = anchor(a, "out"), anchor(b, "in")
    kept = nodes[b][3]
    c.line(x1, y1, x2, y2, color="#555555" if kept else "#AAAAAA", lw=0.8 if kept else 0.6,
           dash=not kept)
for a, b in accepted:
    (x1, y1), (x2, y2) = anchor(a, "out"), anchor(b, "in")
    c.line(x1, y1, x2, y2, color=ACCEPT, lw=2.0, dash=True)
# the bonus token after the accepted path
bx, by = W - 0.02 - 0.70, YR(1)
(x1, y1) = anchor("aab", "out")
c.line(x1, y1, bx, by + NH / 2, color=ACCEPT, lw=2.0, dash=True)
c.box(bx, by, 0.70, NH, "Bonus token", fill="#FFFFFF", line=ACCEPT, lw=1.0, fs=FS,
      color=ACCEPT, radius=0.03, dash=True)
for name, (col, row, label, kept) in nodes.items():
    c.box(XC(col), YR(row), NW, NH, label, fill=KEPT if kept else DROP,
          line=KEPT_LINE if kept else DROP_LINE, lw=0.8 if kept else 0.6, fs=FS,
          color="#000000" if kept else "#777777", radius=0.03, dash=not kept)
c.group_end()

c.save(str(HERE / "tree_example"))
