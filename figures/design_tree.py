#!/usr/bin/env python3
"""Figure: the draft tree of one round and the stream it becomes (Design 4.2).

The example is the round of the timeline figure (design_timeline.py): a root,
five trunk tokens in the first segment (6 hidden states, one foreground batch),
and eight background tokens (two background batches of four). Node labels are
path probabilities. The verdict: the first trunk token (0.62) misses, its rank-2
sibling (0.24, itself in the first segment) is the target's token there, and
the continuation accepts one of the sibling's children (0.11).

    python3 design_tree.py --save   # writes design_tree.pdf / .png / .pptx
"""
import sys
from pathlib import Path
from diagram import Canvas

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
W, H = 3.33, 1.21
FS = 7.5
SOLID, LIGHT_FILL, EDGE = "#C0392B", "#EFC4BF", "#C0392B"     # uplink colors of the timeline figure
TRUNK_LW, BRANCH_LW = 1.3, 0.7
ACCEPT = "#1F6FE0"

c = Canvas(W, H)

# ------------------------------------------------------------------ the tree
NW, NH = 0.36, 0.12
XC = lambda col: 0.02 + 0.44 * col
YR = [0.15, 0.315, 0.48, 0.645]
nodes = {  # name: (col, row, label, first_segment)
    "root": (0, 0, "Root", True),
    "t1": (1, 0, "0.62", True), "t2": (2, 0, "0.50", True), "t3": (3, 0, "0.41", True),
    "t4": (4, 0, "0.28", True), "t5": (5, 0, "0.19", False), "t6": (6, 0, "0.09", False),
    "b1": (1, 1, "0.24", True), "b2": (2, 1, "0.12", False), "b3": (3, 1, "0.08", False),
    "b4": (4, 1, "0.06", False),
    "b1a": (2, 2, "0.11", False), "b2a": (3, 2, "0.05", False),
    "b1b": (2, 3, "0.04", False),
}
edges = [("root", "t1"), ("t1", "t2"), ("t2", "t3"), ("t3", "t4"), ("t4", "t5"), ("t5", "t6"),
         ("root", "b1"), ("t1", "b2"), ("t2", "b3"), ("t3", "b4"),
         ("b1", "b1a"), ("b1", "b1b"), ("b2", "b2a")]
trunk = {("root", "t1"), ("t1", "t2"), ("t2", "t3"), ("t3", "t4"), ("t4", "t5"), ("t5", "t6")}
accepted = [("root", "b1"), ("b1", "b1a")]


def anchor(name, side):
    col, row, _, _ = nodes[name]
    x, y = XC(col), YR[row]
    return (x + NW, y + NH / 2) if side == "out" else (x, y + NH / 2)


c.group_begin()
c.box(XC(0) - 0.02, YR[0] - 0.025, XC(6) + NW - XC(0) + 0.04, NH + 0.05, fill="#D4D4D4", line="#D4D4D4",
      lw=0, radius=0.04)
for a, b in edges:
    (x1, y1), (x2, y2) = anchor(a, "out"), anchor(b, "in")
    c.line(x1, y1, x2, y2, color="#333333" if (a, b) in trunk else "#888888",
           lw=TRUNK_LW if (a, b) in trunk else BRANCH_LW)
for a, b in accepted:
    (x1, y1), (x2, y2) = anchor(a, "out"), anchor(b, "in")
    c.line(x1, y1, x2, y2, color=ACCEPT, lw=2.0, dash=True)
for name, (col, row, label, first) in nodes.items():
    c.box(XC(col), YR[row], NW, NH, label, fill=SOLID if first else LIGHT_FILL, line=EDGE, lw=0.6,
          fs=FS, color="#FFFFFF" if first else "#000000", radius=0.03, dash=not first)
c.group_end()
c.text(XC(2), 0.0, 1.5, 0.12, "Trunk", fs=FS, bold=True, color="#333333")
c.text(0.0, YR[2] - 0.02, 0.46, 0.12, "Branches", fs=FS, bold=True, color="#333333")
c.text(1.74, 0.455, 1.59, 0.11, "Verdict (dashed blue edges):", fs=FS, color=ACCEPT, align="left")
c.text(1.74, 0.565, 1.59, 0.11, "0.62 misses, the target picks", fs=FS, color=ACCEPT, align="left")
c.text(1.74, 0.675, 1.59, 0.11, "0.24, and 0.11 is accepted", fs=FS, color=ACCEPT, align="left")

# ---------------------------------------------------------------- the stream
order = ["root", "t1", "t2", "t3", "t4", "b1", "t5", "b2", "b1a", "t6", "b3", "b4", "b2a", "b1b"]
SW, SH, PITCH, SX0, SY = 0.215, 0.12, 0.234, 0.02, 0.925
SX = lambda i: SX0 + PITCH * i


c.group_begin()
for i, name in enumerate(order):
    _, _, label, first = nodes[name]
    c.box(SX(i), SY, SW, SH, label, fill=SOLID if first else LIGHT_FILL, line=EDGE, lw=0.6,
          fs=FS, color="#FFFFFF" if first else "#000000", radius=0.02, dash=not first)
c.group_end()
# rulers above and below the stream: one line each, the marker is the shared divider
YU, YL = SY - 0.02, SY + SH + 0.02
x0, x1 = SX(0), SX(13) + SW
xm = SX(6) - (PITCH - SW) / 2                     # first-segment marker, between the 6th and 7th box
xb = SX(10) - (PITCH - SW) / 2                    # boundary between the two background batches
for y, tick in ((YU, -0.025), (YL, 0.025)):
    c.line(x0, y, x1, y, color="#333333", lw=0.7)
    c.line(x0, y, x0, y + tick, color="#333333", lw=0.7)
    c.line(x1, y, x1, y + tick, color="#333333", lw=0.7)
c.line(xb, YL, xb, YL + 0.025, color="#333333", lw=0.7)
c.line(xm, YU, xm, YL, color="#000000", lw=1.6)
label = lambda i0, i1, y, t: c.text(SX(i0), y, SX(i1) + SW - SX(i0), 0.10, t, fs=FS, color="#333333")
label(0, 5, YU - 0.105, "First segment (6)")
label(6, 13, YU - 0.105, "Background tokens (8)")
label(0, 5, YL + 0.012, "Foreground batch")
label(6, 9, YL + 0.012, "Background batch")
label(10, 13, YL + 0.012, "Background batch")

if "--save" in sys.argv:
    c.save(str(HERE / "design_tree"))
else:
    c.save(str(HERE / "design_tree"))
