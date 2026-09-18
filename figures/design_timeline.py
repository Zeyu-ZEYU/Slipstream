#!/usr/bin/env python3
"""Figure: one verify round, one-shot tree versus draft streaming (Design 4.2).

Schematic: no time axis numbers are shown. Bar lengths are nevertheless laid
out with the Nemotron cost model fitted to Figure 7 (per hidden state / fixed):
drafting 2.95 / 5.6 ms, uplink 4.0 / 42.7, provider 4.33 / 28.7, downlink
1.47 / 40.7, verify 1.0 / 6.4; one-shot m=8 adds up to the measured 249 ms.
Streaming: a trunk token is drafted every 3 ms and its hidden state leaves in 4 ms;
later hidden states and later outputs on the same connection pay only the ~20 ms
one-way propagation, and the downlink serves foreground outputs first.

    python3 design_timeline.py     # writes design_timeline.pdf / .png / .pptx
"""
from pathlib import Path
from diagram import Canvas

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
W = 7.0
LEFT, RIGHT = 1.10, 6.94          # plot area x-range (inches)
T_MAX = 260.0                      # ms
X = lambda t: LEFT + (RIGHT - LEFT) * t / T_MAX
LANES = ["Drafter + encoder", "Uplink", "Provider middle", "Downlink", "Decoder verify"]   # as in the Figure 7 legend
COL = {"draft": "#1BAF7A", "up": "#C0392B", "prov": "#2A78D6", "down": "#EDA100", "ver": "#E377C2"}
LIGHT = {"draft": "#BFE8D7", "up": "#EFC4BF", "prov": "#BFD7F3", "down": "#FAE3B3", "ver": "#F6CDE9"}
FS = 7.5
LANE_H, LANE_GAP = 0.104, 0.010   # a 7.5 pt line is about 0.104 in tall
PANEL_TITLE_H = 0.0      # panel labels sit below the lanes
A0 = 0.01
A_END = A0 + 5 * (LANE_H + LANE_GAP)
SEP = A_END + 0.025                # dashed separator between the panels
B0 = SEP + 0.035
B_END = B0 + 5 * (LANE_H + LANE_GAP)
YA = B_END + 0.035                 # time arrow
H = YA + 0.04

c = Canvas(W, H)


def lane_y(panel_y0, i):
    return panel_y0 + PANEL_TITLE_H + i * (LANE_H + LANE_GAP)


def bar(panel_y0, lane, t0, t1, key, light=False, label=None, label_fs=None, label_color=None,
        label_inside=True):
    label_fs = label_fs or FS
    y = lane_y(panel_y0, lane)
    c.box(X(t0), y, X(t1) - X(t0), LANE_H, fill=LIGHT[key] if light else COL[key],
          line=COL[key], lw=0.6, radius=0.0, dash=light)
    if label:
        col = label_color or ("#000000" if light or key in ("down",) else "#FFFFFF")
        if label_inside:
            c.text(X(t0), y, X(t1) - X(t0), LANE_H, label, fs=label_fs, color=col)
        else:
            c.text(X(t1) + 0.03, y, 1.6, LANE_H, label, fs=label_fs, color=COL[key], align="left")


def panel(y0):
    for i, name in enumerate(LANES):
        c.text(0.0, lane_y(y0, i), LEFT - 0.05, LANE_H, name, fs=FS, align="right", color="#333333")


def panel_title(y0, text):
    """Panel name at the centroid of the empty bottom-left region (downlink and
    verify lanes, early in the round); the same x in both panels."""
    yc = (lane_y(y0, 3) + lane_y(y0, 4) + LANE_H) / 2
    c.text(X(62) - 1.0, yc - LANE_H / 2, 2.0, LANE_H, text, fs=8.0, bold=True)


# ---------------------------------------------------------------- panel (a)
panel(A0)
bar(A0, 0, 0, 32, "draft", label="Draft the whole tree", label_inside=False)
bar(A0, 1, 32, 111, "up", label="9 hidden states")
bar(A0, 2, 111, 179, "prov", label="All 9 at once")
bar(A0, 3, 179, 233, "down", label="9 outputs")
bar(A0, 4, 233, 249, "ver")

# ---------------------------------------------------------------- panel (b)
panel_title(A0, "One-shot tree")
c.line(0.0, SEP, RIGHT, SEP, color="#777777", lw=0.6, dash=True)
panel(B0)
panel_title(B0, "Draft streaming")
# drafting: root encode, trunk tokens one at a time, then branches best-first
bar(B0, 0, 0, 3, "draft")
for k in range(5):
    bar(B0, 0, 3 + 3.0 * k, 3 + 3.0 * (k + 1), "draft")
bar(B0, 0, 18, 62, "draft", light=True, label="Branches, best-first", label_color="#0B5B3E")
c.text(X(3), lane_y(B0, 0) - 0.005, X(18) - X(3), LANE_H, "Trunk", fs=FS, color="#FFFFFF")
# uplink: first-segment hidden states leave as drafted, branch hidden states follow
bar(B0, 1, 27, 102, "up", light=True)
bar(B0, 1, 3, 70, "up", label="First segment: 6 hidden states")
c.text(X(70), lane_y(B0, 1), X(102) - X(70), LANE_H, "Background", fs=FS, color="#7A1F17")
# provider: foreground step, then background steps while the GPU is idle
bar(B0, 2, 70, 125, "prov", label="Foreground: 6")
bar(B0, 2, 125, 171, "prov", light=True, label="Background: 4", label_color="#123E73")
bar(B0, 2, 171, 217, "prov", light=True, label="Background: 4", label_color="#123E73")
# downlink: outputs return as the middle produces them
bar(B0, 3, 125, 175, "down", label="6 outputs")
bar(B0, 3, 175, 201, "down", light=True)
# client: verify the first segment; continue on the accepted branch
bar(B0, 4, 175, 187, "ver")
bar(B0, 4, 201, 210, "ver", light=True)
c.text(X(210) + 0.03, lane_y(B0, 4), 1.2, LANE_H, "Continuation verified", fs=FS, color=COL["ver"], align="left")
# markers: stable round closes after the first verify; verdict (continue) goes up
c.line(X(187), lane_y(B0, 0) - 0.02, X(187), lane_y(B0, 4) + LANE_H + 0.03, color="#555555", lw=0.6, dash=True)
c.text(X(187) - 1.15, lane_y(B0, 0) - 0.005, 1.13, LANE_H, "Round would close here", fs=FS, color="#333333", align="right")
c.arrow(X(187), lane_y(B0, 4) + LANE_H / 2, X(207), lane_y(B0, 2) + LANE_H, color="#555555", lw=0.8, dash=True)
c.text(X(206), lane_y(B0, 3), 0.7, LANE_H, "Verdict: continue", fs=FS, color="#333333", align="left")

# time axis
c.arrow(LEFT, YA, RIGHT, YA, color="#333333", lw=0.7)
c.text(0.0, YA - 0.065, LEFT - 0.42, 0.13, "Time", fs=FS, color="#000000", bold=True, align="right")

c.save(str(HERE / "design_timeline"))
