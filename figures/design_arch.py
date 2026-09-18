#!/usr/bin/env python3
"""Figure: Slipstream components and the three kinds of traffic (Design overview).

Since 2026-09-15 the figure is hand-maintained in design_arch.pptx (the user's
edited layout, exported to design_arch.pdf). This script only documents the
original geometry; it refuses to overwrite the files unless --regenerate is given.

    python3 design_arch.py --regenerate   # rewrites design_arch.pdf / .png / .pptx
"""
import sys
if "--regenerate" not in sys.argv:
    sys.exit("design_arch.* are hand-maintained; pass --regenerate to overwrite them")
from pathlib import Path
from diagram import Canvas

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
W, H = 3.335, 2.05
# palette follows Figure 1: green client, red provider, blue components, orange middle
CL_FILL, CL_LINE = "#E8F5E9", "#2E7D32"
PR_FILL, PR_LINE = "#FDECEA", "#C62828"
COMP_FILL, COMP_LINE = "#BBDEFB", "#1565C0"
MID_FILL, MID_LINE = "#FFE0B2", "#E65100"
CTL_FILL, CTL_LINE = "#ECEFF1", "#546E7A"
NET = "#C62828"
FS, FS_S = 8.0, 7.5

c = Canvas(W, H)
# zones; two offset outlines behind the client stand for the other clients
for k in (2, 1):
    c.box(0.0 + 0.05 * k, 0.17 + 0.03 * k, 1.40, 1.80, fill="#FFFFFF", line=CL_LINE, dash=True, radius=0.08, lw=0.6)
c.box(0.0, 0.17, 1.40, 1.80, fill=CL_FILL, line=CL_LINE, dash=True, radius=0.08)
c.box(1.935, 0.17, 1.40, 1.86, fill=PR_FILL, line=PR_LINE, dash=True, radius=0.08)
c.text(0.0, 0.0, 1.40, 0.17, "Client (trusted)", fs=FS, color=CL_LINE, bold=True)
c.text(1.935, 0.0, 1.40, 0.17, "Provider (untrusted)", fs=FS, color=PR_LINE, bold=True)

# client components
bx, bw = 0.30, 1.04
c.box(bx, 0.27, bw, 0.27, "Drafter (MTP)", fill=COMP_FILL, line=COMP_LINE, fs=FS)
c.box(bx, 0.63, bw, 0.27, "Encoder (first layer)", fill=COMP_FILL, line=COMP_LINE, fs=FS)
c.box(bx, 0.99, bw, 0.46, "Scheduler\norder, segment,\nadmission", fill=CTL_FILL, line=CTL_LINE, fs=FS)
c.box(bx, 1.54, bw, 0.42, "Decoder (LM head)\nverify", fill=COMP_FILL, line=COMP_LINE, fs=FS)
c.arrow(bx + bw / 2, 0.54, bx + bw / 2, 0.63, color="#333333", lw=0.8)
c.arrow(bx + bw / 2, 0.90, bx + bw / 2, 0.99, color="#333333", lw=0.8)
# accepted positions go to the scheduler; the accepted tokens start the next round
c.arrow(bx + bw - 0.16, 1.54, bx + bw - 0.16, 1.45, color="#333333", lw=0.8)
c.line(bx, 1.75, 0.17, 1.75, color="#333333", lw=0.8)
c.line(0.17, 1.75, 0.17, 0.405, color="#333333", lw=0.8)
c.arrow(0.17, 0.405, bx, 0.405, color="#333333", lw=0.8)
c.text(0.04, 1.20, 0.13, 0.60, "next round", fs=FS, color="#333333", rotation=90)

# provider components
px, pw = 2.06, 1.16
c.box(px, 0.27, pw, 0.44, "Foreground queue\nBackground queue", fill=CTL_FILL, line=CTL_LINE, fs=FS)
c.box(px, 0.81, pw, 0.46, "Middle\n(remaining layers)", fill=MID_FILL, line=MID_LINE, fs=FS)
c.box(px, 1.37, pw, 0.57, "Per-round state\npositions, KV, SSM, W", fill=CTL_FILL, line=CTL_LINE, fs=FS)
c.arrow(px + pw / 2, 0.71, px + pw / 2, 0.81, color="#333333", lw=0.8)
c.arrow(px + pw / 2, 1.27, px + pw / 2, 1.37, color="#333333", lw=0.8)

# the three kinds of traffic, one connection: hidden states and verdicts leave the scheduler,
# results reach the decoder
c.arrow(bx + bw, 1.04, px, 0.40, color=NET, lw=1.3)                 # hidden states up
c.arrow(bx + bw, 1.22, px, 0.66, color=NET, lw=0.9, dash=True)      # verdict up
c.arrow(px, 1.02, bx + bw, 1.68, color=NET, lw=1.3)                 # results down
# what crosses the network, in the words of Figure 1
c.text(1.38, 0.34, 0.50, 0.30, "hidden\nstates", fs=FS, color=NET, italic=True)
c.text(1.54, 0.83, 0.56, 0.18, "verdict", fs=FS, color=NET, italic=True)
c.text(1.42, 1.58, 0.50, 0.30, "hidden\nstates", fs=FS, color=NET, italic=True)

# step numbers of one round: components 1-6, the verdict 7, the next round 8
for n, (x, y) in enumerate([(bx, 0.27), (bx, 0.63), (bx, 0.99), (px, 0.27), (px, 0.81), (bx, 1.54)], start=1):
    c.badge(x, y, str(n))
c.badge(1.42, 1.22, "7")   # the verdict leaves the scheduler
c.badge(0.17, 1.10, "8")   # the next round starts from the accepted tokens

c.save(str(HERE / "design_arch"))
