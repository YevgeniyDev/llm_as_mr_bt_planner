"""Slide 7 (Experiments): Evaluation Protocol — clean scientific version (no cartoon icons)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from figtools import *
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

W, Hin = 12.4, 5.6
fig, ax = new_ax(W, Hin)
YT = H(W, Hin); ax.set_ylim(0, YT)

LX1 = 70.5
# ============ TOP: 4-step protocol pipeline ============
py0, py1 = YT-11.5, YT-0.8
steps = [
    ("1", "Symbolic\nsimulation", "world-state predicates", BLUE_DK),
    ("2", "Stochastic\nLLM", "multiple trials", BLUE),
    ("3", "Evaluate\noutcomes", "metrics per trial", GREEN_DK),
    ("4", "Report", "mean ± std", PURPLE),
]
sw = (LX1 - 1.5 - 3*3.0) / 4
for i, (n, t, sub, c) in enumerate(steps):
    x = 1.5 + i*(sw+3.0)
    box(ax, x, py0, sw, py1-py0, fc=tint(c, .94), ec=c, lw=1.5, rad=0.7)
    ax.add_patch(Circle((x+2.4, py1-2.2), 1.5, fc=c, ec="none", zorder=6)); text(ax, x+2.4, py1-2.2, n, size=11, color="white", weight="bold", z=7)
    text(ax, x+sw/2+1.0, py1-2.2, t, size=9.3, color=c, weight="bold")
    text(ax, x+sw/2, py0+1.7, sub, size=7.6, color=SUB, style="italic")
    if i == 3:  # tiny bar chart
        bx = x+sw/2-4.5; base = py0+3.6
        for k, hh in enumerate([1.8, 2.6, 3.0, 3.6]):
            ax.add_patch(Rectangle((bx+k*2.4, base), 1.6, hh, fc=tint(PURPLE, .4), ec=PURPLE, lw=1.0, zorder=6))
            ax.add_line(Line2D([bx+k*2.4+0.8, bx+k*2.4+0.8], [base+hh-0.4, base+hh+0.6], color=SUB, lw=1.0, zorder=7))
    if i < 3:
        arrow(ax, x+sw+0.3, (py0+py1)/2, x+sw+2.7, (py0+py1)/2, color=SUB, lw=1.8, mut=11)

# Ablations panel (right)
ax0 = LX1+1.5
box(ax, ax0, py0-0.0, 100-ax0-0.5, py1-py0, fc=tint(ORANGE, .95), ec=tint(ORANGE, .5), lw=1.4, rad=0.7)
text(ax, (ax0+99.5)/2, py1-1.8, "Ablations", size=10.5, color=ORANGE, weight="bold", font=HEAD)
line(ax, ax0+2, py1-3.0, 99, py1-3.0, color=tint(ORANGE, .5), lw=0.9)
abls = ["Single-shot  (C=0)", "Self-correction  (C=4)", "Best-of-N  (N=4)", "Two-stage  (plan→BT)", "Combined"]
for i, a in enumerate(abls):
    yy = py1-4.8-i*1.55
    ax.add_patch(Circle((ax0+2.4, yy), 0.55, fc=ORANGE, ec="none", zorder=6))
    text(ax, ax0+4.0, yy, a, size=7.8, color=INK, ha="left")

# ============ MID: two scenario cards ============
my0, my1 = YT-25.5, YT-13.0
cards = [
    ("Gear Assembly", BLUE, ["3 robots", "13 capabilities", "4 goals", "3 sync points"]),
    ("Sensor Calibration Cell", GREEN_DK, ["3 robots", "19 capabilities", "6 goals", "more sync"]),
]
cw = (100 - 1.5 - 3.0)/2 - 1.5
for i, (name, c, stats) in enumerate(cards):
    x = 1.5 + i*(cw+3.0)
    box(ax, x, my0, cw, my1-my0, fc=tint(c, .95), ec=tint(c, .5), lw=1.4, rad=0.7)
    text(ax, x+3.5, my1-1.9, name, size=10.5, color=c, weight="bold", ha="left", font=HEAD)
    # robot chips
    for j, (lab, col) in enumerate([("r₁", BLUE), ("r₂", ORANGE), ("r₃", PURPLE)]):
        robot_chip(ax, x+3.0+j*5.0, my0+1.6, 4.2, 3.0, lab, col)
    for j, s in enumerate(stats):
        col = 2*(j//2); row = j % 2
        text(ax, x+22 + (j//2)*11, my1-4.5 - row*3.0, "• "+s, size=8.3, color=INK, ha="left")

# ============ BOTTOM: metrics band ============
by0, by1 = 1.0, YT-27.0
box(ax, 1.5, by0, 98.5-1.5, by1-by0, fc=PANEL, ec=LINE, lw=1.3, rad=0.7)
text(ax, 4.0, by1-1.7, "Metrics", size=9.5, color=INK, weight="bold", ha="left", font=HEAD)
mets = [
    ("Validity", ic_shield, GREEN_DK), ("Goal success", ic_target, BLUE),
    ("Sync errors", ic_warn, ORANGE), ("Correction rounds", ic_refresh, PURPLE),
    ("Wall time", ic_clock, BLUE_DK), ("BT size · tokens", ic_link, GRAY),
]
n = len(mets); span = 96/n
cy = (by0+by1)/2 - 1.0
for i, (name, icon, c) in enumerate(mets):
    x = 3.5 + i*span + span/2 - 2
    icon(ax, x, cy+0.4, 1.5, c)
    text(ax, x, cy-3.0, name, size=7.8, color=INK)
    if i < n-1:
        line(ax, 3.5+(i+1)*span-1, by0+1.5, 3.5+(i+1)*span-1, by1-1.5, color=LINE, lw=0.8)

fig.savefig(os.path.join(os.path.dirname(__file__), "fig_eval.png"), dpi=220, facecolor="white", pad_inches=0)
print("saved eval")
