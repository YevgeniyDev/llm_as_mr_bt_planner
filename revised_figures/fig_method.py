"""Slide 6 (Our Method): two clean figures — architecture+loop, and reliability levers."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from figtools import *
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib.lines import Line2D

# ============================================================ TOP: architecture + self-correction loop
W, Hin = 12.1, 2.95
fig, ax = new_ax(W, Hin)
YT = H(W, Hin)
ax.set_ylim(0, YT)
midy = YT * 0.60

def stage(x, w, h, title, color, sub=None):
    y = midy - h / 2
    box(ax, x, y, w, h, fc=tint(color, .93), ec=color, lw=1.6, rad=0.7)
    box(ax, x, y + h - 0.9, w, 0.9, fc=color, ec=color, rad=0.4, z=3)
    text(ax, x + w / 2, y + h - 0.45, title, size=9.8, color="white", weight="bold", z=6)
    return x, y, w, h

# Input
ix, iy, iw, ih = stage(1.5, 15, 12, "Input", BLUE_DK)
for i, s in enumerate(["Goal state  G", "World state  I", "Robot capabilities"]):
    text(ax, ix + iw / 2, iy + 8.3 - i * 2.6, s, size=8.2, color=INK)
# LLM Planner
lx, ly, lw_, lh = stage(20.5, 15.5, 12, "LLM Planner", BLUE)
text(ax, lx + lw_ / 2, ly + 5.2, "generates the\nwhole plan", size=8.6, color=INK, style="italic")
# Plan JSON
px, py, pw, ph = stage(40, 17, 12, "Plan  (JSON)", GREEN_DK)
for i, s in enumerate([r"Task graph $T$", r"Assignments $A$", r"Sync set $Y$", r"BT forest $B$"]):
    row, col = divmod(i, 2)
    text(ax, px + 4.6 + col * 8.6, py + 6.0 - row * 3.0, s, size=7.8, color=INK, ha="center")
# Verifier
vx, vy, vw, vh = stage(60.5, 17.5, 12, "Code Verifier", PURPLE)
for i, s in enumerate(["validate", "simulate"]):
    check(ax, vx + 2.6, vy + 6.0 - i * 3.0, r=0.85, color=GREEN)
    text(ax, vx + 4.6, vy + 6.0 - i * 3.0, s, size=8.4, color=INK, ha="left")
# goal
gx = 81.5
ax.add_patch(Circle((gx + 4, midy), 4.4, fc=tint(GREEN, .88), ec=GREEN_DK, lw=1.8, zorder=4))
check(ax, gx + 4, midy + 1.1, r=1.5, color=GREEN)
text(ax, gx + 4, midy - 2.0, "goal\nreached", size=8.4, color=GREEN_DK, weight="bold")

for x1, x2 in [(ix + iw, lx), (lx + lw_, px), (px + pw, vx), (vx + vw, gx - 0.5)]:
    arrow(ax, x1, midy, x2, midy, color=SUB, lw=2.0, mut=13)

# self-correction feedback loop (verifier -> planner)
fy = midy - vh / 2 - 3.2
arrow(ax, vx + vw / 2, vy, vx + vw / 2, fy, color=ORANGE, lw=1.8, mut=11)
line(ax, vx + vw / 2, fy, lx + lw_ / 2, fy, color=ORANGE, lw=1.8)
arrow(ax, lx + lw_ / 2, fy, lx + lw_ / 2, ly, color=ORANGE, lw=1.8, mut=11)
text(ax, (lx + vx + vw) / 2 + 3, fy - 1.7, "structured feedback + previous plan   (self-correction, ≤ C rounds)",
     size=8.2, color=ORANGE, weight="bold", ha="center")
text(ax, gx + 4, midy + 6.4, "code only verifies — never repairs", size=8, color=SUB, style="italic")

fig.savefig(os.path.join(os.path.dirname(__file__), "fig_method_top.png"), dpi=220, facecolor="white", pad_inches=0)
print("saved top")

# ============================================================ BOTTOM: reliability levers (4 panels)
W, Hin = 12.1, 2.6
fig, ax = new_ax(W, Hin)
YT = H(W, Hin)
ax.set_ylim(0, YT)

panels = [
    ("Back-chaining", BLUE, "goal regression in the prompt"),
    ("Blocking-guard sync", GREEN_DK, "producer creates, consumer waits"),
    ("Best-of-N sampling", ORANGE, "keep first valid + simulated"),
    ("Two-stage generation", PURPLE, "action plan → BT encoding"),
]
pw = (100 - 5 * 1.2) / 4
for i, (title, color, sub) in enumerate(panels):
    x = 1.2 + i * (pw + 1.2)
    box(ax, x, 1.0, pw, YT - 2.0, fc=tint(color, .95), ec=tint(color, .55), lw=1.3, rad=0.7)
    ax.add_patch(Circle((x + 2.2, YT - 3.0), 1.25, fc=color, ec="none", zorder=6))
    text(ax, x + 2.2, YT - 3.0, str(i + 1), size=9.5, color="white", weight="bold", z=7)
    text(ax, x + 4.0, YT - 3.0, title, size=9.2, color=color, weight="bold", ha="left")
    text(ax, x + pw / 2, 2.2, sub, size=7.4, color=SUB, style="italic")
    cy = (YT) / 2 - 0.3
    cx = x + pw / 2
    if i == 0:  # back-chaining: G -> ... -> I
        node(ax, cx + 8.5, cy, 1.15, "G", fc=tint(GREEN, .8), ec=GREEN_DK, tcolor=GREEN_DK, tsize=8)
        node(ax, cx - 8.5, cy, 1.15, "I", fc="white", ec=BLUE, tcolor=BLUE, tsize=8)
        for k in range(3):
            node(ax, cx - 8.5 + (k + 1) * 4.25, cy, 0.85, "", fc="white", ec=tint(BLUE, .3))
        for k in range(4):
            arrow(ax, cx + 8.5 - k * 4.25 - 1.1, cy, cx + 8.5 - (k + 1) * 4.25 + 1.1, cy,
                  color=tint(BLUE, .2), lw=1.5, mut=9)
    elif i == 1:  # blocking guard sync
        for j, (lab, col, yy) in enumerate([("r₁ produces", GREEN_DK, cy + 2.6), ("r₂ waits", ORANGE, cy - 2.6)]):
            line(ax, cx - 9, yy, cx + 6, yy, color=tint(col, .3), lw=1.4)
            node(ax, cx - 9, yy, 1.0, "", fc=tint(col, .7), ec=col)
            text(ax, cx - 6.5, yy + 1.4, lab, size=6.8, color=col, ha="left")
        # guard drop
        line(ax, cx + 1.5, cy + 2.6, cx + 1.5, cy - 2.6, color=SUB, lw=1.3, ls=(0, (2, 1.5)))
        check(ax, cx + 1.5, cy, r=1.15, color=GREEN)
        node(ax, cx + 6, cy - 2.6, 1.0, "", fc="white", ec=ORANGE, lw=1.4)
        text(ax, cx + 1.5, cy - 3.6, "guard", size=6.6, color=SUB, style="italic")
    elif i == 2:  # best-of-N
        for k, ok in enumerate([False, True, False]):
            yy = cy + 2.4 - k * 2.4
            col = GREEN if ok else GRAY
            box(ax, cx - 2.5, yy - 0.85, 5.0, 1.6, fc=tint(col, .8), ec=col, lw=1.2, rad=0.3)
            (check if ok else (lambda *a, **k: None))(ax, cx + 3.6, yy, r=0.8, color=GREEN)
            text(ax, cx, yy, "cand %d" % (k + 1), size=6.8, color=INK)
    elif i == 3:  # two-stage
        box(ax, cx - 9, cy - 1.1, 7.5, 2.2, fc="white", ec=PURPLE, lw=1.3, rad=0.3)
        text(ax, cx - 5.25, cy, "action plan", size=7.2, color=PURPLE, weight="bold")
        arrow(ax, cx - 1.3, cy, cx + 1.3, cy, color=PURPLE, lw=1.8, mut=10)
        box(ax, cx + 1.5, cy - 1.1, 7.5, 2.2, fc=tint(PURPLE, .85), ec=PURPLE, lw=1.3, rad=0.3)
        text(ax, cx + 5.25, cy, "BT encoding", size=7.2, color=PURPLE, weight="bold")

fig.savefig(os.path.join(os.path.dirname(__file__), "fig_method_bottom.png"), dpi=220, facecolor="white", pad_inches=0)
print("saved bottom")
