"""Figure: Problem Formulation — clean scientific version (replaces cartoon diagram)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from figtools import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon

W, Hin = 12.6, 5.95
fig, ax = new_ax(W, Hin)
YT = H(W, Hin)  # top of world (~47.2)
ax.set_ylim(0, YT)

LX0, LX1 = 1.5, 70.5           # left column (tiers)
RX0, RX1 = 73.0, 98.5          # right panel (valid plan)
CX0 = 16.0                      # start of content zone inside each tier

TIERS = [
    ("Scenario",         r"$S=(I,G,O,L,R)$",            BLUE),
    ("Capability",       r"$c=(\mathrm{name},\ p,\ \mathrm{pre},\ \mathrm{eff})$", GREEN_DK),
    ("State transition", r"$w \rightarrow c \rightarrow w'$",  ORANGE),
    ("Plan",             r"$P=(T,A,Y,B)$",              PURPLE),
]
th = 9.6
gap = (YT - 1.0 - 4 * th) / 3.0
tops = [YT - 0.6 - i * (th + gap) for i in range(4)]   # top y of each tier

def tier_frame(i):
    color = TIERS[i][2]
    y1 = tops[i]; y0 = y1 - th
    box(ax, LX0, y0, LX1 - LX0, th, fc=tint(color, .94), ec=tint(color, .55), lw=1.3, rad=0.8)
    # accent stripe
    box(ax, LX0, y0, 1.1, th, fc=color, ec=color, rad=0.5, z=3)
    section_tag(ax, LX0 + 4.2, y1 - 2.4, i + 1, color)
    text(ax, LX0 + 7.2, y1 - 2.4, TIERS[i][0], size=12, color=color, weight="bold", ha="left", font=HEAD)
    text(ax, LX0 + 4.2, y0 + 2.4, TIERS[i][1], size=10.5, color=INK, ha="left")
    return y0, y1, color

def col_header(x, y, s, color):
    text(ax, x, y, s, size=9.3, color=SUB, weight="bold", ha="center")

# ---------- Tier 1: Scenario ----------
y0, y1, C = tier_frame(0)
midy = (y0 + y1) / 2 - 0.3
cols = ["Initial  I", "Goal  G", "Objects  O", "Locations  L", "Robots  R"]
xs = [22, 32, 43.5, 54.5, 64.5]
for x, h in zip(xs, cols):
    col_header(x, y1 - 1.4, h, C)
# grids I and G (3x3)
def grid(cx, cy, goal=False):
    g = 1.35
    for r in range(3):
        for c in range(3):
            sq(ax, cx - 1.5 * g + c * g, cy - 1.5 * g + r * g, g, g, fc="white", ec=tint(BLUE, .3), lw=0.9)
    ax.add_patch(Circle((cx - g, cy - g), 0.55, fc=tint(BLUE, .3), ec=BLUE, lw=1.0, zorder=5))
    ax.add_patch(Rectangle((cx + 0.1 * g, cy + 0.1 * g), 0.9, 0.9, fc=tint(GREEN, .4), ec=GREEN, lw=1.0, zorder=5))
    if goal:
        ax.add_patch(Rectangle((cx - 1.5 * g + 0.15, cy - 1.5 * g + 0.15), g - 0.3, g - 0.3,
                               fc="none", ec=GREEN, lw=1.4, ls=(0, (2, 1.5)), zorder=6))
grid(22, midy); grid(32, midy, goal=True)
# objects O
for gk, gx, gc in [("square", 41.5, GREEN), ("circle", 43.5, BLUE), ("triangle", 45.5, ORANGE)]:
    glyph(ax, gk, gx, midy + 0.2, 1.0, gc)
text(ax, 43.5, midy - 2.4, r"$o_1,\ o_2,\ \dots$", size=8.5, color=SUB)
# locations L  (labelled markers)
for lx, lab in [(52.5, r"$l_1$"), (54.5, r"$l_2$"), (56.5, r"$l_3$")]:
    ax.add_patch(RegularPolygon((lx, midy + 0.4), 4, radius=1.0, fc=tint(PURPLE, .55), ec=PURPLE, lw=1.2, zorder=5))
    text(ax, lx, midy - 2.4, lab, size=8.5, color=SUB)
# robots R
for i, (lab, col) in enumerate([("r₁", BLUE), ("r₂", ORANGE), ("r₃", PURPLE)]):
    robot_chip(ax, 61 + i * 3.1 - 1.3, midy - 0.9, 2.6, 2.2, lab, col)

# ---------- Tier 2: Capability ----------
y0, y1, C = tier_frame(1)
midy = (y0 + y1) / 2 - 0.3
for x, h in zip([24, 38, 51, 64], ["action name", "parameters", "preconditions  pre(c)", "effects  eff = (add, del)"]):
    col_header(x, y1 - 1.4, h, C)
box(ax, 20, midy - 1.3, 8.5, 2.6, fc="white", ec=GREEN_DK, lw=1.3, rad=0.4)
text(ax, 24.25, midy, r"$\mathtt{pick(o,l)}$", size=9.5, color=GREEN_DK, weight="bold")
for i, t in enumerate(["o", "l"]):
    box(ax, 35 + i * 3.2, midy - 1.0, 2.6, 2.0, fc=tint(BLUE, .8), ec=BLUE, lw=1.1, rad=0.3)
    text(ax, 36.3 + i * 3.2, midy, t, size=9, color=BLUE_DK, weight="bold")
box(ax, 46.5, midy - 1.1, 9, 2.2, fc="white", ec=tint(GREEN_DK, .4), lw=1.1, rad=0.3)
text(ax, 51, midy, r"$\mathtt{at(o,l)}\ \wedge\ \mathtt{free(r)}$", size=8.2, color=INK)
# effects add / del
ax.add_patch(Circle((60.5, midy + 0.7), 0.85, fc=GREEN, ec="none", zorder=6)); text(ax, 60.5, midy + 0.7, "+", size=11, color="white", weight="bold", z=7)
text(ax, 62, midy + 0.7, r"$\mathtt{holding(o)}$", size=7.8, color=GREEN_DK, ha="left")
ax.add_patch(Circle((60.5, midy - 0.9), 0.85, fc=RED, ec="none", zorder=6)); text(ax, 60.5, midy - 0.9, "−", size=11, color="white", weight="bold", z=7)
text(ax, 62, midy - 0.9, r"$\mathtt{at(o,l)}$", size=7.8, color=RED, ha="left")

# ---------- Tier 3: State transition ----------
y0, y1, C = tier_frame(2)
midy = (y0 + y1) / 2 - 0.3
box(ax, 19.5, midy - 3.0, 9.5, 6.0, fc="white", ec=tint(ORANGE, .4), lw=1.2, rad=0.4)
text(ax, 24.25, midy + 2.0, r"$w$  (world state)", size=8.2, color=SUB)
for i, p in enumerate([r"$\mathtt{at(o,l_1)}$", r"$\mathtt{free(r)}$", r"$\mathtt{clear(l_2)}$"]):
    text(ax, 24.25, midy + 0.4 - i * 1.4, p, size=7.6, color=INK)
arrow(ax, 29.5, midy, 33.0, midy, color=ORANGE, lw=2.0)
box(ax, 33.5, midy - 1.6, 8.5, 3.2, fc=tint(ORANGE, .85), ec=ORANGE, lw=1.3, rad=0.4)
text(ax, 37.75, midy, r"apply $c$", size=9.5, color=ORANGE, weight="bold")
arrow(ax, 42.5, midy, 46.0, midy, color=ORANGE, lw=2.0)
# del then add
ax.add_patch(Circle((49.0, midy + 1.4), 0.8, fc=RED, ec="none", zorder=6)); text(ax, 49.0, midy + 1.4, "−", size=10, color="white", weight="bold", z=7)
text(ax, 50.2, midy + 1.4, "delete", size=8, color=RED, ha="left")
ax.add_patch(Circle((49.0, midy - 1.4), 0.8, fc=GREEN, ec="none", zorder=6)); text(ax, 49.0, midy - 1.4, "+", size=10, color="white", weight="bold", z=7)
text(ax, 50.2, midy - 1.4, "add", size=8, color=GREEN_DK, ha="left")
text(ax, 49.5, midy + 3.2, "delete before add", size=7.6, color=SUB, style="italic")
arrow(ax, 55.5, midy, 59.0, midy, color=ORANGE, lw=2.0)
box(ax, 59.5, midy - 3.0, 9.5, 6.0, fc=tint(GREEN, .93), ec=tint(GREEN_DK, .4), lw=1.2, rad=0.4)
text(ax, 64.25, midy + 2.0, r"$w'$  (new state)", size=8.2, color=SUB)
for i, p in enumerate([r"$\mathtt{holding(o)}$", r"$\mathtt{free(r)}$", "…"]):
    text(ax, 64.25, midy + 0.4 - i * 1.4, p, size=7.6, color=INK)

# ---------- Tier 4: Plan ----------
y0, y1, C = tier_frame(3)
midy = (y0 + y1) / 2 - 0.3
for x, h in zip([23, 38, 50.5, 63], ["task graph  T", "assignment  A:T→R", "sync set  Y", "BT forest  B"], ):
    col_header(x, y1 - 1.4, h, C)
# task graph DAG
pos = {"t1": (23, midy + 2.4), "t2": (20, midy - 0.3), "t3": (26, midy - 0.3),
       "t4": (20, midy - 2.9), "t5": (26, midy - 2.9)}
for a, b in [("t1", "t2"), ("t1", "t3"), ("t2", "t4"), ("t3", "t5")]:
    line(ax, *pos[a], *pos[b], color=tint(PURPLE, .3), lw=1.3)
for k, (x, y) in pos.items():
    node(ax, x, y, 1.15, k[0] + r"$_%s$" % k[1], fc="white", ec=PURPLE, tcolor=PURPLE, tsize=7.5)
# assignment A: t -> r
ax.add_patch(Circle((33.5, midy + 1.6), 1.0, fc="white", ec=PURPLE, lw=1.3, zorder=4)); text(ax, 33.5, midy + 1.6, r"$t_i$", size=7.5, color=PURPLE, weight="bold", z=5)
for i, (lab, col, yy) in enumerate([("r₁", BLUE, midy + 2.4), ("r₂", ORANGE, midy), ("r₃", PURPLE, midy - 2.4)]):
    robot_chip(ax, 40.5, yy - 1.0, 2.4, 2.0, lab, col)
    arrow(ax, 34.7, midy + 1.6 - i * 0.2, 40.3, yy, color=tint(col, .1), lw=1.3, mut=9, rad=0.12)
# sync set Y
for i, s in enumerate([r"$t_2 \rightarrow t_3\ [c_1]$", r"$t_4 \rightarrow t_3\ [c_2]$"]):
    box(ax, 46.5, midy + 0.4 - i * 2.4, 8.5, 1.9, fc="white", ec=tint(BLUE, .35), lw=1.1, rad=0.3)
    text(ax, 50.75, midy + 1.35 - i * 2.4, s, size=7.8, color=BLUE_DK)
# BT forest: 3 tiny trees
def tinytree(cx, cy, col, root="?"):
    node(ax, cx, cy, 1.0, root, fc=tint(col, .8), ec=col, tcolor=col, tsize=8)
    for dx in (-1.8, 1.8):
        line(ax, cx, cy - 1.0, cx + dx, cy - 2.6, color=tint(col, .3), lw=1.1)
        sq(ax, cx + dx - 0.8, cy - 3.4, 1.6, 1.2, fc=tint(col, .8), ec=col, lw=1.0)
for i, col in enumerate([BLUE, ORANGE, PURPLE]):
    tinytree(60 + i * 4.3, midy + 2.0, col, root="?" if i % 2 == 0 else "→")

# ================= Right panel: Valid plan =================
box(ax, RX0, 0.6, RX1 - RX0, YT - 1.2, fc=tint(BLUE, .96), ec=tint(BLUE, .5), lw=1.4, rad=1.0)
text(ax, (RX0 + RX1) / 2, YT - 2.9, "Valid plan", size=13.5, color=BLUE_DK, weight="bold", font=HEAD)
line(ax, RX0 + 2, YT - 4.6, RX1 - 2, YT - 4.6, color=tint(BLUE, .5), lw=1.0)
checks = [
    ("V1", "Assigned action matches robot capability"),
    ("V2", "Every precondition is supported"),
    ("V3", "Each inter-robot dependency\nhas a real producer + guard"),
    ("V4", "Task graph acyclic and BT schema valid"),
    ("V5", "Simulation reaches goal with no deadlock"),
]
ry0 = YT - 6.8; rstep = (ry0 - 4.0) / 4
for i, (v, t) in enumerate(checks):
    yy = ry0 - i * rstep
    ax.add_patch(Circle((RX0 + 2.8, yy), 1.35, fc=BLUE, ec="none", zorder=6))
    text(ax, RX0 + 2.8, yy, v, size=8.2, color="white", weight="bold", z=7)
    check(ax, RX0 + 5.6, yy, r=1.05, color=GREEN)
    text(ax, RX0 + 7.4, yy, t, size=7.5, color=INK, ha="left", va="center")
    if i < 4:
        line(ax, RX0 + 2, yy - rstep / 2, RX1 - 2, yy - rstep / 2, color=tint(BLUE, .7), lw=0.7)
text(ax, (RX0 + RX1) / 2, 2.2, "V1–V4: static checks   |   V5: simulation",
     size=8, color=BLUE_DK, weight="bold", style="italic")

out = os.path.join(os.path.dirname(__file__), "fig_problem.png")
fig.savefig(out, dpi=220, facecolor="white", bbox_inches=None, pad_inches=0)
print("saved", out)
