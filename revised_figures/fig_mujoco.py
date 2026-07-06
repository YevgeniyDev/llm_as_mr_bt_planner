"""NEW slide: MuJoCo Physics Testing (feedback #5) — 3-zone layout."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from figtools import *
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib.lines import Line2D

W, Hin = 12.4, 5.55
fig, ax = new_ax(W, Hin)
YT = H(W, Hin); ax.set_ylim(0, YT)

def head(x, yy, num, title, sub, color):
    ax.add_patch(Circle((x+1.6, yy), 1.4, fc=color, ec="none", zorder=6))
    text(ax, x+1.6, yy, num, size=10.5, color="white", weight="bold", z=7)
    text(ax, x+3.5, yy+0.45, title, size=10.8, color=color, weight="bold", ha="left", font=HEAD)
    text(ax, x+3.5, yy-1.15, sub, size=7.8, color=SUB, ha="left", style="italic")

# ================= ZONE 1 (top): execution pipeline =================
head(1.0, YT-2.3, "1", "Replay BTs in real physics", "the same tick engine drives MuJoCo — physics only embodies", BLUE)
py0, py1 = YT-11.0, YT-6.0
stages = [
    ("Plan\n(BT forest)", GREEN_DK), ("Tick engine\nsimulate()", BLUE),
    ("on_action\nhook", ORANGE), ("MuJoCo\nphysics step", PURPLE),
    ("Verdict\n(symbolic)", GREEN_DK),
]
x0, sw, gap = 2.5, 16.2, 3.05
for i, (t, c) in enumerate(stages):
    x = x0 + i*(sw+gap)
    box(ax, x, py0, sw, py1-py0, fc=tint(c, .93), ec=c, lw=1.5, rad=0.6)
    text(ax, x+sw/2, (py0+py1)/2, t, size=8.5, color=c, weight="bold")
    if i < len(stages)-1:
        arrow(ax, x+sw+0.2, (py0+py1)/2, x+sw+gap-0.2, (py0+py1)/2, color=SUB, lw=1.8, mut=11)
check(ax, x0+4*(sw+gap)+3.0, py1-1.3, r=0.75, color=GREEN)
text(ax, (x0+x0+5*sw+4*gap)/2, py0-1.6,
     "physics embodies & traces every action  ·  success / deadlock / goal are judged by the symbolic engine",
     size=8.0, color=SUB, style="italic")

# horizontal separator
line(ax, 1.5, YT-13.0, 98.5, YT-13.0, color=LINE, lw=1.0)
# vertical divider for lower half
line(ax, 50.0, 1.5, 50.0, YT-13.8, color=LINE, lw=1.1)

# ================= ZONE 2 (bottom-left): scene =================
head(1.0, YT-16.0, "2", "Scene from MuJoCo Menagerie", "fixed-base robots on a shared table cell", GREEN_DK)
sy = YT-19.5
for i, (m, r, c) in enumerate([("Unitree Go2 + Z1", "mobile manipulator", BLUE),
                               ("Franka Panda ×2", "manipulators", ORANGE)]):
    y = sy - i*3.0
    box(ax, 2.0, y-1.2, 27, 2.4, fc=tint(c, .9), ec=tint(c, .5), lw=1.2, rad=0.4)
    ax.add_patch(Circle((3.4, y), 0.55, fc=c, ec="none", zorder=6))
    text(ax, 4.7, y, m, size=7.9, color=c, weight="bold", ha="left")
    text(ax, 28.5, y, r, size=6.8, color=SUB, ha="right", style="italic")
text(ax, 2.0, sy-7.0, "• objects: gear · shaft · screwdriver · tray", size=7.6, color=INK, ha="left")
text(ax, 2.0, sy-9.2, "• drawer = cabinet + real slide joint (0–0.20 m)", size=7.6, color=INK, ha="left")

# schematic scene box (geometric, not cartoon)
scx, scy, scw, sch = 31.5, 2.5, 17.0, 16.5
box(ax, scx, scy, scw, sch, fc=PANEL, ec=LINE, lw=1.2, rad=0.5)
tbl_y = scy+4.5
ax.add_patch(Rectangle((scx+2.5, tbl_y), scw-5, 1.4, fc=tint(ORANGE, .5), ec=ORANGE, lw=1.2, zorder=5))
for lx in (scx+4, scx+scw-4.5):
    ax.add_line(Line2D([lx, lx], [tbl_y, tbl_y-2.4], color=ORANGE, lw=1.6, zorder=5))
for i, (gk, c) in enumerate([("circle", GREEN), ("square", BLUE), ("triangle", PURPLE)]):
    glyph(ax, gk, scx+5.2+i*3.3, tbl_y+2.2, 0.7, c)
for i, (px, c, lab) in enumerate([(scx+4.5, BLUE, "Go2+Z1"), (scx+scw/2, ORANGE, "Panda"), (scx+scw-4.5, PURPLE, "Panda")]):
    by = tbl_y+1.4
    ax.add_patch(Rectangle((px-0.6, by), 1.2, 2.0, fc=tint(c, .6), ec=c, lw=1.1, zorder=6))
    ax.add_line(Line2D([px, px+1.3], [by+2.0, by+3.4], color=c, lw=1.7, zorder=6, solid_capstyle="round"))
    ax.add_line(Line2D([px+1.3, px+2.0], [by+3.4, by+2.5], color=c, lw=1.7, zorder=6, solid_capstyle="round"))
    text(ax, px, by-1.0, lab, size=5.6, color=c, ha="center")
text(ax, scx+scw/2, scy+1.0, "gear-assembly cell", size=6.6, color=SUB, style="italic")

# ================= ZONE 3 (bottom-right): fidelity ladder =================
head(51.5, YT-16.0, "3", "Two-stage fidelity ladder", "raise realism without changing plan semantics", PURPLE)
stages2 = [
    ("Stage 1 — settle", BLUE, ["scripted teleport to target", "gravity settle (mj_step)", "kinematic pick / place / drawer"]),
    ("Stage 2 — IK", ORANGE, ["damped least-squares IK reach", "Jacobian on end-effector body", "snap held object to gripper"]),
]
bw, bh = 21.5, 13.0
bx = [51.5, 51.5+bw+4.0]
by = 8.5
for i, (t, c, bl) in enumerate(stages2):
    x = bx[i]
    box(ax, x, by, bw, bh, fc=tint(c, .93), ec=c, lw=1.6, rad=0.6)
    text(ax, x+bw/2, by+bh-2.0, t, size=9.2, color=c, weight="bold")
    for j, b in enumerate(bl):
        text(ax, x+1.8, by+bh-4.7 - j*2.3, "•  " + b, size=7.4, color=INK, ha="left")
arrow(ax, bx[0]+bw+0.3, by+bh/2, bx[1]-0.3, by+bh/2, color=SUB, lw=1.8, mut=12)
text(ax, (bx[0]+bw+bx[1])/2, by+bh/2+1.6, "more\nrealism", size=6.8, color=SUB, ha="center", style="italic")

# run command + footnote
box(ax, 51.5, 2.5, bw+bw+4.0, 3.6, fc="#1E2733", ec="#1E2733", lw=1.0, rad=0.4)
text(ax, 53.0, 4.3, "python -m llm_mr_bt_planner run  --backend mujoco  --physics ik  --render",
     size=7.2, color="#8FE3B0", ha="left", font="Consolas")
text(ax, 51.7, 0.9, "grasping is kinematic; friction-based grasping is future work",
     size=7.0, color=SUB, ha="left", style="italic")

fig.savefig(os.path.join(os.path.dirname(__file__), "fig_mujoco.png"), dpi=220, facecolor="white", pad_inches=0)
print("saved mujoco")
