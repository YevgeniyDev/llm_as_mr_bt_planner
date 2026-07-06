"""NEW slide: LLM Skills & Prompt Engineering (feedback #4)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from figtools import *
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

W, Hin = 12.4, 5.55
fig, ax = new_ax(W, Hin)
YT = H(W, Hin); ax.set_ylim(0, YT)

def head(x, num, title, sub, color):
    ax.add_patch(Circle((x+1.8, YT-2.3), 1.5, fc=color, ec="none", zorder=6))
    text(ax, x+1.8, YT-2.3, num, size=11, color="white", weight="bold", z=7)
    text(ax, x+3.8, YT-1.9, title, size=11.5, color=color, weight="bold", ha="left", font=HEAD)
    text(ax, x+3.8, YT-3.4, sub, size=8.2, color=SUB, ha="left", style="italic")

# ---------- LEFT: skills as capability cards (markdown -> JSON -> prompt) ----------
LX = 1.0
head(LX, "1", "Skills = capability cards", "each robot skill authored as a Markdown card", BLUE)
# markdown card
cx0, cy0, cw, chh = LX+1.5, YT-20.5, 40, 13.5
box(ax, cx0, cy0, cw, chh, fc="white", ec=BLUE, lw=1.6, rad=0.6)
box(ax, cx0, cy0+chh-2.2, cw, 2.2, fc=tint(BLUE, .85), ec=BLUE, lw=1.6, rad=0.5, z=3)
ax.add_patch(Circle((cx0+1.6, cy0+chh-1.1), 0.45, fc=BLUE, ec="none", zorder=6))
text(ax, cx0+3.0, cy0+chh-1.1, "franka2.skill.md", size=8.6, color=BLUE_DK, weight="bold", ha="left", font="Consolas")
md = [
    ("## skill: mount_gear", BLUE_DK, "bold"),
    ("params:  gear, base", INK, "normal"),
    ("pre:     holding(gear)", INK, "normal"),
    ("         & stabilized(base)", INK, "normal"),
    ("effects:", INK, "normal"),
    ("  + mounted(gear, base)", GREEN_DK, "normal"),
    ("  - holding(gear)", RED, "normal"),
]
for i, (ln, col, wt) in enumerate(md):
    text(ax, cx0+2.0, cy0+chh-3.6-i*1.35, ln, size=8.0, color=col, weight=wt, ha="left", font="Consolas")
# compile chain
arrow(ax, cx0+cw/2, cy0-0.2, cx0+cw/2, cy0-3.0, color=SUB, lw=1.8, mut=11)
text(ax, cx0+cw/2+0.5, cy0-1.6, "compile", size=7.6, color=SUB, ha="left", style="italic")
box(ax, cx0+3, cy0-8.0, cw-6, 4.4, fc=tint(GREEN, .93), ec=GREEN_DK, lw=1.4, rad=0.5)
text(ax, cx0+cw/2, cy0-4.9, "Robot capability library  (JSON)", size=8.8, color=GREEN_DK, weight="bold")
text(ax, cx0+cw/2, cy0-6.9, "name · parameters · preconditions · add/del effects", size=7.6, color=INK)
arrow(ax, cx0+cw/2, cy0-8.2, cx0+cw/2, cy0-10.6, color=SUB, lw=1.8, mut=11)
text(ax, cx0+cw/2, cy0-11.6, "injected into the LLM prompt context", size=8.0, color=INK, style="italic")

# divider
line(ax, 47.5, 2.0, 47.5, YT-1.5, color=LINE, lw=1.2)

# ---------- RIGHT: prompt engineering stack ----------
RX = 49.5
head(RX, "2", "Prompt engineering stack", "task-agnostic, structured, self-correcting", PURPLE)
layers = [
    ("System prompt", "“produce strict JSON plans…”", BLUE_DK),
    ("Scenario + capability library", "world state + robot skills", GREEN_DK),
    ("Output schema (JSON)", "task_graph · assignments · sync · BT forest", BLUE),
    ("24 planning rules", "task-agnostic constraints", ORANGE),
    ("Back-chaining method", "goal-regression chain-of-thought", PURPLE),
    ("Correction feedback", "validator errors + sim trace + prev plan", RED),
]
lx0, lw_ = RX+1.5, 47
lh = 2.9; g = 0.65
top = YT-6.2
for i, (t, s, c) in enumerate(layers):
    y = top - i*(lh+g) - lh
    box(ax, lx0, y, lw_, lh, fc=tint(c, .92), ec=tint(c, .5), lw=1.3, rad=0.4)
    box(ax, lx0, y, 0.9, lh, fc=c, ec=c, rad=0.3, z=3)
    text(ax, lx0+2.0, y+lh*0.62, t, size=8.7, color=c, weight="bold", ha="left")
    text(ax, lx0+2.0, y+lh*0.24, s, size=7.3, color=SUB, ha="left")
    last_y = y
# technique chips filling the bottom-right space
text(ax, lx0, last_y-2.6, "Reliability techniques", size=8.8, color=INK, weight="bold", ha="left", font=HEAD)
tags = [("JSON-constrained output", BLUE), ("Structured self-correction", RED),
        ("Best-of-N sampling", ORANGE), ("Two-stage generation", PURPLE)]
cw2 = (lw_-2.0)/2
for i, (tag, c) in enumerate(tags):
    r, cc = divmod(i, 2)
    x = lx0 + cc*(cw2+2.0)
    y = last_y-5.0 - r*3.4
    box(ax, x, y-1.3, cw2, 2.6, fc=tint(c, .9), ec=tint(c, .5), lw=1.2, rad=0.5)
    ax.add_patch(Circle((x+1.6, y), 0.5, fc=c, ec="none", zorder=6))
    text(ax, x+2.9, y, tag, size=7.9, color=c, weight="bold", ha="left", va="center")

fig.savefig(os.path.join(os.path.dirname(__file__), "fig_skills.png"), dpi=220, facecolor="white", pad_inches=0)
print("saved skills")
