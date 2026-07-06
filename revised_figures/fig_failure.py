"""NEW slide: Failure Detection & Recovery (feedback #3)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from figtools import *
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

W, Hin = 12.4, 5.55
fig, ax = new_ax(W, Hin)
YT = H(W, Hin); ax.set_ylim(0, YT)

def head(x, num, title, sub, color):
    ax.add_patch(Circle((x+1.8, YT-2.3), 1.5, fc=color, ec="none", zorder=6))
    text(ax, x+1.8, YT-2.3, num, size=11, color="white", weight="bold", z=7)
    text(ax, x+3.8, YT-1.9, title, size=11.5, color=color, weight="bold", ha="left", font=HEAD)
    text(ax, x+3.8, YT-3.4, sub, size=8.2, color=SUB, ha="left", style="italic")

def chip(x, y, w, s, c, mono=False):
    box(ax, x, y-1.15, w, 2.3, fc=tint(c, .9), ec=tint(c, .5), lw=1.1, rad=0.4)
    text(ax, x+w/2, y, s, size=7.6, color=c, weight="bold", font=("Consolas" if mono else BODY))

# ================= LEFT: Detection =================
LX = 1.0
head(LX, "1", "Detection", "static + dynamic checks flag failures", BLUE)
# validator card
vx, vy, vw, vh = LX+1.5, YT-16.5, 38, 10.5
box(ax, vx, vy, vw, vh, fc=tint(BLUE, .95), ec=tint(BLUE, .5), lw=1.4, rad=0.6)
text(ax, vx+2.0, vy+vh-1.7, "Static validator", size=9.2, color=BLUE_DK, weight="bold", ha="left")
text(ax, vx+vw-2.0, vy+vh-1.7, "typed errors", size=7.4, color=SUB, ha="right", style="italic")
errs = ["invalid_capability", "unsupported_precondition", "missing_sync_condition", "cyclic_dependency"]
for i, e in enumerate(errs):
    r, c = divmod(i, 2)
    chip(vx+2.0 + c*(vw/2-1.0), vy+vh-4.6 - r*3.2, vw/2-3.0, e, BLUE_DK, mono=True)
# simulator card
sx, sy, sw, sh = LX+1.5, YT-28.5, 38, 9.5
box(ax, sx, sy, sw, sh, fc=tint(PURPLE, .95), ec=tint(PURPLE, .5), lw=1.4, rad=0.6)
text(ax, sx+2.0, sy+sh-1.7, "Tick-based simulator", size=9.2, color=PURPLE, weight="bold", ha="left")
for i, (e, sub) in enumerate([("deadlock", "state unchanged, tree still RUNNING"), ("timeout", "max_ticks exhausted")]):
    yy = sy+sh-4.6 - i*3.0
    chip(sx+2.0, yy, 11.0, e, PURPLE)
    text(ax, sx+14.5, yy, sub, size=7.2, color=SUB, ha="left")
# feedback output
fy = sy-4.0
box(ax, LX+6.5, fy-2.0, 28, 4.0, fc=tint(ORANGE, .9), ec=ORANGE, lw=1.4, rad=0.5)
text(ax, LX+20.5, fy, "structured failure feedback", size=8.6, color=ORANGE, weight="bold")
arrow(ax, vx+vw/2-8, vy, LX+20.5, fy+2.1, color=SUB, lw=1.5, mut=10, rad=-0.15)
arrow(ax, sx+vw/2-8, sy, LX+20.5, fy+2.1, color=SUB, lw=1.5, mut=10, rad=-0.1)

# divider
line(ax, 42.5, 2.0, 42.5, YT-1.5, color=LINE, lw=1.2)

# ================= RIGHT: Recovery ladder =================
RX = 44.5
head(RX, "2", "Recovery ladder", "escalate only if the cheaper fix fails", GREEN_DK)
lx0, lw_ = RX+1.5, 50

def tier(y, h, n, title, color, bullets):
    box(ax, lx0, y, lw_, h, fc=tint(color, .93), ec=color, lw=1.7, rad=0.6)
    ax.add_patch(Circle((lx0+2.6, y+h-2.3), 1.55, fc=color, ec="none", zorder=6))
    text(ax, lx0+2.6, y+h-2.3, n, size=12, color="white", weight="bold", z=7)
    text(ax, lx0+4.9, y+h-2.3, title, size=10.0, color=color, weight="bold", ha="left")
    for i, b in enumerate(bullets):
        text(ax, lx0+2.4, y+h-5.0 - i*2.35, "•  " + b, size=7.9, color=INK, ha="left")

# cross arrow from detection feedback into Tier 1
t1y, t1h = 28.3, 11.0
arrow(ax, 36.5, fy, lx0-0.4, t1y+t1h*0.5, color=ORANGE, lw=2.0, mut=13, rad=-0.12)

tier(t1y, t1h, "1", "Retry  —  same robot", BLUE,
     ["re-attempt the failed action on the same robot",
      "runtime failure oracle decides success / failure",
      "up to max_retries rounds before escalating"])
# self-loop refresh hint on Tier 1
ic_refresh(ax, lx0+lw_-2.6, t1y+t1h-2.3, 1.15, BLUE)
# escalation arrow
arrow(ax, lx0+lw_/2, t1y-0.2, lx0+lw_/2, 23.8, color=RED, lw=2.0, mut=13)
text(ax, lx0+lw_/2+1.0, 25.5, "if still failing", size=7.8, color=RED, ha="left", style="italic")

t2y, t2h = 12.5, 11.0
tier(t2y, t2h, "2", "Reassign  —  ask another robot", ORANGE,
     ["pick another robot whose capability produces the effect",
      "found via candidate_producers over add-effects",
      "move the action to it and re-run execution"])

# outcome: re-verify -> goal reached
oy = 5.6
arrow(ax, lx0+lw_/2, t2y-0.3, lx0+lw_/2, oy+2.3, color=GREEN_DK, lw=1.8, mut=12)
text(ax, lx0+lw_/2+1.2, (t2y+oy)/2+0.3, "re-verify: validate + simulate", size=7.4, color=GREEN_DK, ha="left", style="italic")
box(ax, lx0+lw_/2-13, oy-2.1, 26, 4.2, fc=tint(GREEN, .88), ec=GREEN_DK, lw=1.5, rad=0.5)
check(ax, lx0+lw_/2-9.5, oy, r=1.2, color=GREEN)
text(ax, lx0+lw_/2+1.5, oy, "goal reached", size=9.2, color=GREEN_DK, weight="bold")

fig.savefig(os.path.join(os.path.dirname(__file__), "fig_failure.png"), dpi=220, facecolor="white", pad_inches=0)
print("saved failure")
