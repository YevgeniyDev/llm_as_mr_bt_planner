"""Shared toolkit for clean, publication-style scientific figures (no cartoon icons).

Everything is drawn from geometric primitives: rounded boxes, arrows, small graphs,
mathtext labels, monospace code. Palette + fonts match the slide deck (Cambria titles,
Calibri body, blue/green/orange/purple accents on white)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, RegularPolygon
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

# ---- palette (sampled from the deck's diagrams) ----
INK      = "#1A2230"   # near-black text
SUB      = "#5B6472"   # secondary text
BLUE     = "#2563EB"
BLUE_DK  = "#14315D"
GREEN    = "#1E9E6A"
GREEN_DK = "#166B49"
ORANGE   = "#E07B1A"
PURPLE   = "#7C3AED"
RED      = "#D14D4D"
GRAY     = "#8A93A2"
LINE     = "#C7CDD6"    # hairline separators / arrows
PANEL    = "#F4F6F9"    # very light panel fill
WHITE    = "#FFFFFF"

BODY = "Calibri"
HEAD = "Cambria"

def tint(hexc, frac):
    """Blend a hex color toward white by frac in [0,1]."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * frac); g = int(g + (255 - g) * frac); b = int(b + (255 - b) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"

def new_ax(w, h, dpi=220):
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax

def H(w, h):  # convenient world-height for a given aspect (xlim always 0..100)
    return 100 * h / w

def box(ax, x, y, w, h, fc="white", ec=LINE, lw=1.4, rad=0.4, z=2, alpha=1.0, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha, ls=ls,
                       mutation_aspect=1)
    ax.add_patch(p)
    return p

def sq(ax, x, y, w, h, fc="white", ec=LINE, lw=1.4, z=2, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))

def text(ax, x, y, s, size=11, color=INK, weight="normal", style="normal",
         ha="center", va="center", font=BODY, z=5, rot=0):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, fontstyle=style,
                   ha=ha, va=va, family=font, zorder=z, rotation=rot)

def arrow(ax, x1, y1, x2, y2, color=SUB, lw=1.8, z=3, style="-|>", mut=12, ls="-",
          rad=0.0, alpha=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=mut,
                        lw=lw, color=color, zorder=z, ls=ls, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=0, shrinkB=0, capstyle="round", joinstyle="round")
    ax.add_patch(a)
    return a

def line(ax, x1, y1, x2, y2, color=LINE, lw=1.2, z=2, ls="-"):
    ax.add_line(Line2D([x1, x2], [y1, y2], color=color, lw=lw, zorder=z, ls=ls,
                       solid_capstyle="round"))

def node(ax, x, y, r, label="", fc="white", ec=BLUE, lw=1.6, tcolor=INK, tsize=9, z=4):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))
    if label:
        text(ax, x, y, label, size=tsize, color=tcolor, weight="bold", z=z + 1)

def check(ax, cx, cy, r=1.5, color=GREEN, z=6):
    """A filled disc with a clean white check mark (drawn, not emoji)."""
    ax.add_patch(Circle((cx, cy), r, fc=color, ec="none", zorder=z))
    s = r
    ax.add_line(Line2D([cx - 0.45 * s, cx - 0.08 * s, cx + 0.5 * s],
                       [cy + 0.02 * s, cy - 0.4 * s, cy + 0.45 * s],
                       color="white", lw=1.7 * r, zorder=z + 1,
                       solid_capstyle="round", solid_joinstyle="round"))

def glyph(ax, kind, x, y, s, color, z=4):
    """Abstract object glyphs used in planning figures (NOT cartoons)."""
    if kind == "square":
        ax.add_patch(Rectangle((x - s, y - s), 2 * s, 2 * s, fc=tint(color, .55), ec=color, lw=1.3, zorder=z))
    elif kind == "circle":
        ax.add_patch(Circle((x, y), s, fc=tint(color, .55), ec=color, lw=1.3, zorder=z))
    elif kind == "triangle":
        ax.add_patch(RegularPolygon((x, y), 3, radius=s * 1.25, fc=tint(color, .55), ec=color, lw=1.3, zorder=z))
    elif kind == "diamond":
        ax.add_patch(RegularPolygon((x, y), 4, radius=s * 1.3, fc=tint(color, .55), ec=color, lw=1.3, zorder=z))

def robot_chip(ax, x, y, w, h, label, color):
    """A robot represented as a labelled capability chip (no cartoon face)."""
    box(ax, x, y, w, h, fc=tint(color, .82), ec=color, lw=1.5, rad=0.35)
    text(ax, x + w / 2, y + h / 2, label, size=9.5, color=color, weight="bold", font=BODY)

import numpy as _np
from matplotlib.patches import Arc as _Arc
def ic_target(ax, x, y, s, c):
    for rr in (s, s*0.62, s*0.28):
        ax.add_patch(Circle((x, y), rr, fc="none", ec=c, lw=1.4, zorder=6))
    ax.add_patch(Circle((x, y), s*0.09, fc=c, ec="none", zorder=6))
def ic_shield(ax, x, y, s, c):
    pts = [(x, y+s), (x+0.8*s, y+0.45*s), (x+0.8*s, y-0.35*s), (x, y-s), (x-0.8*s, y-0.35*s), (x-0.8*s, y+0.45*s)]
    ax.add_patch(plt.Polygon(pts, closed=True, fc="none", ec=c, lw=1.5, zorder=6))
    ax.add_line(Line2D([x-0.36*s, x-0.08*s, x+0.42*s], [y+0.02*s, y-0.28*s, y+0.32*s], color=c, lw=1.6, zorder=7, solid_capstyle="round"))
def ic_warn(ax, x, y, s, c):
    ax.add_patch(RegularPolygon((x, y-0.1*s), 3, radius=s*1.15, orientation=0, fc="none", ec=c, lw=1.5, zorder=6))
    ax.add_line(Line2D([x, x], [y+0.42*s, y-0.18*s], color=c, lw=1.6, zorder=7, solid_capstyle="round"))
    ax.add_patch(Circle((x, y-0.5*s), 0.12*s, fc=c, ec="none", zorder=7))
def ic_refresh(ax, x, y, s, c):
    ax.add_patch(_Arc((x, y), 1.7*s, 1.7*s, theta1=60, theta2=330, lw=1.6, color=c, zorder=6))
    ang = _np.deg2rad(60); hx, hy = x+0.85*s*_np.cos(ang), y+0.85*s*_np.sin(ang)
    ax.add_patch(RegularPolygon((hx, hy), 3, radius=0.28*s, orientation=_np.deg2rad(-30), fc=c, ec="none", zorder=7))
def ic_clock(ax, x, y, s, c):
    ax.add_patch(Circle((x, y), s, fc="none", ec=c, lw=1.5, zorder=6))
    ax.add_line(Line2D([x, x], [y, y+0.6*s], color=c, lw=1.5, zorder=7, solid_capstyle="round"))
    ax.add_line(Line2D([x, x+0.42*s], [y, y+0.08*s], color=c, lw=1.5, zorder=7, solid_capstyle="round"))
def ic_link(ax, x, y, s, c):
    for dx in (-0.42*s, 0.42*s):
        ax.add_patch(_Arc((x+dx, y), 1.0*s, 0.7*s, lw=1.5, color=c, zorder=6))
    ax.add_line(Line2D([x-0.32*s, x+0.32*s], [y, y], color=c, lw=1.5, zorder=7))
def ic_gear(ax, x, y, s, c, teeth=8):
    for k in range(teeth):
        a = 2*_np.pi*k/teeth
        ax.add_patch(RegularPolygon((x+0.95*s*_np.cos(a), y+0.95*s*_np.sin(a)), 4, radius=0.28*s, orientation=a, fc=c, ec="none", zorder=6))
    ax.add_patch(Circle((x, y), 0.7*s, fc="none", ec=c, lw=1.5, zorder=7))
    ax.add_patch(Circle((x, y), 0.28*s, fc="none", ec=c, lw=1.3, zorder=7))

def section_tag(ax, x, y, num, color):
    """Numbered tier tag: a filled disc with the tier number + a label."""
    ax.add_patch(Circle((x, y), 1.9, fc=color, ec="none", zorder=6))
    text(ax, x, y, str(num), size=12, color="white", weight="bold", z=7)
