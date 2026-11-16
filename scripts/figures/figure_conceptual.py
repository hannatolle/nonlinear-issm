"""Conceptual 2x2 identifiability grid (linear/nonlinear model x observational/interventional data).

python -m scripts.figures.figure_conceptual --fmt png
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils.io import add_root
from src.utils.style import LINEAR_COLOR, NONLINEAR_COLOR

_TRUE_COLOR = "#333333"
_SURF_COLOR = "#ffffff"

_A1 = np.array([-1.5, 0.0])
_A2 = np.array([ 1.5, 0.0])
_DEPTH = 0.6
_SIGMA = 0.75

# Spiral into A1 (shared by true traj and obs traj)
_SP1_R0    = 1
_SP1_DECAY = 0.1
_SP1_TURNS = 2.5
_SP1_PHASE = np.pi / 2

# Spiral into A2 (true traj only)
_SP2_R0    = 1
_SP2_DECAY = 0.2
_SP2_TURNS = 2.0
_SP2_PHASE = np.pi

# Linear interventional spiral (wide ellipse converging to centre)
_LI_DECAY   = 0.15
_LI_TURNS   = 3
_LI_YSQUISH = 0.5    # squishes y → elliptic
# r0 and phase derived so the spiral starts at the same point as the other trajectories:
#   x(0) = r0*cos(phase) = start_x,  y(0) = r0*sin(phase)*ysquish = start_y
_li_sx  = _A1[0] + _SP1_R0 * np.cos(_SP1_PHASE)
_li_sy  = _A1[1] + _SP1_R0 * np.sin(_SP1_PHASE)
_LI_R0    = np.sqrt(_li_sx**2 + (_li_sy / _LI_YSQUISH)**2)
_LI_PHASE = np.arctan2(_li_sy / _LI_YSQUISH, _li_sx)

# Start-marker size (scatter units)
_MARKER_SIZE = 40


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------

def _z(X, Y):
    """Double-well potential: sum of two negative Gaussians."""
    def well(c):
        return -_DEPTH * np.exp(-((X - c[0])**2 + (Y - c[1])**2) / (2 * _SIGMA**2))
    return well(_A1) + well(_A2)


def _lift(x, y, eps=0.07):
    """Lift 2D parametric curve onto the manifold surface with a small offset."""
    return x, y, _z(x, y) + eps


def _draw_manifold(ax):
    xv = np.linspace(-3.0, 3.0, 45)
    yv = np.linspace(-2.2, 2.2, 45)
    X, Y = np.meshgrid(xv, yv)
    Z = _z(X, Y)
    ax.plot_surface(X, Y, Z, color=_SURF_COLOR, alpha=0.35, linewidth=0, antialiased=True)
    ax.plot_wireframe(X, Y, Z, color="#aaaaaa", linewidth=0.35, alpha=0.55, rstride=3, cstride=3)


# ---------------------------------------------------------------------------
# Trajectory building blocks
# ---------------------------------------------------------------------------

def _spiral(center, n_turns, r0, decay, n=200, phase=0.0):
    t = np.linspace(0, n_turns * 2 * np.pi, n)
    r = r0 * np.exp(-decay * t)
    return center[0] + r * np.cos(t + phase), center[1] + r * np.sin(t + phase)


def _hop(p1, p2, n=70):
    t = np.linspace(0, 1, n)
    s = 1 / (1 + np.exp(-12 * (t - 0.5)))
    return p1[0] + (p2[0] - p1[0]) * s, p1[1] + (p2[1] - p1[1]) * s


# ---------------------------------------------------------------------------
# Named trajectories
# ---------------------------------------------------------------------------

def _true_traj():
    """Spiral into A1, hop to A2, spiral into A2."""
    x1, y1 = _spiral(_A1, n_turns=_SP1_TURNS, r0=_SP1_R0, decay=_SP1_DECAY, phase=_SP1_PHASE)
    # hop ends where spiral-2 starts: A2 + _SP2_R0 * [cos(_SP2_PHASE), sin(_SP2_PHASE)]
    xh, yh = _hop(np.array([x1[-1], y1[-1]]),
                  _A2 + _SP2_R0 * np.array([np.cos(_SP2_PHASE), np.sin(_SP2_PHASE)]))
    x2, y2 = _spiral(_A2, n_turns=_SP2_TURNS, r0=_SP2_R0, decay=_SP2_DECAY, phase=_SP2_PHASE)
    return np.concatenate([x1, xh, x2]), np.concatenate([y1, yh, y2])


def _obs_traj():
    """Single-attractor trajectory: stuck in A1, never visits A2."""
    return _spiral(_A1, n_turns=_SP1_TURNS, r0=_SP1_R0, decay=_SP1_DECAY, phase=_SP1_PHASE)


def _linear_interv_traj():
    """Wide elliptic inward spiral centred at midpoint (0, 0).

    A linear model with interventions sees data near both attractors but can only
    produce a single fixed point, so it compromises at the centre.
    """
    t = np.linspace(0, _LI_TURNS * 2 * np.pi, 450)
    r = _LI_R0 * np.exp(-_LI_DECAY * t)
    return r * np.cos(t + _LI_PHASE), r * np.sin(t + _LI_PHASE) * _LI_YSQUISH


# ---------------------------------------------------------------------------
# Axes setup and drawing
# ---------------------------------------------------------------------------

def _setup_ax(ax):
    _draw_manifold(ax)
    ax.view_init(elev=58, azim=-30)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.2, 2.2)
    ax.set_zlim(-1.35, 0.55)
    ax.set_axis_off()


def _draw_traj(ax, x, y, color, lw=1.6, alpha=1.0, **kwargs):
    px, py, pz = _lift(x, y)
    ax.plot(px, py, pz, color=color, linewidth=lw, alpha=alpha, **kwargs)


def _mark_start(ax, x, y, color):
    px, py, pz = _lift(np.array([x[0]]), np.array([y[0]]))
    ax.scatter(px, py, pz, color=color, s=_MARKER_SIZE, zorder=10,
               edgecolors='white', linewidths=0.6, depthshade=False)


# ---------------------------------------------------------------------------
# Figure assembly helpers
# ---------------------------------------------------------------------------

_ADJ = dict(left=0.04, right=0.99, top=0.93, bottom=0.09, hspace=0.02, wspace=0.02)


def _add_inner_grid_lines(fig):
    """Draw light gray separator lines between the inner grid cells."""
    L, R, T, B = _ADJ['left'], _ADJ['right'], _ADJ['top'], _ADJ['bottom']
    hs, ws = _ADJ['hspace'], _ADJ['wspace']
    axw = (R - L) / (2 + ws)
    axh = (T - B) / (2 + hs)
    x_sep = L + axw + 0.5 * ws * axw   # midpoint of horizontal gap
    y_sep = B + axh + 0.5 * hs * axh   # midpoint of vertical gap
    kw = dict(color='#cccccc', lw=0.8, transform=fig.transFigure, clip_on=False)
    fig.add_artist(Line2D([x_sep, x_sep], [B, T], **kw))
    fig.add_artist(Line2D([L, R],         [y_sep, y_sep], **kw))


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def make_figure():
    xt, yt   = _true_traj()
    xo, yo   = _obs_traj()
    xli, yli = _linear_interv_traj()

    # (subplot_idx, model_x, model_y, model_color)
    panels = [
        (1, xo,  yo,  LINEAR_COLOR),   # Linear    + Observational
        (2, xli, yli, LINEAR_COLOR),   # Linear    + Interventional
        (3, xo,  yo,  NONLINEAR_COLOR),   # Nonlinear + Observational
        (4, xt,  yt,  NONLINEAR_COLOR),   # Nonlinear + Interventional
    ]

    fig = plt.figure(figsize=(4, 4))

    for idx, mx, my, mc in panels:
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        _setup_ax(ax)
        # True trajectory in every panel
        _draw_traj(ax, xt, yt, color=_TRUE_COLOR, lw=1.5, alpha=0.7,
                   linestyle='-', zorder=3)
        _mark_start(ax, xt, yt, _TRUE_COLOR)
        # Model trajectory
        _draw_traj(ax, mx, my, color=mc, lw=2, alpha=1.0, zorder=5)
        _mark_start(ax, mx, my, mc)

    # Column headers
    for col, label in enumerate(["Observational", "Interventional"]):
        fig.text(0.27 + col * 0.50, 0.97, label,
                 ha='center', va='top', fontsize=10)

    # Row labels (rotated, left margin)
    for row, label in enumerate(["Linear", "Nonlinear"]):
        fig.text(0.01, 0.74 - row * 0.45, label,
                 ha='left', va='center', fontsize=10, rotation=90)

    plt.subplots_adjust(**_ADJ)
    _add_inner_grid_lines(fig)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Conceptual identifiability grid figure")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: figures/conceptual/)")
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    out_dir = add_root(args.output_dir or os.path.join("figures", "conceptual"))
    os.makedirs(out_dir, exist_ok=True)

    fig = make_figure()
    out = os.path.join(out_dir, f"identifiability_grid.{args.fmt}")
    fig.savefig(out, bbox_inches='tight', dpi=150 if args.fmt == "png" else None)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
