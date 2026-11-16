"""Flow fields of the three decision toy systems, linear vs nonlinear iSSM (highest stim rate, best seed).

python -m scripts.figures.figure_decision_models --fmt png
"""

import argparse
import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.io import add_root, load_eval_data, load_best_seed_flow_fns, find_stim_job
from src.utils.plotting import plot_flow_fields
from src.utils.style import (
    PANEL_TITLE_SIZE, TEXT_WIDTH,
    LINEAR_COLOR, NONLINEAR_COLOR, TRUE_FLOW_COLOR, TRUE_TRAJ_COLOR,
    VERDICT_OK_COLOR, VERDICT_BAD_COLOR, VERDICT_MARKS, VERDICT_FONT,
    apply_paper_style, save_figure,
)
from scripts.figures._common.constants import (
    DECISION_MODELS as MODELS,
    DECISION_MODEL_LABELS as MODEL_LABELS,
)


# ── constants ────────────────────────────────────────────────────────────────

VARIANT_COLORS = {"linear": LINEAR_COLOR, "nonlinear": NONLINEAR_COLOR}

ROW_SEP_COLOR = "0.7"   # rule under each group heading

# Trajectory starts per system; the Saddle's two lie on opposite sides of the separatrix.
TRAJ_STARTS = {
    "PointAttractor": np.array([[2.5, 2.5]]),
    "LineAttractor":  np.array([[0.0, 2.5]]),
    "Saddle":         np.array([[-0.3, 1.5], [0.3, -1.5]]),
}

# One row of six square cells (linear group, then nonlinear). In inches.
FLOW_SIDE_IN        = 0.82   # nominal cell side
FLOW_GAP_IN         = 0.07   # min gap between cells within a group
FLOW_GROUP_GAP_IN   = 0.30   # gap between the two groups
FLOW_MARGIN_IN      = 0.03   # min outer margin
FLOW_LABEL_CLEAR_IN = 0.06   # min white space between neighbouring captions
FLOW_MARK_PAD_IN    = 0.03   # caption text -> tick/cross
FLOW_CAPTION_GAP_IN = 0.05   # cell bottom -> caption top
FLOW_RULE_GAP_IN    = 0.05   # cell top -> group rule -> group heading
FLOW_CAPTION_DESCENT = 0.25  # fraction of the caption height below its baseline
FLOW_TRAJ_LW        = 1.2    # points
FLOW_TRAJ_MARK      = 5.0

# Tick/cross captions: does the learned flow qualitatively reproduce the true one?
FLOW_VERDICT = {
    "linear":    {"PointAttractor": True, "LineAttractor": True, "Saddle": False},
    "nonlinear": {"PointAttractor": True, "LineAttractor": True, "Saddle": True},
}


# ── data ─────────────────────────────────────────────────────────────────────

def eval_dir_for(base_dir: str, model: str, key: str) -> Path | None:
    """Eval dir of the max-stim job for one (system, variant), or None if absent."""
    job = find_stim_job(Path(base_dir) / model / key, "max")
    if job is None:
        return None
    ev = job / "eval"
    return ev if (ev / "metrics.csv").exists() else None


# ── flow-field figure ────────────────────────────────────────────────────────

def _flow_lims(x_true: np.ndarray) -> tuple[float, float]:
    lo = float(np.asarray(x_true).min())
    hi = float(np.asarray(x_true).max())
    return lo, hi


def _draw_flow_cell(ax, base_dir: str, device: torch.device, model: str, key: str,
                    traj_lw: float = 1.5, traj_end_size: float = 7.0) -> bool:
    """Draw the true and learned flow and trajectories for one (system, variant); False if no data."""
    ev = eval_dir_for(base_dir, model, key)
    true_fn = learned_fn = None
    if ev is not None:
        true_fn, learned_fn, x_true = load_best_seed_flow_fns(
            load_eval_data(str(ev)), device)
    if true_fn is None or learned_fn is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "(no flow)", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return False
    lo, hi = _flow_lims(x_true)
    color = VARIANT_COLORS[key]
    plot_flow_fields([true_fn, learned_fn],
                     colors=[TRUE_FLOW_COLOR, color], ax=ax,
                     xlim=(lo, hi), ylim=(lo, hi),
                     n_points=10, max_arrow_length=0.4 * 0.8,
                     alphas=[1.0, 0.75], widths=[0.01, 0.01],
                     trajectory_starts=TRAJ_STARTS[model],
                     trajectory_colors=[TRUE_TRAJ_COLOR, color],
                     trajectory_steps=150, trajectory_dt=0.1,
                     trajectory_lw=traj_lw, trajectory_end_size=traj_end_size)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    return True


def _center_ink_horizontally(fig) -> None:
    """Equalise the left and right margins around everything the figure draws.

    Inset axes are skipped: their locator ties them to their parent.
    """
    fig.canvas.draw()
    figw = fig.get_figwidth()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())     # inches
    dx = 0.5 * ((figw - bb.x1) - bb.x0) / figw            # figure fraction
    if abs(dx) < 1e-4:
        return
    for ax in fig.axes:
        if ax.get_axes_locator() is not None:
            continue
        p = ax.get_position()
        ax.set_position([p.x0 + dx, p.y0, p.width, p.height])
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((x + dx, y))
    for a in fig.artists:
        if isinstance(a, mlines.Line2D):
            a.set_xdata([x + dx for x in a.get_xdata()])
        elif isinstance(a, mpatches.Rectangle):
            a.set_x(a.get_x() + dx)


def _text_extent_in(text: str, **kw) -> tuple[float, float]:
    """(width, height) of text in inches, measured on a throwaway canvas."""
    fig = plt.figure(figsize=(1, 1))
    t = fig.text(0, 0, text, **kw)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    w, h = bb.width / fig.dpi, bb.height / fig.dpi
    plt.close(fig)
    return w, h


def _solve_flow_row(label_w: float) -> tuple[float, float, float]:
    """(side, gap, margin) in inches for the six-panel row, given the widest label.

    Labels wider than a cell widen the gaps and margins, which shrinks the cells, so
    iterate to the fixed point.
    """
    side = FLOW_SIDE_IN
    for _ in range(50):
        gap = max(FLOW_GAP_IN, label_w + FLOW_LABEL_CLEAR_IN - side)
        margin = max(FLOW_MARGIN_IN, 0.5 * (label_w - side))
        side = (TEXT_WIDTH - 2 * margin - 4 * gap - FLOW_GROUP_GAP_IN) / 6
    return side, gap, margin


def _panel_caption(fig, cx: float, y_base: float, model: str, ok: bool) -> None:
    """System name and tick/cross, centred on baseline y_base.

    Two Text artists: the mark needs a different font and its own colour.
    """
    figw = fig.get_figwidth()
    kw = dict(va="baseline", fontsize=PANEL_TITLE_SIZE)
    name = fig.text(0, y_base, MODEL_LABELS[model], ha="left", **kw)
    mark = fig.text(0, y_base, VERDICT_MARKS[ok], ha="left", fontfamily=VERDICT_FONT,
                    color=VERDICT_OK_COLOR if ok else VERDICT_BAD_COLOR, **kw)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    w_name, w_mark = (t.get_window_extent(rend).width / fig.dpi for t in (name, mark))
    total = w_name + FLOW_MARK_PAD_IN + w_mark
    x0 = cx - 0.5 * total / figw
    name.set_position((x0, y_base))
    mark.set_position((x0 + (w_name + FLOW_MARK_PAD_IN) / figw, y_base))


def make_flow_figure(base_dir: str, device: torch.device) -> plt.Figure:
    """One row of six flow fields (linear group, then nonlinear), true flow in gray."""
    figw = TEXT_WIDTH
    groups = [("Linear", "linear"), ("Nonlinear", "nonlinear")]

    # Solve the row geometry from the caption widths, then the height from the rest.
    mark_w, mark_h = _text_extent_in(VERDICT_MARKS[False], fontsize=PANEL_TITLE_SIZE,
                                     fontfamily=VERDICT_FONT)
    cap_w, cap_h = 0.0, mark_h
    for model in MODELS:
        w, h = _text_extent_in(MODEL_LABELS[model], fontsize=PANEL_TITLE_SIZE)
        cap_w = max(cap_w, w + FLOW_MARK_PAD_IN + mark_w)
        cap_h = max(cap_h, h)
    head_h = _text_extent_in("Nonlinear", fontsize=PANEL_TITLE_SIZE)[1]
    side, gap, margin = _solve_flow_row(cap_w)

    y0_in = FLOW_MARGIN_IN + cap_h + FLOW_CAPTION_GAP_IN          # panel bottoms
    figh = y0_in + side + 2 * FLOW_RULE_GAP_IN + head_h + FLOW_MARGIN_IN
    fig = plt.figure(figsize=(figw, figh))

    # Panel left edges: three per group, with FLOW_GROUP_GAP_IN between the groups.
    x_in = [margin + gi * (3 * side + 2 * gap + FLOW_GROUP_GAP_IN) + ci * (side + gap)
            for gi in range(len(groups)) for ci in range(len(MODELS))]

    for i, (gi, ci) in enumerate([(g, c) for g in range(len(groups))
                                  for c in range(len(MODELS))]):
        model, key = MODELS[ci], groups[gi][1]
        ax = fig.add_axes([x_in[i] / figw, y0_in / figh, side / figw, side / figh])
        _draw_flow_cell(ax, base_dir, device, model, key,
                        traj_lw=FLOW_TRAJ_LW, traj_end_size=FLOW_TRAJ_MARK)
        _panel_caption(fig, (x_in[i] + side / 2) / figw,
                       (FLOW_MARGIN_IN + FLOW_CAPTION_DESCENT * cap_h) / figh,
                       model, FLOW_VERDICT[key][model])

    # Group heading over a grey rule spanning that group's panels.
    y_rule = (y0_in + side + FLOW_RULE_GAP_IN) / figh
    for gi, (name, _) in enumerate(groups):
        left, right = x_in[3 * gi], x_in[3 * gi + 2] + side
        fig.add_artist(mlines.Line2D([left / figw, right / figw], [y_rule, y_rule],
                                     color=ROW_SEP_COLOR, lw=1.0,
                                     transform=fig.transFigure, clip_on=False))
        fig.text(0.5 * (left + right) / figw, y_rule + FLOW_RULE_GAP_IN / figh,
                 name, ha="center", va="baseline", fontsize=PANEL_TITLE_SIZE)

    _center_ink_horizontally(fig)
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", default=None,
                   help="Base dir of decision-model outputs "
                        "(default: outputs/decision_models)")
    p.add_argument("--output_dir", default=None,
                   help="Output dir for figures (default: figures/decision_models)")
    p.add_argument("--fmt", default="png", help="Figure format: png, svg, pdf (default: png)")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_paper_style()
    base_dir   = add_root(args.base_dir or os.path.join("outputs", "decision_models"))
    output_dir = add_root(args.output_dir or os.path.join("figures", "decision_models"))
    fmt    = args.fmt.lstrip(".")
    device = torch.device(args.device)
    os.makedirs(output_dir, exist_ok=True)

    fig_flow = make_flow_figure(base_dir, device)
    p_flow = os.path.join(output_dir, f"decision_models_flows.{fmt}")
    save_figure(fig_flow, p_flow, fmt=fmt)
    plt.close(fig_flow)
    print(f"Saved: {p_flow}")


if __name__ == "__main__":
    main()
