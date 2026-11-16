"""Panels and layout shared by figure_realdata and figure_huang.

Figures are laid out in inches at TEXT_WIDTH and saved uncropped, so the geometry
constants below are physical sizes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch

from src.utils.io import load_sweep, available_dir, read_dt_from_dir
from src.utils.plotting import (
    coarse_axis_ticks, coarsest_axis_ticks,
    plot_multiple_y_predictions,
    plot_paired_tier_comparison,
    plot_sweep,
)
from src.utils.style import (
    panel_label, TEXT_WIDTH, LEGEND_SIZE, PANEL_TITLE_SIZE, TICK_LABEL_SIZE,
)
from scripts.figures._common.constants import (
    METRIC_LABELS_COMPACT, SERIES_LABELS, XLABEL_D_SHORT,
)
from scripts.figures._common.stats_export import compute_tier_stats

# Full label for the two-panel figures; narrower panels use XLABEL_D_SHORT.
XLABEL_D_MATH = "Latent dimension ($D$)"

# Figure geometry, in inches.
RIGHT_PAD   = 0.06
TOP_PAD     = 0.17     # panel labels
BOTTOM_PAD  = 0.40     # x label + tick labels
COL_GAP     = 0.50     # next column's y label + tick labels
ROW_GAP     = 0.50
LBL_DX      = 0.38     # panel label, inches left of its frame
LBL_DY      = 0.03     # panel label, inches above its frame
TITLE_PAD_PT = 3.0     # panel title baseline above its frame, in points
LBL_DY_TITLE = TITLE_PAD_PT / 72   # same, in inches, so labels share the title baseline

# Overview (2 rows x 4 columns, the time-series columns spanning both rows).
OV_ROW_H    = 1.12
OV_LEFT_PAD = 0.22     # time-series panel has no y axis

# Main figure: one row of four panels sharing top and bottom.
MAIN_PANEL_H  = 1.55
MAIN_LEFT_PAD = 0.20
MAIN_GAP_AB   = 0.30   # before a panel with no y axis
MAIN_GAP_BOX  = 0.44   # tick labels + next panel's label
MAIN_BOX_LBL_DX = 0.26 # boxplots use titles, not y labels
BARE_LBL_DX   = 0.16   # panels with no y axis
MAIN_BOTTOM_PAD = 0.46 # x label + tick labels

# Two-panel figures: fixed panel size, pair centred on the line.
SUP_PANEL_W   = 1.50
SUP_PANEL_H   = 1.30
SUP_LEFT_PAD  = 0.50   # y label + tick labels
SUP_COL_GAP   = 0.50   # next panel's y label + tick labels

TS_CHANNELS = 10
_TRACE_LW = 0.6
# Gap between normalised channels, as a fraction of a channel's height.
_NORM_MARGIN = 0.10
# Gap between the bottom trace and the key below it, as a fraction of the panel.
_TS_KEY_CLEAR = 0.04

# Mark and text sizes for the boxplot panels at TEXT_WIDTH.
_TIER_POINT_SIZE = 3.0
_PAIR_LINE_LW    = 0.25
MEDIAN_LW        = 0.9
STAR_SIZE        = 7


# ── panels ───────────────────────────────────────────────────────────────────────

def load_example_npz(train_dir: "str | None", eval_subdir: str) -> "dict | None":
    """Load eval/example_plot_data.npz from a job dir, or None."""
    if not train_dir:
        return None
    p = Path(train_dir) / eval_subdir / "example_plot_data.npz"
    return dict(np.load(p)) if p.exists() else None


def panel_timeseries(
    ax, train_dir1: "str | None", train_dir2: "str | None", eval_subdir: str,
    keys: tuple, name1: str, name2: str, color1: str, color2: str,
    title: str = SERIES_LABELS["rollout"], channels=range(TS_CHANNELS),
    normalize: bool = False, legend: bool = False, mask_label: str = "Masked",
    time_unit: str = "ms", time_ticks=None, title_pad: "float | None" = None,
    legend_band: bool = True, legend_obs: bool = True,
    legend_kw: "dict | None" = None,
) -> bool:
    """Stacked example traces (keys = npz mean/std pair) for both models; False if no data.

    channels are the observation indices to stack (first at the bottom). time_ticks are
    in time_unit ("ms" or "s"). legend_band=False draws the key over the traces instead
    of in a band below them.
    """
    d1 = load_example_npz(train_dir1, eval_subdir)
    d2 = load_example_npz(train_dir2, eval_subdir)
    ref = d2 if d2 is not None else d1
    if ref is None:
        ax.set_visible(False)
        return False

    N = ref["y_true"].shape[1]
    dims = np.array([c for c in channels if c < N])
    u_t = torch.tensor(ref["u"], dtype=torch.float32).unsqueeze(0)
    y_t = torch.tensor(ref["y_true"][:, dims], dtype=torch.float32).unsqueeze(0)
    _mask_arr = ref.get("y_mask")
    mask_t = (torch.tensor(_mask_arr[:, dims], dtype=torch.bool).unsqueeze(0)
              if _mask_arr is not None else None)

    dt = float(ref["dt"]) if "dt" in ref else None
    if dt is None:
        for td in (train_dir2, train_dir1):
            if td:
                dt = read_dt_from_dir(td)
                break
    dt = dt or 1.0

    def _t(arr):
        return torch.tensor(arr[:, dims], dtype=torch.float32).unsqueeze(0)

    key_mean, key_std = keys
    entries = []
    for d, color, name in [(d1, color1, name1), (d2, color2, name2)]:
        if d is not None and key_mean in d:
            entries.append((_t(d[key_mean]), _t(d[key_std]), color, name))
    if not entries:
        ax.set_visible(False)
        return False

    plot_multiple_y_predictions(u_t, y_t, entries, trial_to_plot=0,
                                max_per_panel=len(dims), ax=ax, mask=mask_t,
                                normalize_channels=normalize,
                                channel_margin=(_NORM_MARGIN if normalize else 1.0),
                                obs_lw=_TRACE_LW, pred_lw=_TRACE_LW,
                                mask_label=mask_label)
    ax.set_title(title, pad=title_pad)
    # Ticks are in sample indices; dt is in seconds.
    scale = 1e3 if time_unit == "ms" else 1.0
    if time_ticks is not None:
        ax.set_xticks([t / (scale * dt) for t in time_ticks])
    if time_unit == "ms":
        fmt = lambda x, _, _dt=dt: f"{x * _dt * 1e3:.0f}"
    else:
        fmt = lambda x, _, _dt=dt: f"{x * _dt:.4g}"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt))
    ax.set_xlabel(f"Time ({time_unit})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _leg = ax.get_legend()
    if _leg is not None:
        _leg.remove()
    if legend:
        # Masked segments are already marked by the stim. line; the white frame hides
        # that line behind the key.
        handles, labels = ax.get_legend_handles_labels()
        drop = {mask_label} if legend_obs else {mask_label, "Obs."}
        keep = [(h, l) for h, l in zip(handles, labels) if l not in drop]
        style = dict(loc="lower right", ncol=len(keep),
                     frameon=True, facecolor="white", edgecolor="none",
                     framealpha=1.0, fancybox=False,
                     fontsize=LEGEND_SIZE, handlelength=0.8, handletextpad=0.4,
                     columnspacing=0.6, borderpad=0.0, borderaxespad=0.2)
        leg = ax.legend(*zip(*keep), **{**style, **(legend_kw or {})})
    if legend and legend_band:
        # Lower the y floor by the key's height so the key sits below the traces.
        fig = ax.figure
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        frac = (leg.get_window_extent(rend).height
                / ax.get_window_extent(rend).height) + _TS_KEY_CLEAR
        lo, hi = ax.get_ylim()
        ax.set_ylim(bottom=hi - (hi - lo) / max(1e-3, 1.0 - frac))
    return True


def panels_tier(
    axes, train_dir1: "str | None", train_dir2: "str | None", metrics: list,
    name1: str, name2: str, color1: str, color2: str,
    eval_subdir: str, name_suffix: str, xlabel: "str | None" = None,
    legend: bool = False, legend_loc: str = "lower right",
    legend_band: bool = False, tier_labels: "dict | None" = None,
) -> bool:
    """Per-tier paired boxplots, one metric per axis; False if no data.

    legend draws the model key in the first axis; legend_band lowers that axis's
    floor so the key does not overlap the data.
    """
    csv1 = Path(train_dir1) / eval_subdir / "multi_tier_eval_metrics.csv" if train_dir1 else None
    csv2 = Path(train_dir2) / eval_subdir / "multi_tier_eval_metrics.csv" if train_dir2 else None
    if not (csv1 and csv1.exists() and csv2 and csv2.exists()):
        for ax in axes:
            ax.set_visible(False)
        return False

    plot_paired_tier_comparison(
        pd.read_csv(csv1), pd.read_csv(csv2),
        name1=name1, name2=name2, color1=color1, color2=color2,
        metrics=metrics, metric_labels=METRIC_LABELS_COMPACT,
        axes=list(axes), show_titles=False,
        records=compute_tier_stats(train_dir1, train_dir2, name_suffix, eval_subdir),
        tier_labels=tier_labels,
        point_size=_TIER_POINT_SIZE, median_lw=MEDIAN_LW, star_size=STAR_SIZE,
        tick_label_size=TICK_LABEL_SIZE, legend_size=LEGEND_SIZE,
        title_size=PANEL_TITLE_SIZE, line_width=_PAIR_LINE_LW,
    )
    for ax, metric in zip(axes, metrics):
        ax.set_title("")
        ax.set_ylabel(METRIC_LABELS_COMPACT[metric])
        _leg = ax.get_legend()
        if _leg is not None:
            _leg.remove()
        # Make room for the significance asterisks above the data.
        stars = [t.get_position()[1] for t in ax.texts]
        if stars:
            lo, hi = ax.get_ylim()
            ax.set_ylim(top=max(hi, max(stars) + 0.07 * (hi - lo)))
        coarse_axis_ticks(ax, axis="y", decimals=1, min_ticks=3)
        if xlabel:
            ax.set_xlabel(xlabel)
    if legend:
        leg = axes[0].legend(
            handles=[mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.7, label=n)
                     for c, n in ((color1, name1), (color2, name2))],
            loc=legend_loc, frameon=False, fontsize=LEGEND_SIZE,
            handlelength=1.0, handletextpad=0.4, labelspacing=0.25,
            borderpad=0.0, borderaxespad=0.3)
        if legend_band:
            _open_legend_band(axes[0], leg)
    return True


def _open_legend_band(ax, leg) -> None:
    """Lower ax's floor by the key's height so the key sits below the data."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    frac = (leg.get_window_extent(rend).height
            / ax.get_window_extent(rend).height) + _TS_KEY_CLEAR
    lo, hi = ax.get_ylim()
    ax.set_ylim(bottom=hi - (hi - lo) / max(1e-3, 1.0 - frac))


def panels_sweep(
    axes, d_sw_1: "str | None", d_sw_2: "str | None", metrics: list,
    name1: str, name2: str, color1: str, color2: str,
    eval_subdir: str, out_dir: str, fmt: str, xlabel: str = XLABEL_D_SHORT,
    legend: bool = False, legend_loc: str = "lower right",
    significance: str = "markers", records: "list | None" = None,
) -> bool:
    """Latent-dimension sweep, one metric per axis; False if either sweep is missing.

    records are pre-corrected sweep tests (see compute_sweep_stats); if None,
    plot_sweep tests each panel itself.
    """
    if not (available_dir(d_sw_1) and available_dir(d_sw_2)):
        for ax in axes:
            ax.set_visible(False)
        return False

    kw = dict(config_param="settings.D", eval_subdir=eval_subdir)
    for metric, ax in zip(metrics, axes):
        r1 = load_sweep(d_sw_1, metric=metric, **kw)
        r2 = load_sweep(d_sw_2, metric=metric, **kw)
        plot_sweep([(r1, r2, metric, "solid")], "settings.D",
                   name1, name2, color1, color2,
                   output_dir=out_dir, fmt=fmt, ax=ax,
                   significance=significance, records=records)
        all_xs = sorted({int(r[0]) for r in (r1 or []) + (r2 or [])})
        if all_xs:
            ax.set_xticks(all_xs)
            ax.set_xticklabels([str(x) for x in all_xs])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_LABELS_COMPACT[metric])
        _leg = ax.get_legend()
        if _leg is not None:
            _leg.remove()
        coarsest_axis_ticks(ax, axis="y", min_ticks=3, max_decimals=2)
    if legend:
        axes[0].legend(
            handles=[mlines.Line2D([0], [0], color=c, lw=1.2, marker="o",
                                   markersize=2.6, label=n)
                     for c, n in ((color1, name1), (color2, name2))],
            loc=legend_loc, frameon=False, fontsize=LEGEND_SIZE,
            handlelength=1.1, handletextpad=0.4, labelspacing=0.2,
            borderpad=0.0, borderaxespad=0.3)
    return True


# ── layout helpers ───────────────────────────────────────────────────────────────

def ylabel_to_title(ax) -> None:
    """Move a panel's y label to its title (saves horizontal space)."""
    label = ax.get_ylabel()
    if label:
        ax.set_ylabel("")
        ax.set_title(label, fontsize=PANEL_TITLE_SIZE, pad=TITLE_PAD_PT)


def inch_axes(fig, fig_w: float, fig_h: float):
    """Closure placing axes in inches on a figure of the given size."""
    def _ax(x: float, y: float, w: float, h: float):
        return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
    return _ax


def label_panels(pairs, panel_h: float, dx: float = LBL_DX,
                 dy: float = LBL_DY) -> None:
    """Panel labels at a fixed physical offset from each frame; hidden panels are skipped.

    pairs is [(ax, letter, panel_w), ...]. Use dy=LBL_DY_TITLE for panels with titles.
    """
    for ax, letter, w in pairs:
        if ax.get_visible():
            panel_label(ax, letter, x=-dx / w, y=1 + dy / panel_h)


def two_panel_row(panel_w: float = SUP_PANEL_W, panel_h: float = SUP_PANEL_H):
    """One-row, two-panel figure at TEXT_WIDTH with the pair centred: (fig, [ax, ax], panel_w)."""
    fig_w = TEXT_WIDTH
    fig_h = TOP_PAD + panel_h + BOTTOM_PAD
    block = SUP_LEFT_PAD + 2 * panel_w + SUP_COL_GAP   # y label to last frame edge
    x0 = 0.5 * (fig_w - block) + SUP_LEFT_PAD
    fig = plt.figure(figsize=(fig_w, fig_h))
    _ax = inch_axes(fig, fig_w, fig_h)
    axes = [_ax(x0 + i * (panel_w + SUP_COL_GAP), BOTTOM_PAD, panel_w, panel_h)
            for i in range(2)]
    return fig, axes, panel_w
