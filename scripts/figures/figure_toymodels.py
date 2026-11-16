"""Toy-model figure (linear vs nonlinear): rollouts, latents, flow fields, B matrices and a D sweep.

Panels (a)-(d) use job_1 (D=2). Also writes toymodels_<dataset>_{stats,per_seed}.csv.

python -m scripts.figures.figure_toymodels --dataset fhn
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from src.utils.io import (
    add_root, available_dir, load_eval_data, load_best_seed_flow_fns, load_scale_aligned_B,
    load_sweep, localize_path,
)
from src.utils.plotting import (
    coarse_axis_ticks, draw_matrix_heatmap, plot_dual_mean_predictions,
    plot_flow_fields, plot_sweep, std_clipped_ylim,
)
from src.utils.style import (
    LINEAR_COLOR, NONLINEAR_COLOR, TRUE_FLOW_COLOR, TRUE_TRAJ_COLOR,
    TEXT_WIDTH, MATH_LABEL_SIZE, TICK_LABEL_SIZE, PANEL_TITLE_SIZE, STIM_ALPHA,
    STD_RANGE_FACTOR, apply_paper_style, panel_label, save_figure,
)
from scripts.figures._common.constants import (
    METRIC_LABELS, MODEL_LABELS, SERIES_LABELS, XLABEL_D_SHORT,
)
from scripts.figures._common.stats_export import (
    SWEEP_STATS_EXTRA, bh_stamp, job_metric_by_seed, jobs_by_x, sweep_panel_records,
    write_per_seed_csv, write_stats_csv,
)

DATASET_SYSTEMS = {"fhn": "FitzHughNagumo", "vdp": "VanDerPol"}

# ── constants ────────────────────────────────────────────────────────────────────

# Flow-field trajectory start (true-latent coords); unlisted systems use x_true[0].
FLOW_TRAJ_STARTS = {
    "FitzHughNagumo": np.array([0.5, -0.5]),
    "VanDerPol":      np.array([0.1, -0.1]),
}

# Figure geometry, in inches; axes are hand-placed so panels align across columns.
CH_W, CH_H = 1.5, 0.6          # one channel sub-panel of a time-series panel
CH_GAP     = 0.08 * CH_H       # vertical gap between channels

LEFT_PAD    = 0.52             # (a): y label + tick labels
GAP_AB      = 0.45             # (b): y label + tick labels (minimum)
RIGHT_PAD   = 0.10             # right of the B matrices / D sweep
TOP_PAD     = 0.18             # panel titles and labels
TITLE_PAD   = 0.05             # title/panel-label baseline above the frame
# Title/label pad for (c) and (e).
LOWER_TITLE_PAD = 0.11

# Measured heights of blocks outside a panel frame; too small clips the uncropped figure.
XAXIS_BLOCK    = 0.34          # x tick labels + x label below a time-series panel
TITLE_BLOCK    = 0.14          # title (or panel label) above the frame, incl. TITLE_PAD
FF_MIN_GAP  = 0.05             # least horizontal space between the two flow fields
E_PANEL_CLEARANCE = 0.30       # B stack to (e)'s frame, incl. (e)'s title

# B matrices have square cells: fix the height, width follows from M/D.
B_MAT_H       = 0.75 * 2 / 4
B_MAT_GAP     = 0.14
B_MAT_GAP_MIN = 0.02

# Panel-label offsets left of the frame, in inches ((b) and (c) share one).
LBL_DX_A  = 0.50
LBL_DX_B  = 0.43
LBL_DX_R  = 0.45               # right column: (d) B matrices, (e) D sweep

LW_TRUE  = 1.5
LW_MODEL = 1.0

D_SWEEP_METRIC = "y_corr_rollout"


# ── helpers ────────────────────────────────────────────────────────────────────

def _skip_warn(label: str, dir1: str | None, dir2: str | None) -> None:
    missing = []
    if not available_dir(dir1):
        missing.append(f"model_1 ({dir1!r})")
    if not available_dir(dir2):
        missing.append(f"model_2 ({dir2!r})")
    warnings.warn(f"Skipping {label} — directory not found for {', '.join(missing)}")


def _slice_std(eval_data, key: str, T: int) -> np.ndarray | None:
    """First T timesteps of an eval std array, or None if the eval didn't save it."""
    std = eval_data.get(key)
    return None if std is None else np.asarray(std)[:T]


def _shade_std(ax, t, mean, std, color) -> tuple[float, float]:
    """Shade a ±std band; returns the band's (min, max), or the mean's if std is None."""
    if std is None:
        return float(mean.min()), float(mean.max())
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.25, zorder=2)
    return float((mean - std).min()), float((mean + std).max())


def _clip_ylim(ax, ref_lo: float, ref_hi: float, band_lo: float, band_hi: float) -> None:
    """Cap the y-range at STD_RANGE_FACTOR x the lines' range so wide std bands clip."""
    ylim = std_clipped_ylim(ref_lo, ref_hi, band_lo, band_hi, STD_RANGE_FACTOR)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _system_kind(eval_data) -> str | None:
    """System kind (e.g. 'FitzHughNagumo') from any seed's run config, or None."""
    affine = eval_data.get("affine_maps")
    if not affine:
        return None
    run_dir = localize_path(next(iter(affine.values())).get("run_dir"))
    cfg_path = os.path.join(run_dir, "config.json") if run_dir else None
    if not cfg_path or not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return cfg.get("components", {}).get("system", {}).get("kind")


def _b_learned(eval_data) -> "np.ndarray | None":
    """Learned B for panel (d): scale-aligned if per-seed weights exist, else mean |B|."""
    aligned = eval_data.get("B_aligned")
    return eval_data.get("B_abs_mean") if aligned is None else aligned


def _b_heat_kw(mats: list) -> dict:
    """Shared colour scale for the B stack: diverging if any entry is negative, else sequential."""
    vals = np.concatenate([np.asarray(m, dtype=float).ravel()
                           for m in mats if m is not None]) if mats else np.array([])
    if vals.size == 0 or not np.isfinite(vals).any():
        return {}
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    vabs   = max(abs(lo), abs(hi), 1e-6)
    if lo < 0:
        return dict(cmap="bwr", vmin=-vabs, vmax=vabs)
    return dict(cmap="YlGnBu", vmin=0.0, vmax=vabs)


# ── stats export ───────────────────────────────────────────────────────────────

FIGURE = "toymodels"
N_BOOT = 10000


def d_sweep_stats(d_sweep_1: "str | None", d_sweep_2: "str | None",
                  dataset: str, eval_subdir: str = "eval") -> list:
    """Paired nonlinear-vs-linear records for panel (e), one per swept D."""
    jobs_lin = jobs_by_x(d_sweep_1, "settings.D", eval_subdir)
    jobs_nl  = jobs_by_x(d_sweep_2, "settings.D", eval_subdir)
    records = sweep_panel_records(
        jobs_lin, jobs_nl, figure=FIGURE, panel="e", system=dataset,
        metric=D_SWEEP_METRIC, x_name="D", sweep="D", eval_subdir=eval_subdir,
        n_boot=N_BOOT)
    if records:
        bh_stamp(records, f"BH-FDR over the {len(records)} sweep points of panel (e) "
                          f"({dataset}, D sweep, {D_SWEEP_METRIC})")
    return records


def d_sweep_per_seed_rows(d_sweep_1: "str | None", d_sweep_2: "str | None",
                          dataset: str, eval_subdir: str = "eval") -> list:
    """Per-seed values behind panel (e)."""
    rows = []
    for variant, sweep_dir in [("linear", d_sweep_1), ("nonlinear", d_sweep_2)]:
        for D, job_dir in sorted(jobs_by_x(sweep_dir, "settings.D",
                                           eval_subdir).items()):
            for seed, val in job_metric_by_seed(job_dir, D_SWEEP_METRIC,
                                                eval_subdir).items():
                rows.append(dict(figure=FIGURE, system=dataset, variant=variant,
                                 metric=D_SWEEP_METRIC, x_name="D", x_value=float(D),
                                 seed=int(seed), value=float(val)))
    return rows


# ── main benchmark figure ────────────────────────────────────────────────────────

def make_main_fig(d1, d2, name1, name2, color1, color2, device,
                  d_sweep=(None, None), d_records=None, flow_lim=None):
    """Main figure: (a) rollouts, (b) latents, (c) flow fields, (d) B matrices, (e) D sweep."""
    N  = d1["y_true"].shape[1]
    D  = d1["x_true"].shape[1]
    dt = d1.get("dt", 1.0)

    B_true = d1.get("B_true", d2.get("B_true"))
    B_ref  = B_true if B_true is not None else d1.get("B_abs_mean")
    M      = B_ref.shape[1] if B_ref is not None else D

    # Vertical anchors (inches from the figure's bottom edge).
    a_h   = N * CH_H + (N - 1) * CH_GAP
    b_h   = D * CH_H + (D - 1) * CH_GAP
    fig_w = TEXT_WIDTH
    fig_h = TOP_PAD + a_h + XAXIS_BLOCK
    a_top = b_top = fig_h - TOP_PAD
    a_bot = a_top - a_h
    b_bot = b_top - b_h

    # Horizontal anchors; (d) and (e) share the right column.
    a_left = LEFT_PAD
    mat_h  = B_MAT_H
    mat_w  = mat_h * M / D
    r_left = fig_w - RIGHT_PAD - mat_w

    # Place (b) so the gutters on either side of it match (at least GAP_AB).
    gutter  = ((r_left - LBL_DX_R) - (a_left + 2 * CH_W) - LBL_DX_B) / 2
    b_left  = a_left + CH_W + max(GAP_AB, LBL_DX_B + gutter)
    b_right = b_left + CH_W

    # (c) is square, spans (b)'s width and shares (a)'s frame bottom; (e) shares (c)'s frame.
    band_hi = b_bot - XAXIS_BLOCK
    ff_side = min(band_hi - a_bot - (TITLE_BLOCK - TITLE_PAD + LOWER_TITLE_PAD),
                  (CH_W - FF_MIN_GAP) / 2)
    ff_bot  = a_bot
    ff_top  = ff_bot + ff_side

    # Spread the B stack down to (e).
    mat_gap = min(B_MAT_GAP,
                  (b_top - 3 * mat_h - (ff_top + E_PANEL_CLEARANCE)) / 2)
    if mat_gap < B_MAT_GAP_MIN:          # few latents: shrink the matrices instead
        mat_gap = B_MAT_GAP_MIN
        mat_h   = min(mat_h, (b_top - ff_top - E_PANEL_CLEARANCE - 2 * mat_gap) / 3)
        mat_w   = mat_h * M / D
        r_left  = fig_w - RIGHT_PAD - mat_w

    fig = plt.figure(figsize=(fig_w, fig_h))

    def _ax(x, y, w, h):
        """Axes at (x, y) inches from the figure's bottom-left corner, w x h inches."""
        return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

    def _head(ax, letter, title, w, h, dx, pad=TITLE_PAD):
        """Panel title and label on a shared baseline pad inches above the frame."""
        if title:
            ax.set_title(title, fontsize=PANEL_TITLE_SIZE, pad=pad * 72)
        if letter:
            panel_label(ax, letter, x=-dx / w, y=1 + pad / h)

    # ── (a) y interventional ──────────────────────────────────────────────
    axs_y = [_ax(a_left, a_top - (i + 1) * CH_H - i * CH_GAP, CH_W, CH_H)
             for i in range(N)]
    T_min = min(d1["y_mean"].shape[0], d2["y_mean"].shape[0])
    t_y   = np.arange(T_min) * dt
    plot_dual_mean_predictions(
        d1["u"][:T_min], d1["y_true"][:T_min],
        d1["y_mean"][:T_min], pred_std_1=_slice_std(d1, "y_std", T_min), color_1=color1,
        pred_mean_2=d2["y_mean"][:T_min], pred_std_2=_slice_std(d2, "y_std", T_min),
        color_2=color2,
        axs=axs_y, variable_name="y", label_1=name1, label_2=name2, t_axis=t_y,
        std_range_factor=STD_RANGE_FACTOR, lw_true=LW_TRUE, lw_pred=LW_MODEL)
    _head(axs_y[0], "a", SERIES_LABELS["rollout"], CH_W, CH_H, LBL_DX_A)
    for ax in axs_y[:-1]:
        ax.tick_params(bottom=False, labelbottom=False)

    # ── (b) x interventional ──────────────────────────────────────────────
    axs_x = [_ax(b_left, b_top - (i + 1) * CH_H - i * CH_GAP, CH_W, CH_H)
             for i in range(D)]
    Tx    = min(d1["x_pdc_mean"].shape[0], d2["x_pdc_mean"].shape[0])
    t_x   = np.arange(Tx) * dt

    def _to_np(a):
        return a.detach().cpu().numpy() if hasattr(a, "detach") and callable(a.detach) else np.asarray(a)

    x_true_b   = _to_np(d1["x_true"][:Tx])
    x_lin_b    = _to_np(d1["x_pdc_mean"][:Tx])
    x_nonlin_b = _to_np(d2["x_pdc_mean"][:Tx])
    s_lin_b    = _slice_std(d1, "x_pdc_std", Tx)
    s_nonlin_b = _slice_std(d2, "x_pdc_std", Tx)
    u_b        = _to_np(d1["u"][:Tx])
    step_x     = float(t_x[1] - t_x[0]) if Tx > 1 else 1.0
    xlabel_x   = "Time (s)" if step_x != 1.0 else "Time step"

    for i, ax in enumerate(axs_x):
        for t in range(Tx):
            if np.any(u_b[t] != 0):
                ax.axvspan(t_x[t] - step_x / 2, t_x[t] + step_x / 2,
                           color="lightgray", alpha=STIM_ALPHA, zorder=0)

        ref_range = (max(x_true_b[:, i].max(), x_nonlin_b[:, i].max())
                     - min(x_true_b[:, i].min(), x_nonlin_b[:, i].min()))
        lin_range = x_lin_b[:, i].max() - x_lin_b[:, i].min()
        use_twin  = ref_range > 0 and lin_range > 2 * ref_range

        std_l = None if s_lin_b is None else s_lin_b[:, i]
        std_n = None if s_nonlin_b is None else s_nonlin_b[:, i]

        # Reference range for std clipping; a twinned linear model has its own.
        main = [x_true_b[:, i], x_nonlin_b[:, i]] + ([] if use_twin else [x_lin_b[:, i]])
        ref_lo = min(float(m.min()) for m in main)
        ref_hi = max(float(m.max()) for m in main)

        ax.plot(t_x, x_true_b[:, i],   color="black", lw=LW_TRUE,  label="true", zorder=1)
        b_lo, b_hi = _shade_std(ax, t_x, x_nonlin_b[:, i], std_n, color2)
        ax.plot(t_x, x_nonlin_b[:, i], color=color2, lw=LW_MODEL, label=name2, zorder=3)
        ax.set_ylabel(f"$x_{{{i}}}$", fontsize=MATH_LABEL_SIZE)

        if use_twin:
            ax_r = ax.twinx()
            lin_lo, lin_hi = _shade_std(ax_r, t_x, x_lin_b[:, i], std_l, color1)
            ax_r.plot(t_x, x_lin_b[:, i], color=color1, lw=LW_MODEL, label=name1,
                      linestyle="--", zorder=3)
            _clip_ylim(ax_r, float(x_lin_b[:, i].min()), float(x_lin_b[:, i].max()),
                       lin_lo, lin_hi)
            ax_r.tick_params(axis="y", colors=color1, labelsize=TICK_LABEL_SIZE)
            ax_r.spines["top"].set_visible(False)
            if i < D - 1:
                ax_r.tick_params(bottom=False, labelbottom=False)
        else:
            lin_lo, lin_hi = _shade_std(ax, t_x, x_lin_b[:, i], std_l, color1)
            ax.plot(t_x, x_lin_b[:, i], color=color1, lw=LW_MODEL, label=name1, zorder=3)
            b_lo, b_hi = min(b_lo, lin_lo), max(b_hi, lin_hi)

        _clip_ylim(ax, ref_lo, ref_hi, b_lo, b_hi)

        if i == D - 1:
            ax.set_xlabel(xlabel_x)

    _head(axs_x[0], "b", SERIES_LABELS["latent"], CH_W, CH_H, LBL_DX_B)
    for ax in axs_x[:-1]:
        ax.tick_params(bottom=False, labelbottom=False)

    # ── (e) D sweep: right column, below the B matrices ───────────────────
    e_h = ff_side
    ax_sw = _ax(r_left, ff_bot, mat_w, e_h)
    sw_lin, sw_nl = d_sweep
    if sw_lin and sw_nl:
        # Metric is named by the panel title.
        plot_sweep([(sw_lin, sw_nl, D_SWEEP_METRIC, "solid")], "settings.D",
                   name1, name2, color1, color2,
                   output_dir=None, fmt=None, ax=ax_sw,
                   xlabel=XLABEL_D_SHORT, ylabel="",
                   metric_labels=METRIC_LABELS, legend=False,
                   significance="asterisks",
                   records=d_records or None)
        # Integer D ticks; fixed y tick density across systems.
        ax_sw.set_xticks([r[0] for r in sw_lin])
        ax_sw.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    else:
        ax_sw.text(0.5, 0.5, "D sweep\n(N/A)", ha="center", va="center",
                   transform=ax_sw.transAxes, color="gray")
        ax_sw.set_xticks([])
        ax_sw.set_yticks([])
    _head(ax_sw, "e", METRIC_LABELS[D_SWEEP_METRIC], mat_w, e_h, LBL_DX_R,
          pad=LOWER_TITLE_PAD)

    # ── (c) flow fields ───────────────────────────────────────────────────
    ax_ff1 = _ax(b_left, ff_bot, ff_side, ff_side)
    ax_ff2 = _ax(b_right - ff_side, ff_bot, ff_side, ff_side)
    _head(ax_ff1, "c", "", ff_side, ff_side, LBL_DX_B, pad=LOWER_TITLE_PAD)
    if D == 2:
        # Same start in both panels so the true trajectory is identical.
        x_true_full = _to_np(d1["x_true"])
        traj_start  = FLOW_TRAJ_STARTS.get(_system_kind(d1), x_true_full[0])
        traj_steps  = min(len(x_true_full) - 1, 200)
        for ax, eval_data, col, label in [
                (ax_ff1, d1, color1, f"{name1} flow"),
                (ax_ff2, d2, color2, f"{name2} flow")]:
            true_fn, learned_fn, x_data = load_best_seed_flow_fns(eval_data, device)
            if true_fn is not None:
                if flow_lim is not None:
                    lo, hi = flow_lim
                else:
                    lo = float(x_data.min())
                    hi = float(x_data.max())
                plot_flow_fields(
                    [true_fn, learned_fn],
                    colors=[TRUE_FLOW_COLOR, col], ax=ax,
                    xlim=(lo, hi), ylim=(lo, hi),
                    n_points=16, max_arrow_length=0.4,
                    alphas=[1.0, 0.9],
                    widths=[0.005, 0.005],
                    trajectory_starts=traj_start,
                    trajectory_colors=[TRUE_TRAJ_COLOR, col],
                    trajectory_steps=traj_steps, trajectory_dt=dt,
                    trajectory_lw=1.5)
                # Pin limits so a diverging learned trajectory is clipped.
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_title(label, fontsize=PANEL_TITLE_SIZE,
                             pad=LOWER_TITLE_PAD * 72)
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{label}\n(N/A)", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
    else:
        for ax in (ax_ff1, ax_ff2):
            ax.axis("off")
            ax.text(0.5, 0.5, f"Flow field\nN/A (D={D} ≠ 2)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
    ax_ff2.tick_params(left=False, labelleft=False)
    ax_ff2.set_ylabel("")

    # ── (d) B matrices: right column, stacked down from (b)'s top ─────────
    ax_bt = _ax(r_left, b_top - mat_h, mat_w, mat_h)
    ax_bl = _ax(r_left, b_top - 2 * mat_h - mat_gap, mat_w, mat_h)
    ax_bn = _ax(r_left, b_top - 3 * mat_h - 2 * mat_gap, mat_w, mat_h)

    b_mats = [B_true, _b_learned(d1), _b_learned(d2)]
    b_kw   = _b_heat_kw(b_mats)
    for ax, mat in zip((ax_bt, ax_bl, ax_bn), b_mats):
        draw_matrix_heatmap(ax, mat, "", show_colorbar=False, n_decimals=1, **b_kw)

    _head(ax_bt, "d", "B matrices", mat_w, mat_h, LBL_DX_R)
    for ax, lbl in [(ax_bt, "True"), (ax_bl, name1), (ax_bn, name2)]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(lbl, rotation=90, ha="center", va="center", labelpad=3,
                      fontsize=TICK_LABEL_SIZE)

    # Ticks are chosen at draw time, so coarsen after drawing.
    fig.canvas.draw()
    for ax in axs_x:
        coarse_axis_ticks(ax, axis="y", decimals=0, max_ticks=5, min_ticks=3)
    if sw_lin and sw_nl:
        coarse_axis_ticks(ax_sw, axis="y", decimals=2, max_ticks=5, min_ticks=3)

    # Align the bottom-row x labels with (a)'s (mathtext makes them taller).
    # labelpad is in points and grows downwards.
    inv = fig.dpi_scale_trans.inverted()
    ref = axs_y[-1].xaxis.label.get_window_extent().transformed(inv).y0
    for ax in (ax_ff1, ax_ff2) + ((ax_sw,) if sw_lin and sw_nl else ()):
        y0 = ax.xaxis.label.get_window_extent().transformed(inv).y0
        ax.xaxis.labelpad -= (ref - y0) * 72

    return fig


# ── section runners ────────────────────────────────────────────────────────────

def _run_benchmark(
    dir1: str, dir2: str,
    out_dir: str, fmt: str,
    dataset: str,
    device: torch.device,
    d_sweep_1: "str | None" = None,
    d_sweep_2: "str | None" = None,
    flow_lim: tuple | None = None,
) -> None:
    name1, name2 = MODEL_LABELS["linear"], MODEL_LABELS["nonlinear"]
    print("\n── Benchmark panels ─────────────────────────────────────────────────")
    print(f"  {name1}: {dir1}")
    print(f"  {name2}: {dir2}")
    print(f"  D sweep: {d_sweep_1}")

    # A missing sweep leaves panel (e) as N/A and writes no stats.
    if available_dir(d_sweep_1) and available_dir(d_sweep_2):
        kw = dict(config_param="settings.D", eval_subdir="eval")
        d_sweep = (load_sweep(d_sweep_1, metric=D_SWEEP_METRIC, **kw),
                   load_sweep(d_sweep_2, metric=D_SWEEP_METRIC, **kw))
    else:
        _skip_warn("D sweep panel (e)", d_sweep_1, d_sweep_2)
        d_sweep_1 = d_sweep_2 = None
        d_sweep = (None, None)

    d1 = load_eval_data(dir1)
    d2 = load_eval_data(dir2)

    # B in true latent coordinates: per-seed permutation/scale alignment, then median.
    # None when per-seed weights are unavailable.
    for d in (d1, d2):
        d["B_aligned"] = load_scale_aligned_B(d, device)

    # The figure's asterisks come from these records.
    d_recs = d_sweep_stats(d_sweep_1, d_sweep_2, dataset)
    out = Path(out_dir)
    write_stats_csv(d_recs, out / f"{FIGURE}_{dataset}_stats.csv",
                    extra_columns=SWEEP_STATS_EXTRA)
    write_per_seed_csv(d_sweep_per_seed_rows(d_sweep_1, d_sweep_2, dataset),
                       out / f"{FIGURE}_{dataset}_per_seed.csv")

    fig_main = make_main_fig(d1, d2, name1, name2, LINEAR_COLOR, NONLINEAR_COLOR, device,
                             d_sweep=d_sweep, d_records=d_recs, flow_lim=flow_lim)
    p_main = os.path.join(out_dir, f"fig_toymodels_main_{dataset}.{fmt}")
    save_figure(fig_main, p_main, fmt=fmt)
    plt.close(fig_main)
    print(f"\n  Saved: {p_main}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASET_SYSTEMS),
                   help="Toy system: fhn (FitzHugh-Nagumo) or vdp (Van der Pol)")
    p.add_argument("--d_sweep_dir_1", default=None,
                   help="D-sweep dir of the linear model (default: "
                        "outputs/toymodels/<System>/linear)")
    p.add_argument("--d_sweep_dir_2", default=None,
                   help="D-sweep dir of the nonlinear model (default: "
                        "outputs/toymodels/<System>/nonlinear)")
    p.add_argument("--eval_dir_1", default=None,
                   help="Eval dir of the linear model for panels (a)-(d) "
                        "(default: <d_sweep_dir_1>/job_1/eval)")
    p.add_argument("--eval_dir_2", default=None,
                   help="Eval dir of the nonlinear model for panels (a)-(d) "
                        "(default: <d_sweep_dir_2>/job_1/eval)")
    p.add_argument("--output_dir", default=None,
                   help="Directory for figures and stats (default: figures/toymodels/)")
    p.add_argument("--fmt",    default="png",
                   help="Figure format: png, svg, pdf (default: png)")
    p.add_argument("--device", default="cpu",
                   help="Torch device for flow-field loading (default: cpu)")
    return p.parse_args()


def main() -> None:
    apply_paper_style()
    args = parse_args()
    dataset = args.dataset

    base = os.path.join("outputs", "toymodels", DATASET_SYSTEMS[dataset])
    d_sweep_1 = add_root(args.d_sweep_dir_1 or os.path.join(base, "linear"))
    d_sweep_2 = add_root(args.d_sweep_dir_2 or os.path.join(base, "nonlinear"))
    eval_1 = add_root(args.eval_dir_1 or os.path.join(d_sweep_1, "job_1", "eval"))
    eval_2 = add_root(args.eval_dir_2 or os.path.join(d_sweep_2, "job_1", "eval"))

    out_dir = add_root(args.output_dir or os.path.join("figures", "toymodels"))
    fmt     = args.fmt.lstrip(".")
    device  = torch.device(args.device)
    os.makedirs(out_dir, exist_ok=True)

    # Per-dataset flow-field axis limits (None = auto from data)
    _FLOW_LIMS = {"vdp": (-4.0, 4.0)}
    flow_lim = _FLOW_LIMS.get(dataset)

    if available_dir(eval_1) and available_dir(eval_2):
        _run_benchmark(
            eval_1, eval_2,
            out_dir=out_dir, fmt=fmt, dataset=dataset,
            d_sweep_1=d_sweep_1, d_sweep_2=d_sweep_2,
            device=device, flow_lim=flow_lim,
        )
    else:
        _skip_warn("benchmark panels", eval_1, eval_2)

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
