"""Macaque micro-stimulation figures (linear vs nonlinear) and stats CSVs, from existing eval outputs.

Writes overview_macaque, main_macaque, supplement_macaque_d_sweeps and macaque_*.csv.

python -m scripts.figures.figure_realdata --figures main d_sweeps
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import add_root
from src.utils.style import apply_paper_style, save_figure, TEXT_WIDTH, LINEAR_COLOR, NONLINEAR_COLOR
from scripts.figures._common.constants import SERIES_LABELS
from scripts.figures._common.stats_export import export_realdata_stats, compute_sweep_stats
from scripts.figures._common.realdata_panels import (
    XLABEL_D_MATH, RIGHT_PAD, TOP_PAD, BOTTOM_PAD, COL_GAP, ROW_GAP,
    TITLE_PAD_PT, LBL_DY_TITLE, OV_ROW_H, OV_LEFT_PAD, MAIN_PANEL_H, MAIN_LEFT_PAD, MAIN_GAP_AB,
    MAIN_GAP_BOX, MAIN_BOX_LBL_DX, BARE_LBL_DX, SUP_PANEL_H,
    panel_timeseries, panels_tier, panels_sweep,
    ylabel_to_title, inch_axes, label_panels, two_panel_row,
)

DATASET = "macaque"
DEFAULT_TRAIN_DIRS = ("outputs/realdata/macaque/linear/job_1",
                      "outputs/realdata/macaque/nonlinear/job_1")

XLABEL_TIER = "Trial type"

TIME_UNIT  = "s"
TIME_TICKS = (0, 0.6, 1.2, 1.8)

# Main figure: five channels fit the panel height without normalisation.
MAIN_CHANNELS = range(5)
MAIN_WEIGHTS  = (1.1, 1.1, 1.0, 1.0)   # recon., rollout, then the two boxplots
# No anatomy key below the panels, so tighter than MAIN_BOTTOM_PAD.
MAIN_BOTTOM_PAD_TIGHT = 0.37

LEGEND_LOC       = "lower left"
SWEEP_LEGEND_LOC = "upper left"
# Trace key of the main figure's panel (a), models only.
TRACE_LEGEND_KW  = dict(loc="lower left", fancybox=True, framealpha=0.7,
                        borderpad=0.25, borderaxespad=0.0)
# Gap to the y and x axes, in points.
TRACE_LEGEND_PAD = (2.4, 0.5)


# ── figures ──────────────────────────────────────────────────────────────────────

def _build_overview(
    train_dir1: "str | None", train_dir2: "str | None",
    d_sw_1: "str | None", d_sw_2: "str | None",
    name1: str, name2: str, color1: str, color2: str,
    eval_subdir: str, out_dir: str, fmt: str, name_suffix: str,
    sweep_records: "list | None" = None,
) -> None:
    """Reconstructions | rollouts | 2x2 comparison (tiers over D sweeps), lettered a-f."""
    print("\n── Overview figure ──────────────────────────────────────────────────")

    fig_w = TEXT_WIDTH
    fig_h = TOP_PAD + 2 * OV_ROW_H + ROW_GAP + BOTTOM_PAD
    col_w = (fig_w - OV_LEFT_PAD - RIGHT_PAD - MAIN_GAP_AB - 2 * COL_GAP) / 4
    fig = plt.figure(figsize=(fig_w, fig_h))
    _ax = inch_axes(fig, fig_w, fig_h)

    xs = [OV_LEFT_PAD]
    xs.append(xs[-1] + col_w + MAIN_GAP_AB)
    xs.append(xs[-1] + col_w + COL_GAP)
    xs.append(xs[-1] + col_w + COL_GAP)
    ts_h = 2 * OV_ROW_H + ROW_GAP
    y_top, y_bot = BOTTOM_PAD + OV_ROW_H + ROW_GAP, BOTTOM_PAD
    ax_enc           = _ax(xs[0], y_bot, col_w, ts_h)
    ax_roll          = _ax(xs[1], y_bot, col_w, ts_h)
    ax_tier_ycorr    = _ax(xs[2], y_top, col_w, OV_ROW_H)
    ax_tier_ycorr_r  = _ax(xs[3], y_top, col_w, OV_ROW_H)
    ax_sweep_ycorr   = _ax(xs[2], y_bot, col_w, OV_ROW_H)
    ax_sweep_ycorr_r = _ax(xs[3], y_bot, col_w, OV_ROW_H)

    ts_kw = dict(eval_subdir=eval_subdir, name1=name1, name2=name2,
                 color1=color1, color2=color2, time_unit=TIME_UNIT,
                 time_ticks=TIME_TICKS)
    panel_timeseries(ax_enc, train_dir1, train_dir2, keys=("y_mean", "y_std"),
                     title=SERIES_LABELS["recon"], **ts_kw)
    panel_timeseries(ax_roll, train_dir1, train_dir2, keys=("y_rollout_mean", "y_rollout_std"),
                     title=SERIES_LABELS["rollout"], **ts_kw)
    panels_tier([ax_tier_ycorr, ax_tier_ycorr_r], train_dir1, train_dir2,
                ["y_corr", "y_corr_rollout"], name1, name2, color1, color2,
                eval_subdir, name_suffix, legend=True, legend_loc=LEGEND_LOC,
                legend_band=True)
    panels_sweep([ax_sweep_ycorr, ax_sweep_ycorr_r], d_sw_1, d_sw_2,
                 ["y_corr", "y_corr_rollout"], name1, name2, color1, color2,
                 eval_subdir, out_dir, fmt, significance="asterisks",
                 records=sweep_records)

    label_panels([(ax_enc, "a", col_w), (ax_roll, "b", col_w)], ts_h, dx=BARE_LBL_DX)
    label_panels([(ax_tier_ycorr, "c", col_w), (ax_tier_ycorr_r, "d", col_w)], OV_ROW_H)
    label_panels([(ax_sweep_ycorr, "e", col_w), (ax_sweep_ycorr_r, "f", col_w)], OV_ROW_H)

    path = os.path.join(out_dir, f"overview_{name_suffix}")
    save_figure(fig, path, fmt=fmt)
    plt.close(fig)
    print(f"  Saved: {path}.{fmt}")


def _build_main(
    train_dir1: "str | None", train_dir2: "str | None",
    name1: str, name2: str, color1: str, color2: str,
    eval_subdir: str, out_dir: str, fmt: str, name_suffix: str,
) -> None:
    """One row: reconstructions | rollouts | recon. accuracy | rollout accuracy."""
    print("\n── Main figure ──────────────────────────────────────────────────────")

    fig_w = TEXT_WIDTH
    fig_h = TOP_PAD + MAIN_PANEL_H + MAIN_BOTTOM_PAD_TIGHT
    fig = plt.figure(figsize=(fig_w, fig_h))
    _ax = inch_axes(fig, fig_w, fig_h)

    rest = fig_w - MAIN_LEFT_PAD - RIGHT_PAD - MAIN_GAP_AB - 2 * MAIN_GAP_BOX
    widths = [rest / sum(MAIN_WEIGHTS) * w for w in MAIN_WEIGHTS]
    gaps = [MAIN_GAP_AB, MAIN_GAP_BOX, MAIN_GAP_BOX]
    xs = [MAIN_LEFT_PAD]
    for w, g in zip(widths[:-1], gaps):
        xs.append(xs[-1] + w + g)
    ax_enc, ax_roll, ax_ycorr, ax_roll_acc = [
        _ax(x, MAIN_BOTTOM_PAD_TIGHT, w, MAIN_PANEL_H) for x, w in zip(xs, widths)]

    ts_kw = dict(eval_subdir=eval_subdir, name1=name1, name2=name2,
                 color1=color1, color2=color2, channels=MAIN_CHANNELS,
                 time_unit=TIME_UNIT, time_ticks=TIME_TICKS,
                 title_pad=TITLE_PAD_PT)
    # Key drawn over the traces so the panel keeps the boxplots' frame.
    panel_timeseries(ax_enc, train_dir1, train_dir2, keys=("y_mean", "y_std"),
                     title=SERIES_LABELS["recon"], legend=True, legend_band=False,
                     legend_obs=False, legend_kw={
                         **TRACE_LEGEND_KW,
                         "bbox_to_anchor": (TRACE_LEGEND_PAD[0] / 72 / widths[0],
                                            TRACE_LEGEND_PAD[1] / 72 / MAIN_PANEL_H)},
                     **ts_kw)
    panel_timeseries(ax_roll, train_dir1, train_dir2, keys=("y_rollout_mean", "y_rollout_std"),
                     title=SERIES_LABELS["rollout"], **ts_kw)
    panels_tier([ax_ycorr, ax_roll_acc], train_dir1, train_dir2,
                ["y_corr", "y_corr_rollout"], name1, name2, color1, color2,
                eval_subdir, name_suffix, xlabel=XLABEL_TIER,
                legend=True, legend_loc=LEGEND_LOC, legend_band=True)
    for ax in (ax_ycorr, ax_roll_acc):
        ylabel_to_title(ax)

    label_panels([(ax_enc, "a", widths[0]), (ax_roll, "b", widths[1])],
                 MAIN_PANEL_H, dx=BARE_LBL_DX, dy=LBL_DY_TITLE)
    label_panels([(ax_ycorr, "c", widths[2]), (ax_roll_acc, "d", widths[3])],
                 MAIN_PANEL_H, dx=MAIN_BOX_LBL_DX, dy=LBL_DY_TITLE)

    path = os.path.join(out_dir, f"main_{name_suffix}")
    save_figure(fig, path, fmt=fmt)
    plt.close(fig)
    print(f"  Saved: {path}.{fmt}")


def _build_d_sweeps(
    d_sw_1: "str | None", d_sw_2: "str | None",
    name1: str, name2: str, color1: str, color2: str,
    eval_subdir: str, out_dir: str, fmt: str, name_suffix: str,
    sweep_records: "list | None" = None,
) -> None:
    """Reconstruction and rollout accuracy against D."""
    print("\n── Latent-dimension sweep ───────────────────────────────────────────")
    fig, axes, panel_w = two_panel_row()
    panels_sweep(axes, d_sw_1, d_sw_2, ["y_corr", "y_corr_rollout"],
                 name1, name2, color1, color2, eval_subdir, out_dir, fmt,
                 xlabel=XLABEL_D_MATH, legend=True, legend_loc=SWEEP_LEGEND_LOC,
                 significance="asterisks", records=sweep_records)
    label_panels(list(zip(axes, "ab", [panel_w] * 2)), SUP_PANEL_H)

    path = os.path.join(out_dir, f"supplement_{name_suffix}_d_sweeps")
    save_figure(fig, path, fmt=fmt)
    plt.close(fig)
    print(f"  Saved: {path}.{fmt}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--linear_train_dir", default=DEFAULT_TRAIN_DIRS[0],
                   help="Linear model's group dir (seed_*/ parent)")
    p.add_argument("--nonlinear_train_dir", default=DEFAULT_TRAIN_DIRS[1],
                   help="Nonlinear model's group dir")
    p.add_argument("--linear_d_sweep_dir", default=None,
                   help="Linear D-sweep dir (job_*/ parent; default: parent of the train dir)")
    p.add_argument("--nonlinear_d_sweep_dir", default=None,
                   help="Nonlinear D-sweep dir (default: parent of the train dir)")
    p.add_argument("--name_1", default="Linear", help="Display name for the linear model")
    p.add_argument("--name_2", default="Nonlinear", help="Display name for the nonlinear model")
    p.add_argument("--color_1", default=LINEAR_COLOR, help="Color for linear model")
    p.add_argument("--color_2", default=NONLINEAR_COLOR, help="Color for nonlinear model")
    p.add_argument("--output_dir", default=None,
                   help="Directory for figures and stats (default: figures/realdata/)")
    p.add_argument("--fmt",    default="png",
                   help="Figure format: png, svg, pdf (default: png)")
    p.add_argument("--figures", nargs="*",
                   default=["overview", "main", "d_sweeps"],
                   choices=["overview", "main", "d_sweeps"],
                   help="Which figures to render (default: all three)")
    p.add_argument("--eval_subdir", default="eval",
                   help="Subdir inside each job_*/ for eval outputs (default: eval)")
    return p.parse_args()


def main() -> None:
    apply_paper_style()
    args = parse_args()

    out_dir = add_root(args.output_dir or os.path.join("figures", "realdata"))
    fmt     = args.fmt.lstrip(".")
    os.makedirs(out_dir, exist_ok=True)

    train_1 = add_root(args.linear_train_dir)
    train_2 = add_root(args.nonlinear_train_dir)
    d_sw_1  = add_root(args.linear_d_sweep_dir) or str(Path(train_1).parent)
    d_sw_2  = add_root(args.nonlinear_d_sweep_dir) or str(Path(train_2).parent)

    name1, name2   = args.name_1, args.name_2
    color1, color2 = args.color_1, args.color_2

    # Sweep asterisks use the same records as macaque_stats.csv.
    sweep_records = compute_sweep_stats(d_sw_1, d_sw_2, DATASET, args.eval_subdir)

    if "overview" in args.figures:
        _build_overview(
            train_1, train_2, d_sw_1, d_sw_2,
            name1, name2, color1, color2,
            args.eval_subdir, out_dir, fmt, DATASET,
            sweep_records=sweep_records,
        )

    if "main" in args.figures:
        _build_main(
            train_1, train_2,
            name1, name2, color1, color2,
            args.eval_subdir, out_dir, fmt, DATASET,
        )

    if "d_sweeps" in args.figures:
        _build_d_sweeps(
            d_sw_1, d_sw_2,
            name1, name2, color1, color2,
            args.eval_subdir, out_dir, fmt, DATASET,
            sweep_records=sweep_records,
        )

    export_realdata_stats(
        train_1, train_2, d_sw_1, d_sw_2,
        DATASET, args.eval_subdir, out_dir,
    )
    print(f"\nDone. Figures and stats in {out_dir}")


if __name__ == "__main__":
    main()
