"""Huang (2023) SEEG single-pulse stimulation figures (linear vs nonlinear) and stats CSVs.

Reads existing eval and attractor outputs; writes overview_huang, main_huang,
supplement_huang_{d_sweeps,seizure_zones} and huang_*.csv.

python -m scripts.figures.figure_huang --figures main
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.datasets.huang import _is_white_matter
from src.utils.io import add_root, get_root
from src.utils.plotting import coarse_axis_ticks, significance_stars
from src.utils.stats import ALPHA
from src.utils.style import (
    apply_paper_style, save_figure,
    TEXT_WIDTH, LEGEND_SIZE, LINEAR_COLOR, NONLINEAR_COLOR,
    ZONE_EPI_COLOR, ZONE_HEALTHY_COLOR,
)
from scripts.figures._common.constants import HUANG_TIER_LABELS, SEIZURE_READOUT_LABELS
from scripts.figures._common.stats_export import (
    export_realdata_stats, bh_stamp, compute_sweep_stats,
)
from scripts.figures._common.realdata_panels import (
    XLABEL_D_MATH, RIGHT_PAD, TOP_PAD, BOTTOM_PAD, COL_GAP, ROW_GAP,
    LBL_DY_TITLE, OV_ROW_H, OV_LEFT_PAD, MAIN_LEFT_PAD, MAIN_GAP_AB,
    MAIN_GAP_BOX, MAIN_BOX_LBL_DX, BARE_LBL_DX, MAIN_BOTTOM_PAD, SUP_PANEL_H,
    MEDIAN_LW, STAR_SIZE,
    panel_timeseries, panels_tier, panels_sweep,
    ylabel_to_title, inch_axes, label_panels, two_panel_row,
)

DATASET = "huang"
DEFAULT_TRAIN_DIRS = ("outputs/realdata/huang/linear/job_2",
                      "outputs/realdata/huang/nonlinear/job_2")

XLABEL_STIM_SITE = "Stim. site"

# Seizure readouts shown in the right-hand column: (csv column, panel label, log_y).
_SEIZURE_PANELS = [
    ("resp_peak",  SEIZURE_READOUT_LABELS["resp_peak"],  False),
    ("plateau_uv", SEIZURE_READOUT_LABELS["plateau_uv"], False),
]
# Seizure readouts averaged over seeds and summarised in huang_seizure_per_seed.csv.
_SEIZURE_READOUTS = ["resp_peak", "plateau_uv", "resp_auc", "gain"]

# Geometry specific to this figure; shared layout is in _common/realdata_panels.py.
MAIN_WEIGHTS  = (1.25, 1.0, 1.0)   # rollout, then the two boxplots
MAIN_ROLLOUT_CHANNELS = 5
HUANG_MAIN_PANEL_H = 1.45
LEGEND_BASELINE = 0.04 # fallback when there is no x label to measure against

# Mark sizes for the seizure and anatomy panels at TEXT_WIDTH.
_SEIZ_POINT_SIZE = 3.0
_BOX_LW          = 0.7
_CONTACT_SIZE    = 7.0
_OOD_SITE_SIZE   = 26.0

# Region: hollow (healthy) vs filled (epileptogenic); colour encodes the model.
_REGION_FILLED = {"healthy": False, "epileptogenic": True}


# ── seizure-zone analysis ────────────────────────────────────────────────────────

def _load_seizure_ensemble(
    train_dir: "str | None", attractor_subdir: str,
) -> "pd.DataFrame | None":
    """Seed-averaged virtual-stim readouts per bipolar pair.

    Pairs come from the electrode geometry and readouts are in observation space, so
    they average directly across seeds.
    """
    if not train_dir:
        return None
    paths = sorted(Path(train_dir).glob(f"seed_*/{attractor_subdir}/seizure_virtual_stim.csv"))
    if not paths:
        return None

    seeds = [int(p.parent.parent.name.split("_")[-1]) for p in paths]
    frames = [pd.read_csv(p).set_index("pair") for p in paths]

    # Keep only pairs reported by every seed.
    pairs = frames[0].index
    for f in frames[1:]:
        pairs = pairs.intersection(f.index)
    dropped = len(frames[0]) - len(pairs)
    if dropped:
        print(f"  {dropped} pair(s) absent from at least one seed; dropped.")
    frames = [f.loc[pairs] for f in frames]

    readouts = [c for c in _SEIZURE_READOUTS if c in frames[0].columns]
    out = frames[0].drop(columns=readouts).copy()
    for col in readouts:
        stack = np.stack([f[col].to_numpy(dtype=float) for f in frames])   # seeds x pairs
        out[col] = np.nanmean(stack, axis=0)
    out = out.reset_index()
    out.attrs.update(n_seeds=len(frames), seeds=seeds)
    print(f"  Seizure ensemble: {len(frames)} seeds {seeds}, {len(out)} pairs ({train_dir})")
    return out


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled SD) for a vs b; positive = a (epileptogenic) larger."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def _seizure_arrays(
    df1: "pd.DataFrame | None", df2: "pd.DataFrame | None", readouts: list[str],
) -> dict:
    """Box arrays per (model, readout): {"healthy": arr, "epileptogenic": arr}."""
    out = {}
    for model, df in (("linear", df1), ("nonlinear", df2)):
        for r in readouts:
            epi = hea = np.array([])
            if df is not None and r in df.columns:
                epi = df.loc[df["epileptogenic"], r].dropna().to_numpy()
                hea = df.loc[~df["epileptogenic"], r].dropna().to_numpy()
            out[(model, r)] = {"healthy": hea, "epileptogenic": epi}
    return out


def _seizure_records(
    df1: "pd.DataFrame | None", df2: "pd.DataFrame | None", readouts: list[str],
) -> list:
    """Uncorrected Mann-Whitney records per (readout, model), in STATS_COLUMNS shape.

    Effect sizes are epileptogenic - healthy; one seed-averaged value per bipolar pair.
    """
    from scipy.stats import mannwhitneyu

    arrays = _seizure_arrays(df1, df2, readouts)
    attrs = {"linear": (df1.attrs if df1 is not None else {}),
             "nonlinear": (df2.attrs if df2 is not None else {})}
    records = []
    for model in ("linear", "nonlinear"):
        n_seeds = attrs[model].get("n_seeds", 0)
        for r in readouts:
            epi = arrays[(model, r)]["epileptogenic"]
            hea = arrays[(model, r)]["healthy"]
            rec = {
                "figure": "figure_huang", "panel": "seizure", "system": "huang",
                "metric": r, "comparison": "epileptogenic_vs_healthy",
                "x_name": "model", "x_value": model,
                "unit_of_analysis": f"bipolar pair (mean over {n_seeds} seeds)",
                "n_seeds_aggregated": n_seeds,
                "group_1": "epileptogenic", "group_2": "healthy",
                "mean_1": float(np.mean(epi)) if len(epi) else float("nan"),
                "mean_2": float(np.mean(hea)) if len(hea) else float("nan"),
                "median_diff": (float(np.median(epi) - np.median(hea))
                                if len(epi) and len(hea) else float("nan")),
                "test": "mann_whitney_u",
                "n_epileptogenic": int(len(epi)), "n_healthy": int(len(hea)),
                "median_epileptogenic": float(np.median(epi)) if len(epi) else float("nan"),
                "median_healthy": float(np.median(hea)) if len(hea) else float("nan"),
            }
            if len(epi) >= 2 and len(hea) >= 2:
                U, p = mannwhitneyu(epi, hea, alternative="two-sided")
                rec["test_statistic"] = float(U)
                rec["p_raw"] = float(p)
                rec["rank_biserial"] = float(2 * U / (len(epi) * len(hea)) - 1)
                rec["cohen_d"] = _cohens_d(epi, hea)
            else:
                rec["test_statistic"] = rec["p_raw"] = float("nan")
                rec["rank_biserial"] = rec["cohen_d"] = float("nan")
            records.append(rec)
    return records


def compute_seizure_stats(
    df1: "pd.DataFrame | None", df2: "pd.DataFrame | None", readouts: list[str],
) -> list:
    """BH-FDR corrected Mann-Whitney (epileptogenic vs healthy) records; also drive the panels."""
    records = _seizure_records(df1, df2, readouts)
    scope = (f"seizure epileptogenic-vs-healthy over {len(readouts)} readouts x 2 "
             f"models (figure_huang panel cells)")
    finite = [r for r in records if np.isfinite(r["p_raw"])]
    bh_stamp(finite, scope)
    for rec in records:  # complete the schema for any untested (n<2) cell
        if "significant" not in rec:
            rec.update(correction="BH-FDR", correction_scope=scope,
                       n_comparisons_in_scope=len(finite),
                       p_corrected=float("nan"), alpha=ALPHA, significant=False)
    return records


def _seizure_panel_stats(
    df1: "pd.DataFrame | None", df2: "pd.DataFrame | None",
    readouts: list[str], records: list,
) -> dict:
    """{(model, readout): {healthy, epileptogenic, q}} with q taken from records."""
    arrays = _seizure_arrays(df1, df2, readouts)
    q_by = {(r["x_value"], r["metric"]): r.get("p_corrected", float("nan")) for r in records}
    return {key: {**arrays[key], "q": q_by.get(key, float("nan"))} for key in arrays}


def _seizure_panel(
    ax, stats: dict, metric: str, ylabel: str,
    color1: str, color2: str, log_y: bool = False,
) -> None:
    """One seizure panel: Linear/Nonlinear x Healthy/Epileptogenic boxes with significance brackets."""
    groups = [("linear", color1, 0.0), ("nonlinear", color2, 1.2)]
    offset, width = 0.28, 0.40
    rng = np.random.default_rng(0)

    y_tops = {}
    for model, color, centre in groups:
        d = stats.get((model, metric), {})
        both = []
        for xp, region in [(centre - offset, "healthy"),
                           (centre + offset, "epileptogenic")]:
            vals = d.get(region, np.array([]))
            if len(vals) == 0:
                continue
            filled = _REGION_FILLED[region]
            bp = ax.boxplot([vals], positions=[xp], widths=width,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=MEDIAN_LW))
            for patch in bp["boxes"]:
                patch.set_facecolor(color if filled else "white")
                patch.set_alpha(0.75 if filled else 1.0)
                patch.set_edgecolor(color)
                patch.set_linewidth(_BOX_LW)
            for artist in bp["whiskers"] + bp["caps"]:
                artist.set_color(color)
            jitter = rng.normal(0, 0.05, len(vals))
            ax.scatter(xp + jitter, vals, s=_SEIZ_POINT_SIZE, zorder=4, linewidths=0.4,
                       facecolors=color if filled else "none",
                       edgecolors=color, alpha=0.6 if filled else 0.8)
            both.append(vals)
        y_tops[model] = float(np.nanmax(np.concatenate(both))) if both else 0.0

    ax.set_xticks([c for _, _, c in groups])
    ax.set_xticklabels(["Linear", "Nonlinear"])
    ax.set_xlim(-0.7, 1.9)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    fwd = np.log10 if log_y else (lambda v: v)
    inv = (lambda v: 10.0 ** v) if log_y else (lambda v: v)
    t0, t1 = fwd(ax.get_ylim()[0]), fwd(ax.get_ylim()[1])
    span = (t1 - t0) or 1.0
    top_needed = t1
    for model, _color, centre in groups:
        d = stats.get((model, metric), {})
        y_bar = fwd(y_tops.get(model, 0.0)) + 0.06 * span
        x1, x2 = centre - offset, centre + offset
        ax.plot([x1, x1, x2, x2],
                [inv(y_bar), inv(y_bar + 0.02 * span),
                 inv(y_bar + 0.02 * span), inv(y_bar)],
                color="black", lw=0.6, zorder=5)
        ax.text(centre, inv(y_bar + 0.03 * span),
                significance_stars(d.get("q", float("nan")), ns="n.s."),
                ha="center", va="bottom", color="black", fontsize=STAR_SIZE, zorder=5)
        top_needed = max(top_needed, y_bar + 0.16 * span)
    ax.set_ylim(top=inv(top_needed))


# ── seizure panels ───────────────────────────────────────────────────────────────

def _panels_seizure(
    axes, seiz1: "pd.DataFrame | None", seiz2: "pd.DataFrame | None",
    color1: str, color2: str, records: "list | None", legend_on=None,
) -> bool:
    """Healthy-vs-epileptogenic virtual-stim boxplots, one readout per axis.

    legend_on is the axis for the region key (default: the first).
    """
    metrics = [col for col, _, _ in _SEIZURE_PANELS]
    if seiz1 is None and seiz2 is None:
        for ax in axes:
            ax.set_visible(False)
        return False

    stats = _seizure_panel_stats(seiz1, seiz2, metrics, records or [])
    for ax, (col, lbl, log_y) in zip(axes, _SEIZURE_PANELS):
        _seizure_panel(ax, stats, col, lbl, color1, color2, log_y=log_y)

    ax_leg = legend_on if legend_on is not None else axes[0]
    ax_leg.legend(
        handles=[
            mpatches.Patch(facecolor="white", edgecolor="0.3", label="Healthy"),
            mpatches.Patch(facecolor="0.3", edgecolor="0.3", label="Epileptogenic"),
        ],
        loc="upper left", frameon=False, fontsize=LEGEND_SIZE,
        handlelength=1.0, labelspacing=0.2, borderpad=0.1, borderaxespad=0.2,
    )
    # Headroom so the legend clears the significance bracket.
    _lo, _hi = ax_leg.get_ylim()
    ax_leg.set_ylim(top=_hi + 0.20 * (_hi - _lo))
    for ax, (_col, _lbl, log_y) in zip(axes, _SEIZURE_PANELS):
        if not log_y:
            coarse_axis_ticks(ax, axis="y", decimals=0, min_ticks=3)
    return True


# ── anatomy panel ────────────────────────────────────────────────────────────────

def _load_brain_outline(subject: str, view: str = "axial",
                        path: "str | None" = None) -> "pd.DataFrame | None":
    """Brain silhouette polygons from scripts/huang/assets, or None."""
    p = Path(path) if path else (
        Path(get_root()) / "scripts" / "huang" / "assets"
        / f"{subject}_brain_outline_{view}.csv")
    return pd.read_csv(p) if p.exists() else None


def _outline_aspect(outline: "pd.DataFrame | None") -> float:
    """Width / height of the silhouette."""
    if outline is None or not len(outline):
        return 0.45
    w = float(outline["x"].max() - outline["x"].min())
    h = float(outline["y"].max() - outline["y"].min())
    return w / h if h > 0 else 0.45


def _load_anatomy(train_dir: "str | None", ood_site: "str | None" = None) -> "dict | None":
    """Subject, kept contacts (with epileptogenic and is_stim_member flags), stim sites and
    held-out site rows from the preproc CSVs, or None if anything is missing.
    """
    if not train_dir:
        return None
    cfg_path = next(Path(train_dir).glob("seed_*/config.json"), None)
    if cfg_path is None:
        return None
    s = json.load(open(cfg_path)).get("settings", {})
    subject = s.get("subject", "sub-2")
    data_dir = Path(add_root(s.get("preproc_dir", "data/huang_2023/preproc"))) / subject
    ch_path, site_path = data_dir / "channels.csv", data_dir / "stim_sites.csv"
    if not (ch_path.exists() and site_path.exists()):
        return None

    channels, sites = pd.read_csv(ch_path), pd.read_csv(site_path)
    keep = np.ones(len(channels), dtype=bool)
    if s.get("drop_bad", True):
        keep &= channels["good"].to_numpy()
    if s.get("require_location", True):
        keep &= channels["has_location"].to_numpy().astype(bool)
    site_keep = np.ones(len(sites), dtype=bool)
    if s.get("drop_white_matter", False):
        keep &= ~_is_white_matter(channels["Destrieux_label_text"])
        site_keep &= ~_is_white_matter(sites["region"])
    contacts, kept_sites = channels[keep].copy(), sites[site_keep].copy()

    members = set(kept_sites["member1"]) | set(kept_sites["member2"])
    contacts["is_stim_member"] = contacts["name"].isin(members)
    contacts["epileptogenic"] = contacts["seizure_zone"].notna()

    names = [n.strip() for n in str(ood_site or s.get("ood_test_channels", "")).split(",")
             if n.strip()]
    ood = kept_sites[kept_sites["site_name"].isin(names)] if names else kept_sites.iloc[:0]
    return {"subject": subject, "contacts": contacts, "sites": kept_sites, "ood": ood}


def _panel_anatomy(ax, outline: "pd.DataFrame | None", anat: "dict | None") -> bool:
    """Axial silhouette with implanted contacts; False if either input is missing.

    Colour encodes the seizure zone, fill encodes stim-site membership; the held-out
    site is starred at its pair's midpoint.
    """
    if outline is None or anat is None:
        ax.set_visible(False)
        return False

    for _, poly in outline.groupby("poly_id"):
        poly = poly.sort_values("order")
        ax.fill(poly["x"], poly["y"], facecolor="0.93", edgecolor="none", zorder=0)
        ax.plot(poly["x"], poly["y"], color="0.62", lw=0.6, zorder=1)

    c = anat["contacts"].dropna(subset=["x", "y"])
    colors = np.where(c["epileptogenic"], ZONE_EPI_COLOR, ZONE_HEALTHY_COLOR)
    rec, stim = ~c["is_stim_member"].to_numpy(), c["is_stim_member"].to_numpy()
    ax.scatter(c["x"][rec], c["y"][rec], s=_CONTACT_SIZE, facecolors="none",
               edgecolors=colors[rec], linewidths=0.5, zorder=3)
    ax.scatter(c["x"][stim], c["y"][stim], s=_CONTACT_SIZE, c=colors[stim],
               edgecolors="none", zorder=4)
    ood = anat["ood"].dropna(subset=["x", "y"])
    if len(ood):
        ax.scatter(ood["x"], ood["y"], s=_OOD_SITE_SIZE, marker="*", zorder=5,
                   linewidths=0.6, edgecolors="black",
                   facecolors=np.where(ood["seizure_zone"].notna(),
                                       ZONE_EPI_COLOR, "white"))

    x0, x1 = float(outline["x"].min()), float(outline["x"].max())
    y0, y1 = float(outline["y"].min()), float(outline["y"].max())
    mx, my = 0.02 * (x1 - x0), 0.02 * (y1 - y0)
    ax.set_xlim(x0 - mx, x1 + mx)
    ax.set_ylim(y0 - my, y1 + my)
    ax.axis("off")
    # Orientation key: anterior up, right to the right (native scanner axes).
    ax.annotate("", xy=(0.05, 0.99), xytext=(0.05, 0.87), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color="black",
                                shrinkA=0, shrinkB=0))
    ax.text(-0.02, 0.93, "A", transform=ax.transAxes, fontsize=LEGEND_SIZE,
            color="black", ha="right", va="center")
    ax.text(1.02, 0.93, "R", transform=ax.transAxes, fontsize=LEGEND_SIZE,
            color="black", ha="left", va="center")
    return True


def _key_box(fig, handles, labels, x_in: float, y_in: float,
             fig_w: float, fig_h: float, ncol: int = 2):
    """A figure-level key placed by its bottom-left corner, in inches."""
    return fig.legend(handles=handles, labels=labels, ncol=ncol, loc="lower left",
                      bbox_to_anchor=(x_in / fig_w, y_in / fig_h),
                      fontsize=LEGEND_SIZE, frameon=False,
                      handlelength=1.0, handletextpad=0.4, columnspacing=0.9,
                      labelspacing=0.25, borderpad=0.0, borderaxespad=0.0)


def _xlabel_bottom(fig, ax) -> float:
    """Inches from the canvas foot to the bottom of ax's x label (needs a drawn canvas)."""
    bb = ax.xaxis.label.get_window_extent(fig.canvas.get_renderer())
    return bb.y0 / fig.dpi


def _anatomy_handles() -> list:
    """Marker proxies for the anatomy key, in the order it reads."""
    handles = [
        plt.Line2D([0], [0], marker="o", ls="none", markersize=2.6,
                   markerfacecolor="none", markeredgecolor=ZONE_HEALTHY_COLOR,
                   markeredgewidth=0.5, label="Recording"),
        plt.Line2D([0], [0], marker="o", ls="none", markersize=2.6,
                   color=ZONE_HEALTHY_COLOR, markeredgecolor="none",
                   label="Stim. site"),
        plt.Line2D([0], [0], marker="o", ls="none", markersize=2.6,
                   color=ZONE_EPI_COLOR, markeredgecolor="none",
                   label="Epileptogenic"),
        plt.Line2D([0], [0], marker="*", ls="none", markersize=4.5,
                   markerfacecolor=ZONE_EPI_COLOR, markeredgecolor="black",
                   markeredgewidth=0.5, label="Held-out site"),
    ]
    return handles


def _anatomy_legend(fig, x_in: float, y_in: float, fig_w: float, fig_h: float):
    """The two-column anatomy key below the anatomy panel."""
    handles = _anatomy_handles()
    return _key_box(fig, handles, [h.get_label() for h in handles], x_in, y_in,
                    fig_w, fig_h)


def _anatomy_key_width(fig, fig_w: float, fig_h: float) -> float:
    """Width in inches of the anatomy key, measured by drawing and discarding it."""
    leg = _anatomy_legend(fig, 0.0, 0.0, fig_w, fig_h)
    fig.canvas.draw()
    w = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    leg.remove()
    return w


# ── figures ──────────────────────────────────────────────────────────────────────

def _build_huang_overview(
    train_dir1: "str | None", train_dir2: "str | None",
    d_sw_1: "str | None", d_sw_2: "str | None",
    seiz1: "pd.DataFrame | None", seiz2: "pd.DataFrame | None",
    name1: str, name2: str, color1: str, color2: str,
    eval_subdir: str, out_dir: str, fmt: str, name_suffix: str,
    seizure_records: "list | None" = None,
    sweep_records: "list | None" = None,
) -> None:
    """Overview: rollout | 2x2 comparison | 2x1 seizure, lettered a-g."""
    print("\n── Huang overview figure ────────────────────────────────────────────")

    fig_w = TEXT_WIDTH
    fig_h = TOP_PAD + 2 * OV_ROW_H + ROW_GAP + BOTTOM_PAD
    col_w = (fig_w - OV_LEFT_PAD - RIGHT_PAD - 3 * COL_GAP) / 4
    fig = plt.figure(figsize=(fig_w, fig_h))
    _ax = inch_axes(fig, fig_w, fig_h)

    def _x(col: int) -> float:
        return OV_LEFT_PAD + col * (col_w + COL_GAP)

    y_top, y_bot = BOTTOM_PAD + OV_ROW_H + ROW_GAP, BOTTOM_PAD
    ax_roll = _ax(_x(0), y_bot, col_w, 2 * OV_ROW_H + ROW_GAP)
    ax_tier_ycorr    = _ax(_x(1), y_top, col_w, OV_ROW_H)
    ax_tier_ycorr_r  = _ax(_x(2), y_top, col_w, OV_ROW_H)
    ax_seiz_top      = _ax(_x(3), y_top, col_w, OV_ROW_H)
    ax_sweep_ycorr   = _ax(_x(1), y_bot, col_w, OV_ROW_H)
    ax_sweep_ycorr_r = _ax(_x(2), y_bot, col_w, OV_ROW_H)
    ax_seiz_bottom   = _ax(_x(3), y_bot, col_w, OV_ROW_H)

    panel_timeseries(ax_roll, train_dir1, train_dir2, eval_subdir,
                     ("y_rollout_mean", "y_rollout_std"), name1, name2, color1, color2)
    panels_tier([ax_tier_ycorr, ax_tier_ycorr_r], train_dir1, train_dir2,
                ["y_corr", "y_corr_rollout"], name1, name2, color1, color2,
                eval_subdir, name_suffix, tier_labels=HUANG_TIER_LABELS)
    panels_sweep([ax_sweep_ycorr, ax_sweep_ycorr_r], d_sw_1, d_sw_2,
                 ["y_corr", "y_corr_rollout"], name1, name2, color1, color2,
                 eval_subdir, out_dir, fmt, records=sweep_records)
    _panels_seizure([ax_seiz_top, ax_seiz_bottom], seiz1, seiz2, color1, color2,
                    seizure_records)

    label_panels([(ax_roll, "a", col_w)], 2 * OV_ROW_H + ROW_GAP, dx=BARE_LBL_DX)
    label_panels([(ax_tier_ycorr, "b", col_w), (ax_tier_ycorr_r, "c", col_w),
                 (ax_seiz_top, "d", col_w)], OV_ROW_H)
    label_panels([(ax_sweep_ycorr, "e", col_w), (ax_sweep_ycorr_r, "f", col_w),
                 (ax_seiz_bottom, "g", col_w)], OV_ROW_H)

    path = os.path.join(out_dir, f"overview_{name_suffix}")
    save_figure(fig, path, fmt=fmt)
    plt.close(fig)
    print(f"  Saved: {path}.{fmt}")


def _build_huang_main(
    train_dir1: "str | None", train_dir2: "str | None",
    outline: "pd.DataFrame | None", anat: "dict | None",
    name1: str, name2: str, color1: str, color2: str,
    eval_subdir: str, out_dir: str, fmt: str, name_suffix: str,
) -> None:
    """One-row main figure: anatomy | rollout | recon. accuracy | rollout accuracy."""
    print("\n── Huang main figure ────────────────────────────────────────────────")

    fig_w = TEXT_WIDTH
    fig_h = TOP_PAD + HUANG_MAIN_PANEL_H + MAIN_BOTTOM_PAD
    fig = plt.figure(figsize=(fig_w, fig_h))
    _ax = inch_axes(fig, fig_w, fig_h)

    # Column (a) is as wide as the wider of its key and the silhouette, which is centred.
    x_key = MAIN_LEFT_PAD - BARE_LBL_DX
    key_w = _anatomy_key_width(fig, fig_w, fig_h)
    brain_w = HUANG_MAIN_PANEL_H * _outline_aspect(outline)
    col_a = max(key_w, brain_w)
    rest = (fig_w - x_key - RIGHT_PAD - col_a - MAIN_GAP_AB - 2 * MAIN_GAP_BOX)
    unit = rest / sum(MAIN_WEIGHTS)
    roll_w, box_w = unit * MAIN_WEIGHTS[0], unit * MAIN_WEIGHTS[1]

    y0 = MAIN_BOTTOM_PAD
    xs = [x_key + 0.5 * (col_a - brain_w)]          # (a), centred over its key
    xs.append(x_key + col_a + MAIN_GAP_AB)          # (b)
    xs.append(xs[-1] + roll_w + MAIN_GAP_BOX)       # (c)
    xs.append(xs[-1] + box_w + MAIN_GAP_BOX)        # (d)

    ax_brain    = _ax(xs[0], y0, brain_w, HUANG_MAIN_PANEL_H)
    ax_roll     = _ax(xs[1], y0, roll_w, HUANG_MAIN_PANEL_H)
    ax_ycorr    = _ax(xs[2], y0, box_w, HUANG_MAIN_PANEL_H)
    ax_roll_acc = _ax(xs[3], y0, box_w, HUANG_MAIN_PANEL_H)

    has_anat = _panel_anatomy(ax_brain, outline, anat)
    drawn = panel_timeseries(ax_roll, train_dir1, train_dir2, eval_subdir,
                             ("y_rollout_mean", "y_rollout_std"),
                             name1, name2, color1, color2,
                             channels=range(MAIN_ROLLOUT_CHANNELS), legend=True)
    panels_tier([ax_ycorr, ax_roll_acc], train_dir1, train_dir2,
                ["y_corr", "y_corr_rollout"], name1, name2, color1, color2,
                eval_subdir, name_suffix, xlabel=XLABEL_STIM_SITE, legend=True,
                tier_labels=HUANG_TIER_LABELS)
    for ax in (ax_ycorr, ax_roll_acc):
        ylabel_to_title(ax)

    # (a)'s label sits at the column's left edge, not the silhouette's.
    label_panels([(ax_brain, "a", brain_w)], HUANG_MAIN_PANEL_H,
                 dx=xs[0] - x_key, dy=LBL_DY_TITLE)
    label_panels([(ax_roll, "b", roll_w)], HUANG_MAIN_PANEL_H,
                 dx=BARE_LBL_DX, dy=LBL_DY_TITLE)
    label_panels([(ax_ycorr, "c", box_w), (ax_roll_acc, "d", box_w)],
                 HUANG_MAIN_PANEL_H, dx=MAIN_BOX_LBL_DX, dy=LBL_DY_TITLE)

    # Anatomy key bottom aligned with (b)'s x label.
    if has_anat:
        fig.canvas.draw()
        y_key = _xlabel_bottom(fig, ax_roll) if drawn else LEGEND_BASELINE
        _anatomy_legend(fig, x_key, y_key, fig_w, fig_h)

    path = os.path.join(out_dir, f"main_{name_suffix}")
    save_figure(fig, path, fmt=fmt)
    plt.close(fig)
    print(f"  Saved: {path}.{fmt}")


def _build_huang_d_sweeps(
    d_sw_1: "str | None", d_sw_2: "str | None",
    name1: str, name2: str, color1: str, color2: str,
    eval_subdir: str, out_dir: str, fmt: str, name_suffix: str,
    sweep_records: "list | None" = None,
) -> None:
    """Reconstruction and rollout accuracy against D."""
    print("\n── Huang latent-dimension sweep ─────────────────────────────────────")
    fig, axes, panel_w = two_panel_row()
    panels_sweep(axes, d_sw_1, d_sw_2, ["y_corr", "y_corr_rollout"],
                 name1, name2, color1, color2, eval_subdir, out_dir, fmt,
                 xlabel=XLABEL_D_MATH, legend=True, significance="asterisks",
                 records=sweep_records)
    label_panels(list(zip(axes, "ab", [panel_w] * 2)), SUP_PANEL_H)

    path = os.path.join(out_dir, f"supplement_{name_suffix}_d_sweeps")
    save_figure(fig, path, fmt=fmt)
    plt.close(fig)
    print(f"  Saved: {path}.{fmt}")


def _build_huang_seizure(
    seiz1: "pd.DataFrame | None", seiz2: "pd.DataFrame | None",
    color1: str, color2: str, out_dir: str, fmt: str, name_suffix: str,
    seizure_records: "list | None" = None,
) -> None:
    """Virtual stimulation of epileptogenic vs healthy pairs."""
    print("\n── Huang seizure-zone virtual stimulation ───────────────────────────")
    fig, axes, panel_w = two_panel_row()
    _panels_seizure(axes, seiz1, seiz2, color1, color2, seizure_records,
                    legend_on=axes[0])
    label_panels(list(zip(axes, "ab", [panel_w] * 2)), SUP_PANEL_H)

    path = os.path.join(out_dir, f"supplement_{name_suffix}_seizure_zones")
    save_figure(fig, path, fmt=fmt)
    plt.close(fig)
    print(f"  Saved: {path}.{fmt}")


# ── data export ──────────────────────────────────────────────────────────────────

def export_ood_site_distance(anat: "dict | None", out_dir: str) -> None:
    """Distance of each held-out stim site to the nearest held-in site.

    Euclidean between bipolar-pair midpoints, in mm, in native scanner space; the median
    and max over held-in sites are included for scale.
    """
    if anat is None or not len(anat.get("ood", [])):
        print("  No held-out stim site resolved; skipping the OOD distance export.")
        return

    cols = ["x", "y", "z"]
    sites = anat["sites"].dropna(subset=cols)
    ood = anat["ood"].dropna(subset=cols)
    held_in = sites[~sites["site_name"].isin(set(ood["site_name"]))]
    if not len(held_in):
        print("  No held-in stim sites with coordinates; skipping the OOD distance export.")
        return

    xyz = held_in[cols].to_numpy(dtype=float)
    rows = []
    for _, site in ood.iterrows():
        d = np.linalg.norm(xyz - site[cols].to_numpy(dtype=float), axis=1)
        near = held_in.iloc[int(np.argmin(d))]
        rows.append({
            "held_out_site": site["site_name"],
            "held_out_x": site["x"], "held_out_y": site["y"], "held_out_z": site["z"],
            "held_out_zone": site.get("seizure_zone"),
            "nearest_held_in_site": near["site_name"],
            "nearest_x": near["x"], "nearest_y": near["y"], "nearest_z": near["z"],
            "nearest_zone": near.get("seizure_zone"),
            "same_shaft": bool(str(site["site_name"]).split("-")[0].rstrip("0123456789")
                               == str(near["site_name"]).split("-")[0].rstrip("0123456789")),
            "distance_mm": float(d.min()),
            "n_held_in_sites": int(len(held_in)),
            "median_distance_mm": float(np.median(d)),
            "max_distance_mm": float(d.max()),
        })

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "huang_ood_site_distance.csv")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4g")
    print(f"  Saved held-out site distances: {path}")
    for r in rows:
        print(f"    {r['held_out_site']} -> nearest held-in {r['nearest_held_in_site']}: "
              f"{r['distance_mm']:.1f} mm "
              f"(median over {r['n_held_in_sites']} held-in sites: "
              f"{r['median_distance_mm']:.1f} mm)")


def export_seizure_per_seed(train_dirs: "dict[str, str | None]", attractor_subdir: str,
                            out_dir: str) -> None:
    """Summarise per-seed seizure_zone_summary.csv into huang_seizure_per_seed.csv.

    One row per (arm, readout, subset): number of significant seeds, sign consistency
    and effect-size range. train_dirs maps arm label -> group dir.
    """
    frames = []
    for arm, td in train_dirs.items():
        if not td:
            continue
        paths = sorted(Path(td).glob(f"seed_*/{attractor_subdir}/seizure_zone_summary.csv"))
        for p in paths:
            seed = int(p.parent.parent.name.split("_")[-1])
            df = pd.read_csv(p)
            df = df[(df["test"] == "mannwhitney")
                    & (df["readout"].isin(_SEIZURE_READOUTS))].copy()
            df["seed"] = seed
            df["arm"] = arm
            frames.append(df)
    if not frames:
        print("  No per-seed seizure_zone_summary.csv found; skipping per-seed export.")
        return
    alldf = pd.concat(frames, ignore_index=True)

    rows = []
    for (arm, readout, subset), g in alldf.groupby(["arm", "readout", "subset"]):
        p = g["p_value"].to_numpy()
        q = g["q_value"].to_numpy()
        d = g["cohen_d"].to_numpy()
        fin = np.isfinite(d)
        rows.append({
            "arm": arm, "readout": readout, "subset": subset,
            "n_seeds": int(len(g)),
            "n_sig_raw": int(np.sum((p < 0.05) & np.isfinite(p))),
            "n_sig_bh": int(np.sum((q < 0.05) & np.isfinite(q))),
            "consistent_sign": bool(fin.any() and np.unique(np.sign(d[fin])).size == 1),
            "cohen_d_min": float(np.min(d[fin])) if fin.any() else float("nan"),
            "cohen_d_median": float(np.median(d[fin])) if fin.any() else float("nan"),
            "cohen_d_max": float(np.max(d[fin])) if fin.any() else float("nan"),
        })
    out = pd.DataFrame(rows).sort_values(["arm", "subset", "readout"]).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "huang_seizure_per_seed.csv")
    out.to_csv(path, index=False, float_format="%.6g")
    n_arms = out["arm"].nunique()
    print(f"  Saved per-seed seizure summary: {path}  ({len(out)} rows, {n_arms} arm(s))")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--linear_train_dir", default=DEFAULT_TRAIN_DIRS[0],
                   help="Linear main-model group dir (seed_*/ and eval/)")
    p.add_argument("--nonlinear_train_dir", default=DEFAULT_TRAIN_DIRS[1],
                   help="Nonlinear main-model group dir")
    p.add_argument("--linear_d_sweep_dir", default=None,
                   help="Linear D-sweep dir (job_*/ parent; default: parent of the train dir)")
    p.add_argument("--nonlinear_d_sweep_dir", default=None,
                   help="Nonlinear D-sweep dir (default: parent of the train dir)")
    p.add_argument("--name_1", default="Linear", help="Display name for the linear model")
    p.add_argument("--name_2", default="Nonlinear", help="Display name for the nonlinear model")
    p.add_argument("--color_1", default=LINEAR_COLOR, help="Colour for the linear model")
    p.add_argument("--color_2", default=NONLINEAR_COLOR, help="Colour for the nonlinear model")
    p.add_argument("--eval_subdir", default="eval",
                   help="Subdir inside each job for eval outputs (default: eval)")
    p.add_argument("--attractor_subdir", default="attractor",
                   help="Subdir inside each seed dir for attractor outputs (default: attractor)")
    p.add_argument("--output_dir", default=None,
                   help="Directory for figures and stats (default: figures/huang/)")
    p.add_argument("--fmt", default="png", help="Figure format: png, svg, pdf (default: png)")
    p.add_argument("--figures", nargs="*",
                   default=["overview", "main", "d_sweeps", "seizure"],
                   choices=["overview", "main", "d_sweeps", "seizure"],
                   help="Which figures to render (default: all four)")
    p.add_argument("--ood_site", default=None,
                   help="Held-out stim site(s) to star in the anatomy panel, e.g. "
                        "'RC3-RC4'. Default: the seed config's ood_test_channels")
    p.add_argument("--brain_outline", default=None,
                   help="Brain silhouette CSV. Default: scripts/huang/assets/"
                        "<subject>_brain_outline_axial.csv")
    return p.parse_args()


def main() -> None:
    apply_paper_style()
    args = parse_args()

    out_dir = add_root(args.output_dir or os.path.join("figures", "huang"))
    fmt     = args.fmt.lstrip(".")
    os.makedirs(out_dir, exist_ok=True)

    train_1 = add_root(args.linear_train_dir)
    train_2 = add_root(args.nonlinear_train_dir)
    d_sw_1  = add_root(args.linear_d_sweep_dir) or str(Path(train_1).parent)
    d_sw_2  = add_root(args.nonlinear_d_sweep_dir) or str(Path(train_2).parent)

    name1, name2 = args.name_1, args.name_2
    color1, color2 = args.color_1, args.color_2

    seiz1 = _load_seizure_ensemble(train_1, args.attractor_subdir)
    seiz2 = _load_seizure_ensemble(train_2, args.attractor_subdir)
    if seiz1 is None or seiz2 is None:
        missing = [d for d, s in ((train_1, seiz1), (train_2, seiz2)) if s is None]
        print(f"  Warning: no seed_*/{args.attractor_subdir}/seizure_virtual_stim.csv "
              f"for {missing}; seizure column will be skipped where absent.")

    # These records drive both the panel asterisks and huang_stats.csv.
    seizure_records = compute_seizure_stats(seiz1, seiz2,
                                            [col for col, _, _ in _SEIZURE_PANELS])
    sweep_records = compute_sweep_stats(d_sw_1, d_sw_2, DATASET, args.eval_subdir)

    if "overview" in args.figures:
        _build_huang_overview(
            train_1, train_2, d_sw_1, d_sw_2, seiz1, seiz2,
            name1, name2, color1, color2,
            args.eval_subdir, out_dir, fmt, DATASET,
            seizure_records=seizure_records,
            sweep_records=sweep_records,
        )

    anat = _load_anatomy(train_2 or train_1, ood_site=args.ood_site)
    outline = (_load_brain_outline(anat["subject"], path=args.brain_outline)
               if anat else None)

    if "main" in args.figures:
        if anat is None or outline is None:
            print("  Warning: no anatomy CSVs or no brain outline asset; the main "
                  "figure's anatomy panel will be blank.")
        _build_huang_main(
            train_1, train_2, outline, anat,
            name1, name2, color1, color2,
            args.eval_subdir, out_dir, fmt, DATASET,
        )

    if "d_sweeps" in args.figures:
        _build_huang_d_sweeps(
            d_sw_1, d_sw_2, name1, name2, color1, color2,
            args.eval_subdir, out_dir, fmt, DATASET, sweep_records=sweep_records,
        )

    if "seizure" in args.figures:
        _build_huang_seizure(
            seiz1, seiz2, color1, color2, out_dir, fmt, DATASET,
            seizure_records=seizure_records,
        )

    export_realdata_stats(
        train_1, train_2, d_sw_1, d_sw_2,
        DATASET, args.eval_subdir, out_dir,
        extra_stats_records=seizure_records,
    )
    export_seizure_per_seed({"linear": train_1, "nonlinear": train_2},
                            args.attractor_subdir, out_dir)
    export_ood_site_distance(anat, out_dir)

    print(f"\nDone. Figures and stats in {out_dir}")


if __name__ == "__main__":
    main()
