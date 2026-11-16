"""Latent accuracy against the stimulation rate and protocol (timing x amplitude), and against
rollout/reconstruction accuracy; saddle figure plus point/line attractor figure and stats CSV.

python -m scripts.figures.figure_interventions --fmt pdf
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.io import add_root, available_dir, load_sweep, read_stim_scale, scale_records
from src.utils.plotting import (
    coarse_axis_ticks, pin_outliers, plot_sweep, robust_panel_range, significance_stars,
)
from src.utils.stats import eval_metric_by_seed, paired_comparison
from src.utils.style import (
    AXIS_LABEL_SIZE, LEGEND_SIZE, NONLINEAR_COLOR, PANEL_TITLE_SIZE, TEXT_WIDTH,
    TICK_LABEL_SIZE, TIMING_COLOR_POISSON, TIMING_COLOR_REGULAR,
    apply_paper_style, fit_to_ink, panel_label, save_figure,
)
from scripts.figures._common.constants import (
    DECISION_MODELS, METRIC_LABELS, SYSTEM_LABELS, XLABEL_N_INTERVENTIONS,
)
from scripts.figures._common.stats_export import bh_stamp, write_stats_csv


FIGURE = "interventions"
FIGURE_SUPP = "supp_fig_interventions_point_line"
METRIC = "x_pdc_aligned_corr"

SYSTEMS = {"saddle": "Saddle", "point": "PointAttractor", "line": "LineAttractor"}
SUPP_SYSTEMS = ["point", "line"]

SWEEP_PARAM = "components.stimulator.params.stim_rate"

XLABEL_AMPLITUDE = "Stim. amplitude"

DETAIL_METRICS = ["y_corr", "y_corr_rollout", "x_pdc_aligned_corr"]

# Geometry, in inches.
PANEL_H    = 1.00
LEFT_PAD   = 0.52
COL_GAP    = 0.55
TOP_PAD    = 0.16
BOTTOM_PAD = 0.42
LBL_DX     = 0.40
LBL_DY     = 0.03
AB_WIDTH   = 1.1                   # width of (a) and (b), relative to the square (c)
MARGIN     = 1 / 25.4               # 1 mm above and below the ink
ROW_GAP    = 0.67                   # supp figure: x labels + next row's header
ROW_TITLE_DY = 0.19                 # supp figure: row header baseline above the frame

PROXY_STYLE = {
    "y_corr":         dict(facecolors="none", edgecolors="0.6", linewidths=0.5),
    "y_corr_rollout": dict(facecolors="0.15", edgecolors="none"),
}
PROXY_NAMES = {"y_corr": "Recon.", "y_corr_rollout": "Rollout"}
YLABEL_LATENT = METRIC_LABELS[METRIC]
XLABEL_PROXY = "Recon./rollout accuracy"
PROXY_MARKER_S = 10                 # scatter marker area in (c), pt^2

LEGEND_KW = dict(frameon=False, fontsize=LEGEND_SIZE, handlelength=1.1,
                 handletextpad=0.4, borderpad=0.1, labelspacing=0.25,
                 borderaxespad=0.2)

# Stimulation protocol: 2x2 factorial of timing x amplitude.
TIMINGS = ["regular", "poisson"]
AMPS = ["constant", "diverse"]
CELLS = [f"{t}_{a}" for t in TIMINGS for a in AMPS]
TIMING_COLORS = {"regular": TIMING_COLOR_REGULAR, "poisson": TIMING_COLOR_POISSON}

OUTLIER_FACTOR = 1.5   # flag an extreme point if its gap to the rest exceeds
                       # OUTLIER_FACTOR x the span of all remaining points

BOX_WIDTH = 0.6        # total width of a dodged (regular, poisson) box pair
BOX_GAP = 0.18         # fraction of each box's slot left empty, so pairs don't touch
BOX_DODGE = BOX_WIDTH / 4   # seaborn's hue offset of each box from the amplitude x
BOX_JITTER = 0.05      # +/- jitter of the per-seed points, kept inside the box

# (factor, level_a, level_b, other_factor); level_b is group_1, so positive
# cohen_d / median_diff favours level_b.
FACTOR_SPEC = [
    ("timing", "regular", "poisson", "amplitude"),   # pair over (amplitude, seed)
    ("amplitude", "constant", "diverse", "timing"),  # pair over (timing, seed)
]


# ── stimulation protocol: data and statistics ─────────────────────────────────

def load_long_df(protocol_dir: str) -> pd.DataFrame:
    """Long-format df [system, timing, amplitude, seed, <DETAIL_METRICS>]; missing cells skipped."""
    rows = []
    for system in DECISION_MODELS:
        for cell in CELLS:
            csv = os.path.join(protocol_dir, system, cell, "job_0", "eval", "metrics.csv")
            if not os.path.isfile(csv):
                print(f"  missing (skipped): {csv}")
                continue
            df = pd.read_csv(csv)
            timing, amplitude = cell.split("_")
            for _, r in df.iterrows():
                row = {"system": system, "timing": timing,
                       "amplitude": amplitude, "seed": int(r.get("seed", -1))}
                for m in DETAIL_METRICS:
                    row[m] = float(r[m]) if m in df.columns else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _cell_series(sub: pd.DataFrame, metric: str, factor: str, level: str,
                 other: str) -> pd.Series:
    """Seed-wise values for one level of factor, indexed by (other factor, seed).

    Pairing on this index keeps a dropped NaN from mis-pairing the rest.
    """
    g = sub[sub[factor] == level]
    return g.set_index([other, "seed"])[metric].dropna()


def compute_main_effect_stats(long_df: pd.DataFrame) -> list:
    """Paired tests of the timing and amplitude main effects on METRIC per system, BH-FDR corrected."""
    records = []
    for system in DECISION_MODELS:
        sub = long_df[long_df["system"] == system]
        if sub.empty:
            continue
        for factor, lvl_a, lvl_b, other in FACTOR_SPEC:
            rec = paired_comparison(
                _cell_series(sub, METRIC, factor, lvl_b, other),
                _cell_series(sub, METRIC, factor, lvl_a, other),
                group_1=lvl_b, group_2=lvl_a,
                figure=FIGURE, panel="b", system=system, metric=METRIC,
                comparison=f"{factor}: {lvl_b} - {lvl_a}",
                x_name="factor", x_value=factor)
            if rec is not None:
                records.append(rec)
    return bh_stamp(records, f"{len(DECISION_MODELS)} systems x {len(FACTOR_SPEC)} factors")


def stars_lookup(records: list) -> dict:
    """(system, factor) -> star string, read off the BH-stamped records."""
    return {(r["system"], r["x_value"]): significance_stars(r["p_corrected"])
            for r in records}


# ── panels ───────────────────────────────────────────────────────────────────

def _no_data(ax, why: str) -> None:
    """Blank a panel whose runs are missing, and say so on stderr."""
    warnings.warn(why)
    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes,
            color="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def resolve_sweep_dir(name: str, base_dir: str, override: "str | None") -> str:
    """The nonlinear stim_rate sweep dir of one system: --sweep_dir, else convention."""
    return add_root(override or os.path.join(base_dir, SYSTEMS[name], "nonlinear"))


def panel_rate(ax, sweep_dir: "str | None") -> None:
    """Latent accuracy vs the number of training interventions, nonlinear model only."""
    if not available_dir(sweep_dir):
        return _no_data(ax, f"Stim_rate panel: no stim_rate sweep dir at {sweep_dir!r}; "
                            f"pass --sweep_dir <system>=<path>.")
    try:
        # stim_rate (Hz) -> expected number of interventions in the training set
        scale = read_stim_scale(sweep_dir)
    except FileNotFoundError as e:
        return _no_data(ax, f"Stim_rate panel: {e}")
    recs = scale_records(load_sweep(sweep_dir, metric=METRIC, config_param=SWEEP_PARAM,
                                    eval_subdir="eval"), scale)
    if not recs:
        return _no_data(ax, f"Stim_rate panel: no {METRIC} records under {sweep_dir!r}.")

    plot_sweep([([], recs, METRIC, "solid")], SWEEP_PARAM,
               "", "", "none", NONLINEAR_COLOR,
               output_dir=None, fmt=None, ax=ax,
               xlabel=XLABEL_N_INTERVENTIONS, ylabel=METRIC_LABELS[METRIC],
               metric_labels=METRIC_LABELS, legend=False,
               significance="asterisks", records=[])


def _seed_xpos(sub: pd.DataFrame, dodge: float, jitter: float = 0.0) -> dict:
    """(timing, amplitude, seed) -> x, dodged by timing and deterministically jittered.

    Markers and connector endpoints share these positions.
    """
    rng = np.random.default_rng(0)
    xpos = {}
    for ti, timing in enumerate(TIMINGS):
        off = (2 * ti - 1) * dodge
        for i, amp in enumerate(AMPS):
            seeds = sub[(sub["timing"] == timing) &
                        (sub["amplitude"] == amp)]["seed"].to_numpy()
            j = rng.uniform(-jitter, jitter, seeds.size) if jitter else \
                np.zeros(seeds.size)
            for s, js in zip(seeds, j):
                xpos[(timing, amp, int(s))] = i + off + float(js)
    return xpos


def _paired_connectors(ax, sub: pd.DataFrame, metric: str, xpos: dict,
                       outliers, zorder: float = 1) -> None:
    """Regular-poisson connector per (amplitude, seed), skipping pairs with a clipped outlier."""
    for amp in AMPS:
        reg = sub[(sub["timing"] == "regular") &
                  (sub["amplitude"] == amp)].set_index("seed")[metric]
        poi = sub[(sub["timing"] == "poisson") &
                  (sub["amplitude"] == amp)].set_index("seed")[metric]
        for s in reg.index.intersection(poi.index):
            rv, pv = reg[s], poi[s]
            if not (np.isfinite(rv) and np.isfinite(pv)):
                continue
            if rv in outliers or pv in outliers:
                continue
            ax.plot([xpos[("regular", amp, int(s))], xpos[("poisson", amp, int(s))]],
                    [rv, pv], color="0.65", lw=0.5, alpha=0.45, zorder=zorder)


def _box_panel(ax, sub: pd.DataFrame, metric: str, system: str,
               stars: dict, label_sep: str = "    ") -> None:
    """Grouped boxplots (x=amplitude, hue=timing) with paired per-seed points and stars."""
    if sub.empty or sub[metric].notna().sum() == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=TICK_LABEL_SIZE, color="gray")
        ax.set_xticks([])
        return

    ylo, yhi, outliers = robust_panel_range(sub[metric].to_numpy(),
                                             factor=OUTLIER_FACTOR)
    span = yhi - ylo
    sns.boxplot(data=sub, x="amplitude", y=metric, hue="timing",
                order=AMPS, hue_order=TIMINGS, palette=TIMING_COLORS,
                showfliers=False, width=BOX_WIDTH, gap=BOX_GAP, ax=ax,
                boxprops=dict(alpha=0.55), legend=False)
    xpos = _seed_xpos(sub, BOX_DODGE, jitter=BOX_JITTER)
    _paired_connectors(ax, sub, metric, xpos, outliers, zorder=2.5)

    # per-seed points at their jittered x, i.e. where the connectors end
    for ti, timing in enumerate(TIMINGS):
        off = (2 * ti - 1) * BOX_DODGE
        for i, amp in enumerate(AMPS):
            g = sub[(sub["timing"] == timing) &
                    (sub["amplitude"] == amp)].dropna(subset=[metric])
            v = g[metric].to_numpy()
            x = np.array([xpos[(timing, amp, int(s))] for s in g["seed"]])
            inl = ~np.isin(v, outliers)
            ax.scatter(x[inl], v[inl], s=10, color=TIMING_COLORS[timing],
                       alpha=0.9, edgecolor="black", linewidth=0.3, zorder=3)
            pin_outliers(ax, i + off, v[~inl], ylo, yhi, TIMING_COLORS[timing])

    ax.set_ylim(ylo, yhi + 0.10 * span)
    ax.set_xlabel("")
    ax.set_xticks(range(len(AMPS)))
    ax.set_xticklabels([a.capitalize() for a in AMPS])
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    label = (f"Timing {stars.get((system, 'timing'), 'ns')}{label_sep}"
             f"Amp. {stars.get((system, 'amplitude'), 'ns')}")
    ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=TICK_LABEL_SIZE - 1, color="#333333")


def _box_legend_handles() -> list:
    return [Patch(facecolor=TIMING_COLORS[t], alpha=0.7, label=t.capitalize())
            for t in TIMINGS]


def panel_protocol(ax, system: str, long_df, stars: dict,
                   legend_loc: str = "lower center") -> None:
    """Latent accuracy across the 2x2 timing x amplitude protocol factorial."""
    sub = long_df[long_df["system"] == system]
    if sub.empty:
        return _no_data(ax, f"Protocol panel: no stimulation-protocol metrics for {system}.")
    _box_panel(ax, sub, METRIC, system, stars, label_sep="  ")
    ax.set_ylabel(METRIC_LABELS[METRIC], fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel(XLABEL_AMPLITUDE, fontsize=AXIS_LABEL_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.08 * (hi - lo))
    ax.legend(handles=_box_legend_handles(), loc=legend_loc,
              **{**LEGEND_KW, "fontsize": TICK_LABEL_SIZE - 1})


def protocol_yticks(ax) -> None:
    """Ticks up to 1.0 only, not into the headroom above it."""
    lo, _ = ax.get_ylim()
    ticks = [abs(t) if t == 0 else t    # else -0.0
             for t in np.round(np.arange(-1.0, 1.01, 0.2), 1) if t >= lo]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.1f}" for t in ticks])


def pad_rate_xlim(ax) -> None:
    """Room for the last marker."""
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.04 * (hi - lo))


def _job_frame(eval_dir: Path) -> pd.DataFrame:
    """One job's per-seed DETAIL_METRICS; incomplete seeds dropped."""
    return pd.concat({m: eval_metric_by_seed(eval_dir, m) for m in DETAIL_METRICS},
                     axis=1).dropna()


def pooled_accuracies(system: str, sweep_dir: "str | None", long_df) -> pd.DataFrame:
    """Per-seed DETAIL_METRICS of every experiment condition, stacked."""
    frames = []
    if available_dir(sweep_dir):
        for job in sorted(Path(sweep_dir).glob("job_*")):
            if (job / "eval" / "metrics.csv").exists():
                frames.append(_job_frame(job / "eval"))
    frames.append(long_df.loc[long_df["system"] == system, DETAIL_METRICS].dropna())
    return pd.concat(frames, ignore_index=True)


def proxy_lim(df: pd.DataFrame) -> tuple:
    """Shared x/y range of a proxy scatter, up to 1, padded."""
    lo = min(0.0, np.floor(df[DETAIL_METRICS].to_numpy(float).min() * 10) / 10)
    return (lo - 0.05, 1.05)


def panel_proxy(ax, df: pd.DataFrame, lim: "tuple | None" = None,
                s: float = 5, compact_legend: bool = False) -> None:
    """Latent vs rollout and reconstruction accuracy, one point per seed."""
    if df.empty:
        return _no_data(ax, "Proxy panel: no pooled accuracies.")
    latent = df[METRIC].to_numpy(float)
    lim = lim or proxy_lim(df)
    ax.plot(lim, lim, color="0.5", linestyle=(0, (3, 3)), linewidth=0.6, zorder=0)

    handles = []
    sep = "\n" if compact_legend else " "
    for metric in ("y_corr", "y_corr_rollout"):
        acc = df[metric].to_numpy(float)
        r = np.corrcoef(acc, latent)[0, 1]
        mad = np.mean(np.abs(acc - latent))
        print(f"    {metric}: n={len(acc)}  r={r:.3f}  mean|metric - latent|={mad:.3f}")
        ax.scatter(acc, latent, s=s, zorder=2, clip_on=True, **PROXY_STYLE[metric])
        handles.append(mlines.Line2D(
            [], [], linestyle="none", marker="o", markersize=2.5 * np.sqrt(s / 5),
            markerfacecolor=PROXY_STYLE[metric]["facecolors"],
            markeredgecolor=PROXY_STYLE[metric]["edgecolors"],
            markeredgewidth=0.5, label=f"{PROXY_NAMES[metric]}{sep}($r$={r:.2f})"))

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(XLABEL_PROXY, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(YLABEL_LATENT, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=handles, loc="upper left", **LEGEND_KW)


def _row_axes(fig, widths: list, gaps: list, top: float = TOP_PAD) -> list:
    """One row of axes at the given widths and gaps, top inches below the top edge."""
    fig_w, fig_h = fig.get_size_inches()
    axes, x = [], LEFT_PAD
    y = fig_h - top - PANEL_H
    for i, w in enumerate(widths):
        axes.append(fig.add_axes([x / fig_w, y / fig_h, w / fig_w, PANEL_H / fig_h]))
        x += w + (gaps[i] if i < len(gaps) else 0)
    return axes


def _draw_row(axes: list, widths: list, letters: str, system: str,
              sweep_dir: "str | None", long_df, stars: dict,
              compact_legend: bool = False) -> None:
    """stim_rate sweep, protocol factorial and pooled proxy scatter of one system."""
    panel_rate(axes[0], sweep_dir)
    pad_rate_xlim(axes[0])
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    panel_protocol(axes[1], system, long_df, stars, legend_loc="lower right")
    pooled = pooled_accuracies(system, sweep_dir, long_df)
    panel_proxy(axes[2], pooled, s=PROXY_MARKER_S, compact_legend=compact_legend)

    for ax, letter, w in zip(axes, letters, widths):
        panel_label(ax, letter, x=-LBL_DX / w, y=1 + LBL_DY / PANEL_H)

    # after layout: tick locators depend on axes size
    for axis in ("x", "y"):
        coarse_axis_ticks(axes[0], axis=axis, decimals=2, min_ticks=3)
        coarse_axis_ticks(axes[2], axis=axis, decimals=1, min_ticks=3)
    protocol_yticks(axes[1])


def make_figure(system: str, sweep_dir: "str | None", long_df,
                stars: dict) -> plt.Figure:
    """1x3 grid: stim_rate sweep, protocol factorial, pooled proxy scatter."""
    fig_w = TEXT_WIDTH
    fig_h = TOP_PAD + PANEL_H + BOTTOM_PAD
    widths = [AB_WIDTH * PANEL_H] * 2 + [PANEL_H]

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = _row_axes(fig, widths, [COL_GAP, COL_GAP])
    _draw_row(axes, widths, "abc", system, sweep_dir, long_df, stars)
    fit_to_ink(fig, axes, MARGIN)
    return fig


def make_supp_figure(rows: list, long_df, stars: dict) -> plt.Figure:
    """make_figure's row once per (system, sweep_dir) in rows, titled by system."""
    fig_w = TEXT_WIDTH
    top = TOP_PAD + ROW_TITLE_DY
    fig_h = top + len(rows) * PANEL_H + (len(rows) - 1) * ROW_GAP + BOTTOM_PAD
    widths = [AB_WIDTH * PANEL_H] * 2 + [PANEL_H]

    fig = plt.figure(figsize=(fig_w, fig_h))
    all_axes = []
    for i, (system, sweep_dir) in enumerate(rows):
        axes = _row_axes(fig, widths, [COL_GAP, COL_GAP],
                         top=top + i * (PANEL_H + ROW_GAP))
        _draw_row(axes, widths, "abcdef"[3 * i:3 * i + 3], system, sweep_dir,
                  long_df, stars, compact_legend=True)
        axes[0].text(-LBL_DX / widths[0], 1 + ROW_TITLE_DY / PANEL_H,
                     SYSTEM_LABELS.get(system, system), transform=axes[0].transAxes,
                     fontsize=PANEL_TITLE_SIZE, fontweight="bold", va="baseline")
        all_axes += axes
    fit_to_ink(fig, all_axes, MARGIN)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decision_models_dir", default=None,
                   help="Base dir of the stim_rate sweeps "
                        "(default: outputs/decision_models)")
    p.add_argument("--sweep_dir", action="append", default=None,
                   metavar="SYSTEM=PATH",
                   help="Override the nonlinear stim_rate sweep dir of one system, "
                        "e.g. saddle=outputs/.../nonlinear. Repeatable.")
    p.add_argument("--protocol_dir", default=None,
                   help="Base dir of the stimulation-protocol outputs "
                        "(default: outputs/stimulation_protocol)")
    p.add_argument("--output_dir", default=None,
                   help="Where to write the figures and stats "
                        "(default: figures/interventions)")
    p.add_argument("--fmt", default="png", help="Figure format: png, svg, pdf")
    return p.parse_args()


def parse_sweep_overrides(pairs: "list | None") -> dict:
    """['saddle=path', ...] -> {'saddle': 'path'}."""
    out = {}
    for item in pairs or []:
        name, _, path = item.partition("=")
        if not path or name not in SYSTEMS:
            raise SystemExit(f"--sweep_dir expects SYSTEM=PATH with SYSTEM in "
                             f"{sorted(SYSTEMS)}; got {item!r}.")
        out[name] = path
    return out


def main() -> None:
    apply_paper_style()
    args = parse_args()
    fmt = args.fmt.lstrip(".")
    overrides = parse_sweep_overrides(args.sweep_dir)
    base_dir = add_root(args.decision_models_dir or os.path.join("outputs", "decision_models"))
    protocol_dir = add_root(args.protocol_dir
                            or os.path.join("outputs", "stimulation_protocol"))
    output_dir = add_root(args.output_dir or os.path.join("figures", "interventions"))
    os.makedirs(output_dir, exist_ok=True)

    long_df = load_long_df(protocol_dir)
    if long_df.empty:
        raise SystemExit(f"No stimulation-protocol metrics under {protocol_dir!r}.")
    records = compute_main_effect_stats(long_df)
    write_stats_csv(records, Path(output_dir) / f"{FIGURE}_stats.csv")
    stars = stars_lookup(records)

    def _sweep_dir(name):
        return resolve_sweep_dir(name, base_dir, overrides.get(name))

    def _save(fig, stem):
        path = os.path.join(output_dir, f"{stem}.{fmt}")
        save_figure(fig, path, fmt=fmt)
        plt.close(fig)
        print(f"Saved: {path}")

    _save(make_figure(SYSTEMS["saddle"], _sweep_dir("saddle"), long_df, stars),
          f"fig_{FIGURE}_saddle")
    rows = [(SYSTEMS[n], _sweep_dir(n)) for n in SUPP_SYSTEMS]
    _save(make_supp_figure(rows, long_df, stars), FIGURE_SUPP)


if __name__ == "__main__":
    main()
