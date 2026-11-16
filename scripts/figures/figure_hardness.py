"""Latent accuracy of iSSM (m1), SSM (m2) and input-conditioned SSM (m3) against intervention hardness alpha.

Asterisks mark alphas where m1 is significantly better than both other models (paired
Wilcoxon over seeds, BH-FDR). Writes hardness.{fmt}, hardness_stats.csv, hardness_per_seed.csv.

python -m scripts.figures.figure_hardness
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import add_root
from src.utils.plotting import coarse_axis_ticks
from src.utils.stats import (
    eval_metric_by_seed, paired_bootstrap_ci, paired_comparison, sem, wilcoxon_p_floor,
)
from src.utils.style import (
    AXIS_LABEL_SIZE, PANEL_TITLE_SIZE, TICK_LABEL_SIZE, TEXT_WIDTH,
    HARDNESS_MODEL_COLORS, apply_paper_style, fit_to_ink, panel_label, save_figure,
)
from scripts.figures._common.constants import (
    HARDNESS_MODEL_LABELS, METRIC_LABELS, SYSTEM_LABELS, XLABEL_ALPHA,
)
from scripts.figures._common.stats_export import (
    bh_stamp, higher_is_better, write_per_seed_csv, write_stats_csv,
)


# ── constants ────────────────────────────────────────────────────────────────

FIGURE = "hardness"
X_NAME = "alpha"

# Panel order; systems without eval data are dropped with a warning.
SYSTEMS = ["Saddle", "FitzHughNagumo"]

MODELS = ["m1", "m2", "m3"]
REF = "m1"                                    # the arm the others are tested against

METRIC = "x_pdc_aligned_corr"

# Geometry, in inches.
PANEL_W = 1.237
PANEL_H = 0.931
COL_GAP = 0.56
MARGIN  = 1 / 25.4    # 1 mm of white space above and below the ink
ASTERISK_SIZE = 9

N_BOOT = 10000

# Appended to the shared stats schema; see write_stats_csv(extra_columns=...).
STATS_EXTRA = [
    "unit_of_analysis", "p_floor", "n_nonzero_diff",
    "boot_ci_lo", "boot_ci_hi", "boot_n",
    "higher_is_better", "verdict", "panel_verdict", "asterisk_drawn",
]


# ── discovery ────────────────────────────────────────────────────────────────

def _alpha_of_job(job: Path) -> "float | None":
    """The alpha this job was trained at, read from a seed's saved config (not configs/)."""
    seeds = sorted(job.glob("seed_*"))
    if not seeds:
        return None
    cfg_path = seeds[0] / "config.json"
    if not cfg_path.is_file():
        return None
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    a = cfg.get("components", {}).get("system", {}).get("params", {}).get("alpha")
    return None if a is None else float(a)


def _sweep_series(base_dir: str, system: str, model: str,
                  metric: str) -> "dict[float, object]":
    """{alpha: per-seed Series} over one model's alpha sweep."""
    vdir = Path(base_dir) / system / model
    if not vdir.is_dir():
        return {}
    out = {}
    for job in sorted(vdir.glob("job_*")):
        alpha = _alpha_of_job(job)
        if alpha is None or not (job / "eval" / "metrics.csv").exists():
            continue
        vals = eval_metric_by_seed(job / "eval", metric)
        if len(vals):
            out[alpha] = vals
    return out


# ── stats ────────────────────────────────────────────────────────────────────

def compute_panel_stats(base_dir: str) -> list:
    """Paired tests of m1 vs each other model at each alpha, BH-FDR corrected."""
    records = []
    for system in SYSTEMS:
        ser_ref = _sweep_series(base_dir, system, REF, METRIC)
        panel = []
        for other in MODELS:
            if other == REF:
                continue
            ser_o = _sweep_series(base_dir, system, other, METRIC)
            for a in sorted(set(ser_ref) & set(ser_o)):
                # Reference first, so a positive Cohen's d reads as "iSSM better".
                rec = paired_comparison(
                    ser_ref[a], ser_o[a],
                    HARDNESS_MODEL_LABELS[REF], HARDNESS_MODEL_LABELS[other],
                    figure=FIGURE, panel=f"{system}/{METRIC} sweep",
                    system=system, metric=METRIC,
                    comparison=f"{REF}_vs_{other}",
                    x_name=X_NAME, x_value=a)
                if rec is None:
                    continue
                # Same pairs as paired_comparison.
                idx = ser_ref[a].index.intersection(ser_o[a].index)
                lo, hi = paired_bootstrap_ci(
                    ser_ref[a].loc[idx].to_numpy(float),
                    ser_o[a].loc[idx].to_numpy(float), n_boot=N_BOOT)
                rec.update(
                    unit_of_analysis="seed",
                    p_floor=wilcoxon_p_floor(rec["n_pairs"]),
                    boot_ci_lo=lo, boot_ci_hi=hi, boot_n=N_BOOT)
                panel.append(rec)
        if not panel:
            warnings.warn(f"No stats for {system}/{METRIC}: missing eval dirs.")
        records += bh_stamp(
            panel, f"panel:{system}/{METRIC} ({len(panel)} tests)")
    return stamp_verdicts(records)


def _favours_ref(rec: dict) -> bool:
    """True if this row's point estimate favours the reference arm (m1).

    cohen_d is of (m1 - other); median_diff is used when d is undefined (zero spread).
    """
    diff = rec["cohen_d"]
    if diff is None or np.isnan(diff):
        diff = rec["median_diff"]
    return diff > 0 if rec["higher_is_better"] else diff < 0


def stamp_verdicts(records: list) -> list:
    """Stamp direction-aware verdicts; asterisks mark alphas where m1 beats every other model."""
    for r in records:
        r["higher_is_better"] = higher_is_better(r["metric"])
        r["verdict"] = ("ns" if not r["significant"]
                        else f"{REF}_better" if _favours_ref(r) else f"{REF}_worse")

    # "all" counts every other model, so a cell missing an arm never reads as beats_all.
    n_expected = len(MODELS) - 1
    key = lambda r: (r["system"], r["metric"], r["x_value"])
    for k in dict.fromkeys(key(r) for r in records):
        cell = [r for r in records if key(r) == k]
        n_better = sum(v["verdict"] == f"{REF}_better" for v in cell)
        n_worse = sum(v["verdict"] == f"{REF}_worse" for v in cell)
        if n_better and n_worse:
            verdict = "mixed"
        elif n_better:
            verdict = f"{REF}_beats_" + ("all" if n_better == n_expected else "some")
        elif n_worse:
            verdict = f"{REF}_loses_" + ("all" if n_worse == n_expected else "some")
        else:
            verdict = "ns"
        for r in cell:
            r["panel_verdict"] = verdict
            r["asterisk_drawn"] = verdict == f"{REF}_beats_all"
    return records


def significant_alphas(records: list, system: str, metric: str) -> set:
    """Alphas where m1 beats every other model (see stamp_verdicts)."""
    return {r["x_value"] for r in records
            if r["system"] == system and r["metric"] == metric and r["asterisk_drawn"]}


def collect_per_seed_rows(base_dir: str) -> list:
    """Raw per-seed values behind every plotted point."""
    rows = []
    for system in SYSTEMS:
        for model in MODELS:
            for a, ser in sorted(_sweep_series(base_dir, system, model, METRIC).items()):
                for seed, val in ser.items():
                    rows.append(dict(figure=FIGURE, system=system, variant=model,
                                     metric=METRIC, x_name=X_NAME, x_value=a,
                                     seed=int(seed), value=float(val)))
    return rows


def check_coverage(base_dir: str) -> None:
    """Warn unless every arm of every alpha was evaluated on the same seeds."""
    for system in SYSTEMS:
        seeds = {b: {a: set(v.index) for a, v in
                     _sweep_series(base_dir, system, b, METRIC).items()}
                 for b in MODELS}
        alphas = sorted({a for per_alpha in seeds.values() for a in per_alpha})
        for a in alphas:
            per_arm = {b: seeds[b].get(a, set()) for b in MODELS}
            if len({frozenset(s) for s in per_arm.values()}) != 1:
                warnings.warn(
                    f"{system} @ alpha={a}: arms were evaluated on different seeds "
                    f"({ {b: len(s) for b, s in per_arm.items()} }); the paired tests "
                    f"use each contrast's own intersection.")
        n = {len(s) for per_alpha in seeds.values() for s in per_alpha.values()}
        print(f"  {system}: {len(alphas)} alphas x {len(MODELS)} arms, "
              f"{sorted(n)} seeds per cell")


# ── figure ───────────────────────────────────────────────────────────────────

def _plot_hardness_panel(ax, series_by_model: dict, sig_alphas: set,
                         metric: str) -> None:
    """One panel: mean +/- sem vs alpha, one line per model, asterisks on top."""
    tops = []
    for model in MODELS:
        ser = series_by_model.get(model, {})
        if not ser:
            continue
        xs = sorted(ser)
        means = [float(ser[a].mean()) for a in xs]
        sems = [sem(ser[a].to_numpy()) for a in xs]
        ax.errorbar(xs, means, yerr=sems, color=HARDNESS_MODEL_COLORS[model],
                    marker="o", markersize=4, linewidth=1.5,
                    capsize=3, capthick=1.0, elinewidth=1.0)
        tops += [m + s for m, s in zip(means, sems)]

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel(XLABEL_ALPHA, fontsize=AXIS_LABEL_SIZE)

    # One asterisk row above all error bars.
    if sig_alphas and tops:
        ylo, yhi = ax.get_ylim()
        span = yhi - ylo
        y_ast = max(tops) + 0.04 * span
        for a in sig_alphas:
            ax.text(a, y_ast, "*", ha="center", va="bottom",
                    fontsize=ASTERISK_SIZE, color="black")
        ax.set_ylim(ylo, max(yhi, y_ast + 0.14 * span))


def _legend_handles() -> list:
    return [mlines.Line2D([], [], color=HARDNESS_MODEL_COLORS[b], marker="o",
                          markersize=4, label=HARDNESS_MODEL_LABELS[b])
            for b in MODELS]


def make_figure(base_dir: str, records: list) -> plt.Figure:
    """One row of panels, one per system, legend on the right, centred."""
    panels = [(system, METRIC) for system in SYSTEMS]
    fig_w, fig_h = TEXT_WIDTH, 2 * PANEL_H    # refitted to the ink below
    x0 = (fig_w - len(panels) * PANEL_W - (len(panels) - 1) * COL_GAP) / 2

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for i, (system, metric) in enumerate(panels):
        x = x0 + i * (PANEL_W + COL_GAP)
        ax = fig.add_axes([x / fig_w, (fig_h - PANEL_H) / 2 / fig_h,
                           PANEL_W / fig_w, PANEL_H / fig_h])
        series = {b: _sweep_series(base_dir, system, b, metric) for b in MODELS}
        _plot_hardness_panel(ax, series, significant_alphas(records, system, metric),
                             metric)
        ax.set_title(SYSTEM_LABELS.get(system, system),
                     fontsize=PANEL_TITLE_SIZE, loc="left")
        panel_label(ax, "abcdefghi"[i])
        axes.append(ax)

    axes[-1].legend(handles=_legend_handles(), loc="center left",
                    bbox_to_anchor=(1.04, 0.5), frameon=False,
                    fontsize=TICK_LABEL_SIZE, handlelength=1.6)
    for ax in axes:
        for axis in ("x", "y"):
            coarse_axis_ticks(ax, axis=axis, decimals=2, min_ticks=3,
                              fontsize=TICK_LABEL_SIZE)
    fit_to_ink(fig, axes, MARGIN)
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", default=None,
                   help="Base dir of the intervention-hardness outputs "
                        "(default: outputs/intervention_hardness)")
    p.add_argument("--output_dir", default=None,
                   help="Output dir for the figure and stats (default: figures/hardness)")
    p.add_argument("--systems", nargs="*", default=None,
                   help=f"Systems to plot as panels (default: {' '.join(SYSTEMS)})")
    p.add_argument("--fmt", default="png", help="Figure format: png, svg, pdf")
    return p.parse_args()


def main() -> None:
    global SYSTEMS
    args = parse_args()
    apply_paper_style()
    base_dir = add_root(args.base_dir or os.path.join("outputs", "intervention_hardness"))
    output_dir = add_root(args.output_dir or os.path.join("figures", "hardness"))
    fmt = args.fmt.lstrip(".")
    if args.systems:
        SYSTEMS = list(args.systems)
    os.makedirs(output_dir, exist_ok=True)

    have = [s for s in SYSTEMS
            if any(_sweep_series(base_dir, s, b, METRIC) for b in MODELS)]
    for s in SYSTEMS:
        if s not in have:
            warnings.warn(f"No eval data for system {s!r} under {base_dir}; "
                          f"dropping its panel.")
    if not have:
        raise SystemExit(f"No eval data for any of {SYSTEMS} under {base_dir}.")
    SYSTEMS = have

    # The figure draws its asterisks from these records.
    print("── Statistics ──")
    check_coverage(base_dir)
    records = compute_panel_stats(base_dir)
    for r in records:  # invariant the plotter relies on
        assert r["significant"] == (r["p_corrected"] < r["alpha"]), r
    write_stats_csv(records, Path(output_dir) / f"{FIGURE}_stats.csv",
                    extra_columns=STATS_EXTRA)
    write_per_seed_csv(collect_per_seed_rows(base_dir),
                       Path(output_dir) / f"{FIGURE}_per_seed.csv")

    print("── Figure ──")
    fig = make_figure(base_dir, records)
    path = os.path.join(output_dir, f"{FIGURE}.{fmt}")
    save_figure(fig, path, fmt=fmt)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
