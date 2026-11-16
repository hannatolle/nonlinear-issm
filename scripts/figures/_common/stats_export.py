"""Stats and per-seed CSV export shared by the figure scripts.

<name>_stats.csv holds one row per statistical test (p_corrected is what the plotted
asterisks use); <name>_per_seed.csv holds the raw per-seed values behind each point.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_sweep, available_dir
from src.utils.plotting import _bh_fdr
from src.utils.stats import (
    ALPHA, wilcoxon_p_floor, paired_array_comparison, paired_bootstrap_ci,
    paired_comparison, eval_metric_by_seed, compute_sweep_sig_records,
)


def bh_stamp(records: list, scope: str, alpha: float = ALPHA) -> list:
    """BH-FDR correct records as one family; stamp p_corrected and significant.

    scope names the family in the exported CSV. The plotter reads significant off the
    same records, so the CSV and the plotted asterisks agree.
    """
    if not records:
        return records
    q = _bh_fdr(np.array([r["p_raw"] for r in records]))
    for r, qq in zip(records, q):
        r["correction"] = "BH-FDR"
        r["correction_scope"] = scope
        r["n_comparisons_in_scope"] = len(records)
        r["p_corrected"] = float(qq)
        r["alpha"] = alpha
        r["significant"] = bool(qq < alpha)
    return records


# One schema for every test row, across the figure scripts.
STATS_COLUMNS = [
    "figure", "panel", "system", "metric", "comparison",
    "x_name", "x_value",
    "group_1", "group_2", "n_pairs",
    "mean_1", "sem_1", "mean_2", "sem_2", "median_diff",
    "test", "test_statistic",
    "p_raw", "correction", "correction_scope", "n_comparisons_in_scope",
    "p_corrected", "alpha", "significant",
    "cohen_d",
]

PER_SEED_COLUMNS = [
    "figure", "system", "variant", "metric", "x_name", "x_value", "seed", "value",
]

# Metrics where a lower value is better. `cohen_d` is always signed as
# (group_1 - group_2), so reading a row as a win or a loss needs this too.
LOWER_IS_BETTER_TOKENS = frozenset(
    {"mae", "mse", "rmse", "nrmse", "error", "err", "loss"})


def higher_is_better(metric: str) -> bool:
    """True if a larger value of metric is the better result."""
    return not (LOWER_IS_BETTER_TOKENS & set(metric.split("_")))


def write_stats_csv(records: list, path: Path,
                    extra_columns: "list | None" = None) -> None:
    """Write test records using the shared schema; extra_columns follow STATS_COLUMNS."""
    if not records:
        print(f"  No stats records; skipping {path}")
        return
    df = pd.DataFrame(records)
    cols = STATS_COLUMNS + list(extra_columns or [])
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    df[cols].to_csv(path, index=False, float_format="%.6g")
    n_sig = int(df["significant"].sum())
    print(f"  Saved stats: {path}  ({len(df)} tests, {n_sig} significant)")


def write_per_seed_csv(rows: list, path: Path) -> None:
    """Write the raw per-seed values behind every plotted point."""
    if not rows:
        print(f"  No per-seed rows; skipping {path}")
        return
    df = pd.DataFrame(rows)[PER_SEED_COLUMNS]
    df.to_csv(path, index=False, float_format="%.8g")
    print(f"  Saved per-seed data: {path}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Real-data stats export
# ─────────────────────────────────────────────────────────────────────────────

# Metrics reported for every real-data panel.
_REALDATA_METRICS = ["y_corr", "y_corr_rollout"]

# Extra stats columns beyond the shared STATS_COLUMNS schema.
_STATS_EXTRA_COLUMNS = ["unit_of_analysis", "p_floor", "n_nonzero_diff"]

# Tier display order (real-data multi-tier eval).
_TIER_ORDER = ["Control", "Stim", "Stim_OOD"]


# ── sweeps over a config parameter ─────────────────────────────────────────────

# Appended to STATS_COLUMNS by a figure that reports sweep tests.
SWEEP_STATS_EXTRA = [
    "sweep", "x_value_raw", "unit_of_analysis",
    "n_nonzero_diff", "p_floor", "boot_ci_lo", "boot_ci_hi", "boot_n",
    "higher_is_better", "eval_subdir",
]

N_BOOT = 10000


def resolve_dotted_param(cfg: dict, dotted: str):
    """Value of a dotted config path (e.g. components.stimulator.params.stim_rate)."""
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def jobs_by_x(sweep_dir: "str | None", config_param: str, eval_subdir: str) -> dict:
    """{raw swept value: job dir} for every job under sweep_dir that has eval results.

    Keys are the configured values (e.g. D, or stim_rate in Hz), not rescaled plot x.
    """
    out = {}
    if not available_dir(sweep_dir):
        return out
    for job_dir in sorted(Path(sweep_dir).glob("job_*")):
        cfg_path = next(iter(sorted(job_dir.glob("seed_*/config.json"))), None)
        if cfg_path is None:
            continue
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        try:
            x = resolve_dotted_param(cfg, config_param)
        except (KeyError, TypeError):
            continue
        if (job_dir / eval_subdir / "metrics.csv").exists():
            out[x] = job_dir
    return out


def job_metric_by_seed(job_dir: "str | Path", metric: str,
                       eval_subdir: str) -> pd.Series:
    """Per-seed values of a metric from a job's eval metrics.csv, indexed by seed."""
    eval_dir = Path(job_dir) / eval_subdir
    if not (eval_dir / "metrics.csv").exists():
        return pd.Series(dtype=float)
    return eval_metric_by_seed(eval_dir, metric)


def sweep_panel_records(jobs_lin: dict, jobs_nonlin: dict, *, figure: str, panel: str,
                        system: str, metric: str, x_name: str, sweep: str,
                        eval_subdir: str, n_boot: int = N_BOOT) -> list:
    """One paired nonlinear-vs-linear record per shared x, uncorrected (see bh_stamp)."""
    records = []
    for x_raw in sorted(set(jobs_lin) & set(jobs_nonlin)):
        s_nl = job_metric_by_seed(jobs_nonlin[x_raw], metric, eval_subdir)
        s_lin = job_metric_by_seed(jobs_lin[x_raw], metric, eval_subdir)
        rec = paired_comparison(
            s_nl, s_lin, "nonlinear", "linear",
            figure=figure, panel=panel, system=system, metric=metric,
            comparison="nonlinear_vs_linear",
            x_name=x_name, x_value=float(x_raw))
        if rec is None:
            warnings.warn(f"{system}/{sweep}/{metric}@{x_raw}: no testable pairs; "
                          f"skipping")
            continue

        idx = s_nl.index.intersection(s_lin.index)
        a = s_nl.loc[idx].to_numpy(float)
        b = s_lin.loc[idx].to_numpy(float)
        lo, hi = paired_bootstrap_ci(a, b, n_boot=n_boot)
        rec.update(
            sweep=sweep,
            x_value_raw=float(x_raw),
            unit_of_analysis="seed",
            # The signed-rank test discards exactly-zero differences, so the p-value and
            # its floor are governed by n_nonzero_diff, not by n_pairs.
            p_floor=wilcoxon_p_floor(rec["n_nonzero_diff"]),
            boot_ci_lo=lo, boot_ci_hi=hi, boot_n=n_boot,
            higher_is_better=higher_is_better(metric),
            eval_subdir=eval_subdir,
        )
        records.append(rec)
    return records


def _jobs_by_D(sweep_dir: "str | None", eval_subdir: str) -> "dict[int, Path]":
    """Map latent dimension D -> job_* dir for every job under sweep_dir with eval."""
    out: dict[int, Path] = {}
    if not available_dir(sweep_dir):
        return out
    for job_dir in sorted(Path(sweep_dir).glob("job_*")):
        seed_dirs = sorted(job_dir.glob("seed_*"))
        if not seed_dirs:
            continue
        cfg_path = seed_dirs[0] / "config.json"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        try:
            D = int(cfg["settings"]["D"])
        except (KeyError, TypeError, ValueError):
            continue
        if (job_dir / eval_subdir / "metrics.csv").exists():
            out[D] = job_dir
    return out


def _sweep_stats_family(
    d_sw_1: "str | None", d_sw_2: "str | None",
    dataset: str, eval_subdir: str, metric: str, scope: str,
) -> list:
    """Paired sweep tests for one metric over D, BH-corrected as one family.

    Significance and corrected p come from compute_sweep_sig_records; descriptives are
    added from the same per-seed arrays. d_sw_1 is the linear arm, d_sw_2 the nonlinear.
    """
    kw = dict(config_param="settings.D", eval_subdir=eval_subdir)
    r1 = load_sweep(d_sw_1, metric, **kw)  # linear
    r2 = load_sweep(d_sw_2, metric, **kw)  # nonlinear

    sig_recs = compute_sweep_sig_records([(r1, r2, metric, "solid")])
    sig_by = {r["x"]: r for r in sig_recs}

    records = []
    map_lin = {r[0]: r[3] for r in r1}
    map_nl  = {r[0]: r[3] for r in r2}
    for x in sorted(set(map_lin) & set(map_nl)):
        lin_arr, nl_arr = map_lin[x], map_nl[x]
        rec = paired_array_comparison(
            nl_arr, lin_arr, "nonlinear", "linear",
            figure="figure_realdata", panel="d_sweep", system=dataset, metric=metric,
            comparison="nonlinear_vs_linear", x_name="D", x_value=int(x),
            unit_of_analysis="seed")
        if rec is None:
            continue
        sr = sig_by.get(x)
        rec["p_floor"] = wilcoxon_p_floor(rec["n_pairs"])
        rec["correction"] = "BH-FDR"
        rec["correction_scope"] = scope
        rec["n_comparisons_in_scope"] = len(sig_recs)
        rec["alpha"] = ALPHA
        if sr is not None:
            # Same arrays and test, so raw p must match (two-sided is orientation-invariant).
            assert abs(sr["p_raw"] - rec["p_raw"]) < 1e-9, (metric, x, sr, rec)
            rec["test"] = sr["test"]
            rec["test_statistic"] = sr["test_statistic"]
            rec["p_corrected"] = sr["p_corrected"]
            rec["significant"] = sr["significant"]
        else:
            rec["p_corrected"] = float("nan")
            rec["significant"] = False
        records.append(rec)
    return records


def compute_sweep_stats(
    d_sw_1: "str | None", d_sw_2: "str | None",
    dataset: str, eval_subdir: str,
) -> list:
    """Paired sweep tests (nonlinear vs linear), one row per (metric, D)."""
    if not (available_dir(d_sw_1) and available_dir(d_sw_2)):
        return []
    records = []
    for m in _REALDATA_METRICS:
        records += _sweep_stats_family(
            d_sw_1, d_sw_2, dataset, eval_subdir, m, f"D-sweep over D for {m} ({dataset})")
    return records


def compute_tier_stats(
    train_1: "str | None", train_2: "str | None",
    dataset: str, eval_subdir: str, metrics: "list | None" = None,
) -> list:
    """Paired per-tier tests (nonlinear vs linear), one row per (metric, tier).

    Wilcoxon signed-rank across the trials of a tier (matched on trial_idx), BH-FDR
    corrected. train_1 is linear, train_2 nonlinear.
    """
    if not (train_1 and train_2):
        return []
    csv_lin = Path(train_1) / eval_subdir / "multi_tier_eval_metrics.csv"
    csv_nl  = Path(train_2) / eval_subdir / "multi_tier_eval_metrics.csv"
    if not (csv_lin.exists() and csv_nl.exists()):
        return []
    merged = pd.read_csv(csv_lin).merge(
        pd.read_csv(csv_nl), on=["trial_idx", "trial_tier"], suffixes=("_lin", "_nl"))
    tiers = [t for t in _TIER_ORDER if t in set(merged["trial_tier"])]
    tiers += sorted(set(merged["trial_tier"]) - set(tiers))

    records = []
    for m in metrics or _REALDATA_METRICS:
        col_lin, col_nl = f"{m}_lin", f"{m}_nl"
        if col_lin not in merged.columns or col_nl not in merged.columns:
            continue
        recs_m = []
        for tier in tiers:
            sub = merged.loc[merged["trial_tier"] == tier, [col_lin, col_nl]].dropna()
            rec = paired_array_comparison(
                sub[col_nl].values, sub[col_lin].values, "nonlinear", "linear",
                figure="figure_realdata", panel="tier_comparison", system=dataset,
                metric=m, comparison="nonlinear_vs_linear", x_name="tier", x_value=tier,
                unit_of_analysis="trial")
            if rec is None:
                continue
            rec["p_floor"] = wilcoxon_p_floor(rec["n_pairs"])
            recs_m.append(rec)
        bh_stamp(recs_m, f"tier Control-vs-Stim within {m} ({dataset} panels c,d)")
        records += recs_m
    return records


def collect_per_seed_rows(
    d_sw_1: "str | None", d_sw_2: "str | None",
    dataset: str, eval_subdir: str,
) -> list:
    """Per-seed values behind the sweep panels, one row per (arm, D, seed, metric)."""
    rows = []
    for arm, sw in [("linear", d_sw_1), ("nonlinear", d_sw_2)]:
        for D, job_dir in _jobs_by_D(sw, eval_subdir).items():
            df = pd.read_csv(job_dir / eval_subdir / "metrics.csv")
            metric_cols = [c for c in _REALDATA_METRICS if c in df.columns]
            for _, r in df.iterrows():
                for metric in metric_cols:
                    val = r[metric]
                    if pd.isna(val):
                        continue
                    rows.append({
                        "figure": "figure_realdata", "system": dataset, "variant": arm,
                        "metric": metric, "x_name": "D", "x_value": int(D),
                        "seed": int(r["seed"]), "value": float(val),
                    })
    return rows


def collect_per_trial_rows(
    train_1: "str | None", train_2: "str | None",
    dataset: str, eval_subdir: str,
) -> list:
    """Per-trial tier metrics, one row per (arm, trial, tier, metric)."""
    rows = []
    for arm, td in [("linear", train_1), ("nonlinear", train_2)]:
        if not td:
            continue
        p = Path(td) / eval_subdir / "multi_tier_eval_metrics.csv"
        if not p.exists():
            continue
        D = _D_of_dir(td)
        df = pd.read_csv(p)
        metric_cols = [c for c in _REALDATA_METRICS if c in df.columns]
        for _, r in df.iterrows():
            for metric in metric_cols:
                val = r[metric]
                if pd.isna(val):
                    continue
                rows.append({
                    "figure": "figure_realdata", "system": dataset, "variant": arm,
                    "D": D, "trial_idx": int(r["trial_idx"]),
                    "trial_tier": r["trial_tier"], "metric": metric,
                    "value": float(val),
                })
    return rows


def _D_of_dir(train_dir: str) -> "int | None":
    """Latent dimension D read from the first seed_*/config.json under train_dir."""
    cfg_path = next(Path(train_dir).glob("seed_*/config.json"), None)
    if cfg_path is None:
        return None
    with open(cfg_path) as f:
        try:
            return int(json.load(f)["settings"]["D"])
        except (KeyError, TypeError, ValueError):
            return None


def export_realdata_stats(
    train_1: "str | None", train_2: "str | None",
    d_sw_1: "str | None", d_sw_2: "str | None",
    dataset: str, eval_subdir: str, stats_dir: str,
    extra_stats_records: "list | None" = None,
) -> None:
    """Write {dataset}_stats.csv, {dataset}_per_seed.csv and {dataset}_per_trial.csv.

    extra_stats_records must already be BH-stamped; their extra columns are added to the CSV.
    """
    print("\n── Stats export ─────────────────────────────────────────────────────")
    os.makedirs(stats_dir, exist_ok=True)

    stats = (compute_tier_stats(train_1, train_2, dataset, eval_subdir)
             + compute_sweep_stats(d_sw_1, d_sw_2, dataset, eval_subdir)
             + list(extra_stats_records or []))
    for r in stats:  # invariant the plotter relies on
        assert bool(r.get("significant")) == bool(r.get("p_corrected", 1.0) < r.get("alpha", ALPHA)), r
    extra_cols = list(_STATS_EXTRA_COLUMNS)
    for rec in (extra_stats_records or []):
        for k in rec:
            if k not in STATS_COLUMNS and k not in extra_cols:
                extra_cols.append(k)
    write_stats_csv(stats, Path(stats_dir) / f"{dataset}_stats.csv",
                    extra_columns=extra_cols)

    write_per_seed_csv(collect_per_seed_rows(d_sw_1, d_sw_2, dataset, eval_subdir),
                       Path(stats_dir) / f"{dataset}_per_seed.csv")

    per_trial = collect_per_trial_rows(train_1, train_2, dataset, eval_subdir)
    if per_trial:
        pd.DataFrame(per_trial).to_csv(
            Path(stats_dir) / f"{dataset}_per_trial.csv", index=False,
            float_format="%.8g")
        print(f"  Saved per-trial data: {dataset}_per_trial.csv  ({len(per_trial)} rows)")
