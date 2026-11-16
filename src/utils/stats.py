"""Seeding and paired statistical tests over per-seed eval metrics."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import random
from scipy.stats import wilcoxon

ALPHA = 0.05


# Reproducibility --------------------------------------------------------------

def fix_random_seed(seed: int):
    '''Seed python, numpy and torch, and make cuDNN deterministic.'''
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Paired tests over eval metrics -----------------------------------------------

def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> "tuple[float, float] | None":
    """Paired Wilcoxon signed-rank test; returns (W, p) or None if not testable."""
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[:n], b[:n]
    try:
        stat, p = wilcoxon(a, b)
        return float(stat), float(p)
    except ValueError:
        return None  # e.g. all-zero differences


def compute_sweep_sig_records(metric_entries: list) -> list:
    """Paired Wilcoxon test per (metric entry, shared x) of two sweeps, BH-FDR corrected.

    metric_entries are (records_1, records_2, metric, linestyle) with sweep records
    carrying raw per-seed arrays, paired on seed. Cells with fewer than 3 seeds are
    skipped, as are cells with unequal n (with a warning). Correction is over all
    returned tests.
    """
    from src.utils.plotting import _bh_fdr

    recs = []
    for entry_idx, (r1, r2, metric, _ls) in enumerate(metric_entries):
        if not r1 or not r2 or len(r1[0]) < 4:
            continue
        map1 = {r[0]: r[3] for r in r1}
        map2 = {r[0]: r[3] for r in r2}
        for x in sorted(set(map1) & set(map2)):
            a, b = map1[x], map2[x]
            if len(a) < 3 or len(b) < 3:
                continue
            if len(a) != len(b):
                warnings.warn(
                    f"Sweep cell {metric}@{x}: arms have unequal n "
                    f"({len(a)} vs {len(b)}); skipping (mismatched seed sets).")
                continue
            res = paired_wilcoxon(a, b)
            if res is None:
                continue
            stat, p = res
            recs.append(dict(entry_idx=entry_idx, metric=metric, x=x,
                             test="wilcoxon_signed_rank_paired",
                             test_statistic=float(stat), p_raw=float(p),
                             n1=len(a), n2=len(b)))

    if not recs:
        return []
    corrected = _bh_fdr(np.array([r["p_raw"] for r in recs]))
    for r, cp in zip(recs, corrected):
        r["p_corrected"] = float(cp)
        r["significant"] = bool(cp < 0.05)
    return recs


# Paired comparisons (uncorrected; see plotting._bh_fdr for BH-FDR) -------------

def sem(x: np.ndarray) -> float:
    """Standard error of the mean (ddof=1); 0 for a single observation."""
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def wilcoxon_p_floor(n: int) -> float:
    """Smallest two-sided p of the exact Wilcoxon test for n pairs, 2 / 2**n.

    NaN above n = 25, where scipy uses the normal approximation.
    """
    if n < 1 or n > 25:
        return float("nan")
    return float(2.0 ** (1 - n))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                        alpha: float = ALPHA, seed: int = 0) -> "tuple[float, float]":
    """Percentile bootstrap CI for the mean paired difference (a - b).

    Unlike the exact Wilcoxon p, it has no floor at small n.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def significant_keys(records: "list | None") -> "set | None":
    """{(metric, rounded x)} of records flagged significant (no correction here); None for None.

    x is rounded to 9 dp so values rescaled by float products (e.g. x *= T * n_train)
    still match. Accepts records keyed x_value or x.
    """
    if records is None:
        return None
    out = set()
    for r in records:
        if not r.get("significant"):
            continue
        x = r["x_value"] if "x_value" in r else r["x"]
        out.add((r["metric"], round(float(x), 9)))
    return out


def paired_array_comparison(a, b, group_1: str, group_2: str, **meta) -> "dict | None":
    """Paired Wilcoxon test and descriptives for two aligned arrays (a[i] paired with b[i]).

    cohen_d is of the paired difference a - b. Returns None when not testable
    (fewer than 2 pairs, or all differences zero).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[:n], b[:n]
    res = paired_wilcoxon(a, b)
    if res is None:
        return None
    stat, p = res
    diff = a - b
    sd = float(diff.std(ddof=1))
    return dict(
        **meta,
        group_1=group_1, group_2=group_2,
        n_pairs=int(n),
        mean_1=float(a.mean()), sem_1=sem(a),
        mean_2=float(b.mean()), sem_2=sem(b),
        median_diff=float(np.median(diff)),
        test="wilcoxon_signed_rank_paired",
        test_statistic=float(stat),
        p_raw=float(p),
        cohen_d=float(diff.mean() / sd) if sd > 0 else float("nan"),
        n_nonzero_diff=int(np.count_nonzero(diff)),
    )


def eval_metric_by_seed(eval_dir: Path, metric: str) -> pd.Series:
    """Per-seed values of a metric from eval_dir/metrics.csv, indexed by seed."""
    df = pd.read_csv(Path(eval_dir) / "metrics.csv")
    if metric not in df.columns or "seed" not in df.columns:
        return pd.Series(dtype=float)
    return df.set_index("seed")[metric].dropna().sort_index()


def paired_comparison(s1: pd.Series, s2: pd.Series, group_1: str, group_2: str,
                      **meta) -> "dict | None":
    """paired_array_comparison on the shared index of two Series (e.g. seed).

    Pairing on the index keeps a NaN dropped from one arm from mis-pairing the rest.
    """
    idx = s1.index.intersection(s2.index)
    if len(idx) < 2:
        return None
    return paired_array_comparison(
        s1.loc[idx].to_numpy(dtype=float),
        s2.loc[idx].to_numpy(dtype=float),
        group_1, group_2, **meta)
