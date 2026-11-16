"""Per-epoch signal processing for the Huang 2023 pipeline, following the dataset authors' saveBPCs.m.

Epochs are (ch, t) arrays in microvolts. remove_line_noise imports mne lazily so the
rest of the module works without it (the dataloader reuses decimate_signal).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import decimate


# ----------------------------------------------------------------------------
# Common-average referencing
# ----------------------------------------------------------------------------

def common_average_reference(epoch: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Subtract the mean of the ref_mask channels (n_ch,) from every channel of epoch (n_ch, T)."""
    if ref_mask.sum() == 0:
        return epoch.copy()
    car = epoch[ref_mask].mean(axis=0, keepdims=True)
    return epoch - car


def adjusted_car(epoch: np.ndarray, tt: np.ndarray, ref_mask: np.ndarray,
                 pct_thresh: float = 95.0, ratio_thresh: float = 1.5
                 ) -> tuple[np.ndarray, dict]:
    """Per-trial adjusted CAR (port of the dataset authors' ccep_CAR.m).

    Additionally excludes from ref_mask channels with high late variance (0.5-1 s,
    above pct_thresh percentile; residual artefact) and strongly responsive channels
    (early/late variance ratio > ratio_thresh and early variance above the 75th
    percentile; evoked CCEPs). Returns (epoch_car, info) with exclusion counts.
    """
    var_late = epoch[:, (tt > 0.5) & (tt < 1.0)].var(axis=1)
    var_early = epoch[:, (tt > 0.015) & (tt < 0.1)].var(axis=1)

    incl = ref_mask.copy()
    ref_idx = np.where(ref_mask)[0]

    max_late = np.percentile(var_late[ref_idx], pct_thresh)
    n_var = int(((var_late > max_late) & incl).sum())
    incl &= ~(var_late > max_late)

    max_early = np.percentile(var_early[ref_idx], 75)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(var_late > 0, var_early / var_late, np.inf)
    resp = (ratio > ratio_thresh) & (var_early > max_early)
    n_resp = int((resp & incl).sum())
    incl &= ~resp

    epoch_car = common_average_reference(epoch, incl)
    info = {"n_var_excluded": n_var, "n_resp_excluded": n_resp,
            "n_ref": int(incl.sum())}
    return epoch_car, info


# ----------------------------------------------------------------------------
# Line-noise removal (spectrum fitting)
# ----------------------------------------------------------------------------

def remove_line_noise(epoch: np.ndarray, fs: float, line_freq: float = 60.0
                      ) -> np.ndarray:
    """Remove line noise at line_freq and harmonics (mne spectrum_fit, Mewett 2001).

    Matches the dataset authors' removeLN('SpectrumEstimation'). Returns float64.
    """
    from mne.filter import notch_filter

    freqs = np.arange(line_freq, fs / 2.0, line_freq)
    cleaned = notch_filter(
        epoch.astype(np.float64), Fs=fs, freqs=freqs,
        method="spectrum_fit", filter_length="auto", verbose=False,
    )
    return cleaned


# ----------------------------------------------------------------------------
# Baseline correction
# ----------------------------------------------------------------------------

def subtract_baseline(epoch: np.ndarray, tt: np.ndarray,
                      window: tuple[float, float] = (-0.5, -0.05)) -> np.ndarray:
    """Subtract the per-channel median over the pre-stim window (seconds)."""
    sel = (tt >= window[0]) & (tt < window[1])
    base = np.median(epoch[:, sel], axis=1, keepdims=True)
    return epoch - base


# ----------------------------------------------------------------------------
# Stim-artifact interpolation (samples are replaced, never dropped)
# ----------------------------------------------------------------------------

def artifact_window(onset_idx: int, fs: float, pre_ms: float = 2.0,
                    post_ms: float = 15.0) -> tuple[int, int]:
    """Inclusive [a, b] sample indices spanning [-pre_ms, +post_ms] around onset_idx.

    The default window is asymmetric: pre-stim signal is clean, while amplifier ringing
    extends to ~+15 ms and would otherwise enter the continuous latent dynamics.
    """
    a = onset_idx - int(round(pre_ms * 1e-3 * fs))
    b = onset_idx + int(round(post_ms * 1e-3 * fs))
    return a, b


def interpolate_artifact(epoch: np.ndarray, onset_idx: int, fs: float,
                         pre_ms: float = 2.0, post_ms: float = 15.0,
                         method: str = "pchip") -> np.ndarray:
    """Return a copy of epoch (n_ch, T) with the [-pre_ms, +post_ms] window interpolated.

    Samples are replaced rather than dropped so the time series stays continuous.
    Methods: "crowther" (tapered blend of time-reversed neighbouring signal, as in
    saveBPCs.m; can kink at a steep N1), "pchip" (default; shape-preserving cubic,
    no overshoot) and "spline" (cubic spline; can overshoot near steep edges).
    """
    a, b = artifact_window(onset_idx, fs, pre_ms, post_ms)
    out = epoch.copy()
    ix = np.arange(a, b + 1)
    T = epoch.shape[1]

    if method == "crowther":
        npoints = len(ix)
        ix_rev = ix[::-1]
        ix_back = ix_rev - npoints
        ix_forward = ix_rev + npoints
        if ix_back.min() < 0 or ix_forward.max() >= T:
            raise ValueError("Artifact window too close to epoch edge for crowther.")
        ratios = np.linspace(1.0 / (npoints + 1), 1.0 - 1.0 / (npoints + 1), npoints)
        out[:, ix] = (epoch[:, ix_back] * ratios[::-1][None, :]
                      + epoch[:, ix_forward] * ratios[None, :])
        return out

    if method in ("pchip", "spline"):
        pad = max(10, len(ix) // 2)
        left = np.arange(a - pad, a)
        right = np.arange(b + 1, b + 1 + pad)
        if left.min() < 0 or right.max() >= T:
            raise ValueError(f"Artifact window too close to epoch edge for {method}.")
        fit_x = np.concatenate([left, right])
        Interp = PchipInterpolator if method == "pchip" else CubicSpline
        for c in range(epoch.shape[0]):
            out[c, ix] = Interp(fit_x, epoch[c, fit_x])(ix)
        return out

    raise ValueError(f"method must be 'crowther', 'pchip' or 'spline', got {method!r}")


# ----------------------------------------------------------------------------
# Decimation
# ----------------------------------------------------------------------------

def _decimation_factors(q: int) -> list[int]:
    """Split decimation factor q into stages <= 8 (e.g. 32 -> [8, 4]).

    scipy's IIR decimate is numerically poor for q > 13.
    """
    if q < 1:
        raise ValueError(f"decimation factor must be >= 1, got {q}")
    stages: list[int] = []
    while q > 1:
        for f in (8, 7, 6, 5, 4, 3, 2):
            if q % f == 0:
                stages.append(f)
                q //= f
                break
        else:  # prime > 8: one (imperfect) stage
            stages.append(q)
            q = 1
    return stages


def decimate_signal(x: np.ndarray, fs: float, target_fs: float) -> tuple[np.ndarray, int]:
    """Anti-aliased downsample of x (..., T) along the last axis from fs to target_fs.

    target_fs must divide fs. Returns (y, q) with q = fs / target_fs; decimated
    sample i corresponds to original index i * q.
    """
    if fs % target_fs != 0:
        raise ValueError(
            f"target_fs={target_fs} must evenly divide fs={fs} "
            f"(got remainder {fs % target_fs})."
        )
    q = int(round(fs / target_fs))
    if q == 1:
        return np.asarray(x, dtype=float).copy(), 1
    y = np.asarray(x, dtype=float)
    for f in _decimation_factors(q):
        y = decimate(y, f, ftype="iir", zero_phase=True, axis=-1)
    return y, q


# ----------------------------------------------------------------------------
# Stim-artefact matched-filter template
# ----------------------------------------------------------------------------

def artefact_template(X: np.ndarray, tt: np.ndarray, template_trials: np.ndarray,
                      keep_idx: np.ndarray, window=(-0.01, 0.04)
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean |X| artefact template over keep_idx channels of template_trials.

    Absolute values make sign-varying deflections add constructively. X is
    (trials, ch, t), tt in seconds with the pulse at 0, window in seconds.
    Returns (template, template_sd, sel): (n_w,), (n_w,) and the (t,) window mask.
    """
    sel = (tt >= window[0]) & (tt <= window[1])
    absX = np.abs(X[np.ix_(template_trials, keep_idx, np.where(sel)[0])])  # (n_tr, n_keep, n_w)
    stacked = absX.reshape(-1, absX.shape[-1])
    return stacked.mean(0), stacked.std(0), sel


def template_correlation(X: np.ndarray, sel: np.ndarray, template: np.ndarray
                         ) -> np.ndarray:
    """(trials, ch) Pearson correlation of |X[:, :, sel]| to template (n_w,)."""
    V = np.abs(X[:, :, sel])                                # (trials, ch, n_w)
    Vc = V - V.mean(axis=2, keepdims=True)
    tc = template - template.mean()
    num = Vc @ tc                                           # (trials, ch)
    den = np.sqrt((Vc ** 2).sum(axis=2)) * np.sqrt((tc ** 2).sum()) + 1e-12
    return num / den
