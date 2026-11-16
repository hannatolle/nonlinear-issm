"""Evaluation metrics: masked prediction scores, latent alignment and Jacobian comparison."""

import numpy as np
import torch
from typing import Union, Optional, Any, Tuple, Dict
from scipy.optimize import linear_sum_assignment
import warnings


# Prediction performance -------------------------------------------------------

def _build_keep_mask(shape: tuple,
                     u: Optional[np.ndarray],
                     mask: Optional[np.ndarray]) -> np.ndarray:
    """(B, T, N) keep mask: timepoints with all-zero u, minus cells where mask is True."""
    keep = np.ones(shape, dtype=bool)
    if u is not None:
        keep &= (~np.any(u != 0, axis=2))[:, :, None]  # (B, T, 1) broadcast over N
    if mask is not None:
        keep &= ~mask
    return keep

def _as_trial_flags(flags: Optional[np.ndarray], B: int) -> Optional[np.ndarray]:
    """Validate a (B,) per-trial boolean flag array, or pass through None."""
    if flags is None:
        return None
    if isinstance(flags, torch.Tensor):
        flags = flags.detach().cpu().numpy()
    flags = np.asarray(flags, dtype=bool)
    if flags.shape != (B,):
        raise ValueError(f"Per-trial flags must have shape ({B},); got {flags.shape}")
    return flags


def invalid_trial_mask(**outputs: Union[np.ndarray, torch.Tensor]
                       ) -> Tuple[np.ndarray, Dict[str, int]]:
    """(B,) mask of trials where any output (name -> (B, ...) array) is non-finite, plus per-output counts.

    Checking the decoded observations as well as the latents catches rollouts that
    are finite in latent space but overflow the emission, without a magnitude threshold.
    Flagged trials score 0 on correlations and are dropped from MAE/ELBO.
    """
    invalid = None
    counts: Dict[str, int] = {}
    for name, arr in outputs.items():
        if isinstance(arr, torch.Tensor):
            bad = ~torch.isfinite(arr).reshape(arr.shape[0], -1).all(dim=1)
            bad = bad.detach().cpu().numpy()
        else:
            bad = ~np.isfinite(np.asarray(arr).reshape(np.shape(arr)[0], -1)).all(axis=1)
        counts[name] = int(bad.sum())
        invalid = bad if invalid is None else (invalid | bad)
    if invalid is None:
        raise ValueError("invalid_trial_mask needs at least one output array.")
    counts["total"] = int(invalid.sum())
    return invalid, counts


def assert_true_trials_scorable(true: Union[np.ndarray, torch.Tensor],
                                u: Optional[Union[np.ndarray, torch.Tensor]] = None,
                                mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
                                name: str = "trials") -> None:
    """Raise unless every trial's kept ground-truth cells are finite and non-constant.

    Precondition for invalid_trials in compute_corr_per_trial: it guarantees that an
    undefined correlation comes from the prediction.
    """
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    keep = _build_keep_mask(true.shape, u, mask)
    bad_finite, bad_const, bad_empty = [], [], []
    for b in range(true.shape[0]):
        vals = true[b][keep[b]]
        if vals.size < 2:
            bad_empty.append(b)
        elif not np.isfinite(vals).all():
            bad_finite.append(b)
        elif np.std(vals) == 0:
            bad_const.append(b)

    problems = []
    if bad_finite:
        problems.append(f"non-finite: {bad_finite}")
    if bad_const:
        problems.append(f"constant: {bad_const}")
    if bad_empty:
        problems.append(f"fewer than 2 kept cells: {bad_empty}")
    if problems:
        raise ValueError(
            f"Ground-truth {name} are not scorable ({'; '.join(problems)}). Every "
            "correlation metric assumes valid ground truth, so these trials would "
            "make an undefined score ambiguous between a data defect and a model "
            "failure. Regenerate the test set or fix the data.")


def compute_corr_per_trial(
    true: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    u: Optional[Union[np.ndarray, torch.Tensor]] = None,
    absolute: bool = False,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    invalid_trials: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """Per-trial correlation (B,) of true vs pred (B, T, N), pooled over kept (t, channel) cells.

    Args:
        u: optional (B, T, M); only timesteps with all-zero u are used.
        mask: optional (B, T, N) bool, True = ignore that cell.
        invalid_trials: optional (B,) bool of invalid model outputs (e.g. diverged
            rollouts). If given, flagged and otherwise undefined trials score 0, the
            worst attainable score; callers must ensure the ground truth is scorable
            (assert_true_trials_scorable). If None, undefined trials are NaN.
    """
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    B = true.shape[0]
    invalid_trials = _as_trial_flags(invalid_trials, B)
    keep = _build_keep_mask(true.shape, u, mask)  # (B, T, N)

    corrs = np.full(B, np.nan)
    for b in range(B):
        if invalid_trials is not None and invalid_trials[b]:
            continue
        true_vals = true[b][keep[b]]
        pred_vals = pred[b][keep[b]]
        if true_vals.size >= 2 and np.std(true_vals) > 0 and np.std(pred_vals) > 0:
            corrs[b] = np.corrcoef(true_vals, pred_vals)[0, 1]

    if absolute:
        corrs = np.abs(corrs)
    if invalid_trials is not None:
        corrs[~np.isfinite(corrs)] = 0.0
    return corrs


def compute_mae_per_trial(
    true: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    u: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    invalid_trials: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """Per-trial MAE (B,), pooled over kept cells; arguments as in compute_corr_per_trial.

    Unlike correlations, invalid trials are left NaN (dropped from means): an
    unbounded error has no worst-case value to fill in. NaN also where no cells remain.
    """
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    B = true.shape[0]
    invalid_trials = _as_trial_flags(invalid_trials, B)
    keep = _build_keep_mask(true.shape, u, mask)  # (B, T, N)

    maes = np.full(B, np.nan)
    for b in range(B):
        if invalid_trials is not None and invalid_trials[b]:
            continue
        true_vals = true[b][keep[b]]
        pred_vals = pred[b][keep[b]]
        if true_vals.size > 0:
            maes[b] = float(np.mean(np.abs(true_vals - pred_vals)))
    return maes


def compute_mean_corr(true: Union[np.ndarray, torch.Tensor],
                      pred: Union[np.ndarray, torch.Tensor],
                      u: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      absolute: bool = False,
                      per_trial: bool = True,
                      mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      invalid_trials: Optional[Union[np.ndarray, torch.Tensor]] = None
                      ) -> Tuple[float, float]:
    """(mean, std) of correlations, per trial (pooled over (T, N)) or per (trial, channel) over T.

    Other arguments as in compute_corr_per_trial; invalid_trials requires per_trial=True.
    """
    if per_trial:
        corrs = compute_corr_per_trial(true, pred, u=u, absolute=absolute, mask=mask,
                                       invalid_trials=invalid_trials)
        return float(np.nanmean(corrs)), float(np.nanstd(corrs))

    if invalid_trials is not None:
        raise ValueError("invalid_trials is only supported with per_trial=True.")

    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    B, T, N = true.shape
    keep = _build_keep_mask(true.shape, u, mask)  # (B, T, N)

    corrs = np.full((B, N), np.nan)
    for b in range(B):
        for n in range(N):
            kk = keep[b, :, n]
            true_vals = true[b, kk, n]
            pred_vals = pred[b, kk, n]
            if true_vals.size >= 2 and np.std(true_vals) > 0 and np.std(pred_vals) > 0:
                corrs[b, n] = np.corrcoef(true_vals, pred_vals)[0, 1]

    if absolute:
        corrs = np.abs(corrs)
    return float(np.nanmean(corrs)), float(np.nanstd(corrs))

def compute_mean_mae(true: Union[np.ndarray, torch.Tensor],
                     pred: Union[np.ndarray, torch.Tensor],
                     u: Optional[Union[np.ndarray, torch.Tensor]] = None,
                     per_trial: bool = True,
                     mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
                     invalid_trials: Optional[Union[np.ndarray, torch.Tensor]] = None
                     ) -> Tuple[float, float]:
    """(mean, std) of MAE, per trial (pooled over (T, N)) or per (trial, channel) over T.

    Other arguments as in compute_mae_per_trial; invalid_trials requires per_trial=True.
    """
    if per_trial:
        maes = compute_mae_per_trial(true, pred, u=u, mask=mask,
                                     invalid_trials=invalid_trials)
        return float(np.nanmean(maes)), float(np.nanstd(maes))

    if invalid_trials is not None:
        raise ValueError("invalid_trials is only supported with per_trial=True.")

    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    B, T, N = true.shape
    keep = _build_keep_mask(true.shape, u, mask)  # (B, T, N)
    maes = np.full((B, N), np.nan)

    for b in range(B):
        for n in range(N):
            kk = keep[b, :, n]
            true_filtered = true[b, kk, n]
            pred_filtered = pred[b, kk, n]
            if true_filtered.size > 0:
                maes[b, n] = np.mean(np.abs(true_filtered - pred_filtered))

    return float(np.nanmean(maes)), float(np.nanstd(maes))


# Basic metrics -----------------------------------------------------------------

def cosine_similarity(a: np.ndarray,
                      b: np.ndarray,
                      eps: float = 1e-12) -> Union[float, np.ndarray]:
    """Row-wise cosine similarity of (D,) or (N, D) arrays; NaN where a norm product is < eps."""
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    scalar_out = False
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
        scalar_out = True
    elif a.ndim != 2:
        raise ValueError(f"Expected 1D or 2D arrays, got {a.ndim}D with shape {a.shape}")

    num = np.sum(a * b, axis=1)                                  # (N,)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)  # (N,)

    out = np.full(a.shape[0], np.nan, dtype=float)
    valid = den >= eps
    out[valid] = num[valid] / (den[valid] + eps)

    if scalar_out:
        return float(out[0])
    return out


# Approximate equivalence -------------------------------------------------------

def hungarian_assignment(cost: np.ndarray) -> np.ndarray:
    """Min-cost assignment; returns perm with row i -> column perm[i]."""
    r, c = linear_sum_assignment(cost)
    perm = np.zeros(cost.shape[0], dtype=int)
    perm[r] = c
    return perm

def align_by_B(B_ref: np.ndarray, B_tgt: np.ndarray, use_abs: bool = True) -> np.ndarray:
    """Match latent rows of (D, M) B matrices by cosine similarity; ref row i -> tgt row perm[i]."""
    if B_ref.shape != B_tgt.shape:
        raise ValueError(f"B shape mismatch: ref {B_ref.shape} vs tgt {B_tgt.shape}")
    D = B_ref.shape[0]
    cost = np.zeros((D, D), dtype=float)
    for i in range(D):
        for j in range(D):
            sim = cosine_similarity(B_ref[i], B_tgt[j])
            if np.isnan(sim):
                cost[i, j] = 1.0
            else:
                sim = abs(sim) if use_abs else sim
                cost[i, j] = 1.0 - sim
    return hungarian_assignment(cost)

# Allowed values of the latent_alignment eval setting.
LATENT_ALIGNMENT_MODES = ("B", "corr")

def resolve_latent_alignment(mode: Any) -> str:
    """Validate a latent_alignment setting value and return it."""
    if mode not in LATENT_ALIGNMENT_MODES:
        raise ValueError(
            f"settings.latent_alignment must be one of {LATENT_ALIGNMENT_MODES}; "
            f"got {mode!r}.")
    return mode

def corr_perm(x_ref: Union[np.ndarray, torch.Tensor],
              x_tgt: Union[np.ndarray, torch.Tensor],
              u: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
    """Match latents (B, T, D) by absolute correlation; ref i -> tgt perm[i], as in align_by_B.

    For models where B is unusable (n_stim = 0) or not comparable. Constant series
    get the worst cost 1.0. u (B, T, M), if given, restricts to all-zero-u timesteps;
    pass the same u as the metrics and affine fit use.
    """
    if isinstance(x_ref, torch.Tensor):
        x_ref = x_ref.detach().cpu().numpy()
    if isinstance(x_tgt, torch.Tensor):
        x_tgt = x_tgt.detach().cpu().numpy()
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()

    x_ref = np.asarray(x_ref, dtype=float)
    x_tgt = np.asarray(x_tgt, dtype=float)
    if x_ref.shape[-1] != x_tgt.shape[-1]:
        raise ValueError(
            f"Latent dim mismatch: ref {x_ref.shape[-1]} vs tgt {x_tgt.shape[-1]}")
    D = x_ref.shape[-1]

    if u is None:
        ref, tgt = x_ref.reshape(-1, D), x_tgt.reshape(-1, D)
    else:
        # The u-derived mask is identical across dimensions, so collapse it to (B, T).
        keep = _build_keep_mask(x_ref.shape, u, None)[:, :, 0]
        ref, tgt = x_ref[keep], x_tgt[keep]

    finite = np.isfinite(ref).all(axis=1) & np.isfinite(tgt).all(axis=1)
    ref, tgt = ref[finite], tgt[finite]
    if ref.shape[0] < 2:
        raise ValueError(
            f"corr_perm needs >=2 usable timepoints to match latents; got {ref.shape[0]} "
            "after dropping interventional and non-finite ones. Returning a permutation "
            "here would be arbitrary.")

    cost = np.ones((D, D), dtype=float)
    for i in range(D):
        for j in range(D):
            if np.std(ref[:, i]) == 0 or np.std(tgt[:, j]) == 0:
                continue
            r = np.corrcoef(ref[:, i], tgt[:, j])[0, 1]
            if not np.isnan(r):
                cost[i, j] = 1.0 - abs(r)
    return hungarian_assignment(cost)

def latent_perm(mode: str,
                *,
                D: int,
                n_stim: int,
                train_B: bool,
                B_ref: Optional[np.ndarray] = None,
                B_tgt: Optional[np.ndarray] = None,
                x_ref: Optional[Union[np.ndarray, torch.Tensor]] = None,
                x_tgt: Optional[Union[np.ndarray, torch.Tensor]] = None,
                u: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
    """Permutation (D,) putting tgt latents into ref order, by mode "B" or "corr".

    The mode is a config choice; an unsupported mode raises instead of falling back,
    so all compared models use the same estimator. B_ref/B_tgt (D, n_stim) are needed
    for "B" with train_B, x_ref/x_tgt (B, T, D) for "corr".
    """
    mode = resolve_latent_alignment(mode)

    if mode == "corr":
        if x_ref is None or x_tgt is None:
            raise ValueError("latent_alignment='corr' requires x_ref and x_tgt.")
        return corr_perm(x_ref, x_tgt, u=u)

    if n_stim == 0:
        raise ValueError(
            "latent_alignment='B' is undefined for a model with no stimulation "
            "channels (n_stim = M - n_task_channels = 0): B is (D, 0), every cosine "
            "is NaN, the cost matrix is constant, and the assignment would silently "
            "return the identity. Set latent_alignment='corr'.")

    # Checked after n_stim: a fixed B has the system's latent order by construction.
    if not train_B:
        return np.arange(D)

    if B_ref is None or B_tgt is None:
        raise ValueError("latent_alignment='B' requires B_ref and B_tgt.")
    return align_by_B(B_ref, B_tgt, use_abs=True)

def apply_perm_latents(x: torch.Tensor, perm_ref_to_tgt: np.ndarray) -> torch.Tensor:
    """Reorder learned latents x (..., D) into ref order: out[..., i] = x[..., perm[i]]."""
    idx = torch.tensor(perm_ref_to_tgt, dtype=torch.long, device=x.device)
    return x.index_select(dim=-1, index=idx)

def fit_affine_per_dim(x_ref: np.ndarray,
                       x_tgt: np.ndarray,
                       eps: float = 1e-12,
                       return_xt_hat: bool = False,
                       ) -> Union[Tuple[np.ndarray, np.ndarray, Dict[str, Any]], 
                                  Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]]:
    """Least-squares fit x_tgt[:, i] ~ d[i] * x_ref[:, i] + c[i] on (n, D) arrays.

    Returns d, c, fit metrics (r2, rmse, nrmse_ref) and, if return_xt_hat, the fitted x_tgt.
    """
    if x_ref.shape != x_tgt.shape:
        raise ValueError(f"Affine fit shape mismatch: ref {x_ref.shape}, tgt {x_tgt.shape}")
    if x_ref.ndim != 2:
        raise ValueError(f"Expected x_ref and x_tgt to be 2D arrays (N,D); got {x_ref.ndim}D")

    N, D = x_ref.shape
    d = np.zeros(D, dtype=float)
    c = np.zeros(D, dtype=float)
    
    r2 = np.full(D, np.nan, dtype=float)
    rmse = np.full(D, np.nan, dtype=float)
    nrmse_ref = np.full(D, np.nan, dtype=float)

    x_tgt_hat = np.empty((N, D), dtype=float) if return_xt_hat else None

    for i in range(D):
        xr = x_ref[:, i]
        xt = x_tgt[:, i]

        mxr = xr.mean()
        mxt = xt.mean()

        vxr = ((xr - mxr) ** 2).mean()
        if vxr < eps:
            d[i] = 1.0
            c[i] = mxt - d[i] * mxr
        else:
            cov = ((xr - mxr) * (xt - mxt)).mean()
            d[i] = cov / vxr
            c[i] = mxt - d[i] * mxr

        xt_hat_i = d[i] * xr + c[i]
        resid = xt - xt_hat_i

        if return_xt_hat:
            x_tgt_hat[:, i] = xt_hat_i

        rmse_i = float(np.sqrt(np.mean(resid ** 2)))
        rmse[i] = rmse_i

        var_tgt = float(np.mean((xt - mxt) ** 2))
        if var_tgt < eps:
            r2[i] = np.nan
        else:
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((xt - mxt) ** 2))
            r2[i] = 1.0 - (ss_res / (ss_tot + eps))

        std_ref = float(np.std(xr))
        nrmse_ref[i] = rmse_i / (std_ref + eps)
    
    small = np.abs(d) < eps
    if np.any(small):
        warnings.warn("Near-zero scaling detected; affine alignment likely failed.")
        d[small] = np.where(d[small] == 0, eps, np.sign(d[small]) * eps)


    metrics: Dict[str, Any] = {
        "r2": r2,
        "rmse": rmse,
        "nrmse_ref": nrmse_ref}

    if return_xt_hat:
        return d, c, metrics, x_tgt_hat
    return d, c, metrics

def canonicalize_to_ref(x_tgt: np.ndarray,
                        d: np.ndarray,
                        c: np.ndarray,
                        eps: float = 1e-12) -> np.ndarray:
    """Map x_tgt (N, D) into ref coordinates, (x_tgt - c) / d; |d| < eps is treated as 1."""
    d_safe = d.copy()
    d_safe[np.abs(d_safe) < eps] = 1.0
    return (x_tgt - c[None, :]) / d_safe[None, :]


# Latent Jacobian similarity ----------------------------------------------------

def compute_jacobians(system: torch.nn.Module,
                      x_points: Union[np.ndarray, torch.Tensor], 
                      device: Optional[torch.device] = None) -> np.ndarray:
    """Jacobians (n_points, D, D) of system.flow at x_points (n_points, D)."""
    if not callable(getattr(system, "flow", None)):
        raise ValueError(f"system must have a flow method. Got {type(system)}")
    if not hasattr(system, "D"):
        raise ValueError("system must have a property 'D' specifying latent dimension")
    D_expected = getattr(system, "D")
    if x_points.shape[1] != D_expected:
        raise ValueError(f"x_points' shape[1] ({x_points.shape[1]}) does not match system.D ({D_expected})")

    if device is None:
        device = torch.device("cpu")

    x = torch.as_tensor(x_points, dtype=torch.float32, device=device)
    x = x.requires_grad_(True)

    if isinstance(system, torch.nn.Module):
        system.eval()
        system.to(device=device)

    def flow_single(x1: torch.Tensor) -> torch.Tensor:
        return system.flow(x1[None, :])[0]

    from torch.func import jacrev, vmap
    J = vmap(jacrev(flow_single))(x)  # (N, D, D)

    return J.detach().cpu().numpy()


def transform_jacobians_to_ref(J_tgt: np.ndarray,
                               perm_ref_to_tgt: Optional[np.ndarray] = None,
                               d: Optional[np.ndarray] = None,
                               eps: float = 1e-12) -> np.ndarray:
    """Express target Jacobians (N, D, D) in reference coordinates.

    Permutation (ref_i -> tgt perm[i]): P^T J P with P[i, perm[i]] = 1.
    Scaling (x_tgt = diag(d) x_ref + c): diag(d)^-1 J diag(d).
    """
    if J_tgt.ndim != 3 or J_tgt.shape[1] != J_tgt.shape[2]:
        raise ValueError(f"Expected J_tgt shape (N,D,D). Got {J_tgt.shape}")
    D_dim = J_tgt.shape[1]

    if perm_ref_to_tgt is not None:
        perm = np.asarray(perm_ref_to_tgt, dtype=int)
        if perm.shape != (D_dim,):
            raise ValueError(f"perm shape {perm.shape} != ({D_dim},)")
        # (P^T J P)[i, k] = J[perm[i], perm[k]]
        J_tgt = J_tgt[:, perm, :][:, :, perm]

    if d is not None:
        d = np.asarray(d, dtype=float)
        if d.shape != (D_dim,):
            raise ValueError(f"d shape {d.shape} != ({D_dim},)")
        d_safe = np.where(np.abs(d) < eps, np.sign(d) * eps + (d == 0) * eps, d)
        d_inverse = 1.0 / d_safe
        # Rows scaled by 1/d, columns by d.
        J_tgt = (d_inverse[None, :, None] * J_tgt) * (d_safe[None, None, :])

    return J_tgt
