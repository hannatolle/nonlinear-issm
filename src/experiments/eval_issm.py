"""
Evaluate a trained iSSM model.

Naming conventions:
- x*_p_aligned: latent indices are aligned with true system
- x*_pdc_aligned: permutation, scale, shift aligned with true
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models import FlexibleLDS
from src.simulations.interventions import RandomStimulator
from src.utils.io import (
    add_root, get_root, load_json, save_json, nested_dicts_equal,
    build_component_from_run, load_component_from_run,
)
from src.utils.stats import fix_random_seed
from src.utils.metrics import (
    apply_perm_latents, assert_true_trials_scorable, canonicalize_to_ref, compute_jacobians,
    compute_mean_corr, cosine_similarity, fit_affine_per_dim, invalid_trial_mask, latent_perm,
    resolve_latent_alignment, transform_jacobians_to_ref,
)
from .base import Experiment


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: All per-seed metric column names (excluding "seed").
METRIC_COLUMNS: List[str] = [
    # Observation prediction (non-interventional timepoints only) — rollout-based
    "y_corr_rollout", "y_corr_rollout_std",
    # Observation reconstruction
    "y_corr", "y_corr_std",
    # Latent faithfulness
    "x_pdc_aligned_corr", "x_pdc_aligned_corr_std",
    # Jacobian similarity
    "jac_cosine", "jac_cosine_std",
    # Identifiability proxy (goodness of affine fit)
    "r2_mean",
    # B matrix faithfulness (vs true B; only meaningful when train_B=True)
    "b_cosine",
    "b_zero_mae",
]

#: Minimum non-interventional timepoints for affine fit before issuing a warning.
_MIN_AFFINE_FIT_POINTS = 50


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _nan_metrics_row(seed: int) -> dict:
    """Return a metrics row with seed set and all numeric fields as np.nan."""
    return {"seed": seed, **{k: np.nan for k in METRIC_COLUMNS}}


def _to_np(a):
    """Convert a tensor or array-like to a numpy array."""
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def _n_stim_from_config(cfg: Dict) -> Optional[int]:
    """n_stim = M - n_task_channels declared by a run config, or None.

    Read from the config so alignment can be validated before weights load
    (and because dynamics.B is None when a b_net supplies B).
    """
    params = cfg.get("components", {}).get("dynamics", {}).get("params", {})
    M = params.get("M", cfg.get("settings", {}).get("M"))
    if M is None:
        return None
    return int(M) - int(params.get("n_task_channels", 0) or 0)


# ---------------------------------------------------------------------------
# EvalISSM experiment
# ---------------------------------------------------------------------------

class EvalISSM(Experiment):
    """Evaluate a group of iSSMs trained with different seeds on fresh data
    simulated from the true system (group_dir/seed_*/).
    """

    @classmethod
    def preprocess_config(cls, config: dict) -> dict:
        import copy
        config = copy.deepcopy(config)
        settings = config.get("settings", {})

        # Default output dir
        if settings.get("use_default_output_dir", True):
            group_dir = settings.get("group_dir", "")
            if group_dir:
                default_output = os.path.join(group_dir, "eval")
                provided = config.get("output_dir", "./outputs/default")
                if provided not in ("./outputs", default_output):
                    print(f"Warning: use_default_output_dir=True; ignoring output_dir={provided!r}, using {default_output!r}")
                config["output_dir"] = default_output

        # Derive the eval stimulator from the training stimulator, unless the
        # user supplied one.
        components = config.setdefault("components", {})
        if "stimulator" not in components:
            group_dir = settings.get("group_dir", "")
            if group_dir:
                group_dir_abs = add_root(group_dir)
                seed_dirs = sorted([
                    d for d in os.listdir(group_dir_abs) if d.startswith("seed_")
                ])
                if seed_dirs:
                    first_run = os.path.join(group_dir_abs, seed_dirs[0])
                    components["stimulator"] = cls._derive_eval_stimulator_spec(first_run)

        return config

    @classmethod
    def _derive_eval_stimulator_spec(cls, run_dir: str) -> dict:
        """Build the default eval stimulator spec from the training stimulator.

        The eval stimulator is always a RandomStimulator, inheriting stim_d and
        sigma from training and forcing stim_control_ratio=1.0. Amplitude
        (pre-scaling; run() applies stim_amp_scaling) is taken from training:
        a RandomStimulator's signed (min, max) is used as is; any other
        stimulator is symmetrized to (-A, +A) with A = max(|min|, |max|). The
        stim_rate is a placeholder — _generate_stim_inputs overwrites it per
        trial from the stim_rate grid.
        """
        train_stim = build_component_from_run(run_dir, "stimulator")
        lo, hi = float(train_stim.min_amplitude), float(train_stim.max_amplitude)
        if not isinstance(train_stim, RandomStimulator):
            if not (lo < 0 < hi):
                warnings.warn(
                    f"Training {type(train_stim).__name__} has a constant-sign "
                    f"amplitude range [{lo}, {hi}]; the eval RandomStimulator "
                    f"symmetrizes it to (-A, +A), so the amplitude sign is NOT "
                    f"held constant during evaluation.")
            A = max(abs(lo), abs(hi))
            lo, hi = -A, A
        return {
            "kind": "RandomStimulator",
            "params": {
                "M": int(train_stim.M),
                "stim_rate": 0.5,          # placeholder; overwritten by rate grid
                "stim_d": float(train_stim.stim_d),
                "sigma": float(train_stim.sigma),
                "min_amplitude": lo,
                "max_amplitude": hi,
                "stim_control_ratio": 1.0,
            },
        }

    def check_config(self):
        """
        Checks:
        - group_dir exists and contains seed_* subdirectories
        - Each seed dir has required weight files and config.json
        - settings and components match across all seeds
        """
        s = self.fetch_settings()
        if "group_dir" not in s:
            raise ValueError("EvalISSM requires settings.group_dir.")
        group_dir = add_root(s.group_dir)
        if not os.path.isdir(group_dir):
            raise ValueError(f"EvalISSM settings.group_dir is not a directory: {group_dir}")
        seed_dirs = sorted([d for d in os.listdir(group_dir) if d.startswith("seed_")])
        if not seed_dirs:
            raise ValueError(f"EvalISSM group_dir has no seed_* subdirectories: {group_dir}")

        required_weights = ["dynamics", "emission", "likelihood", "recognition"]
        configs = []
        for seed_dir in seed_dirs:
            run_dir = os.path.join(group_dir, seed_dir)
            for name in required_weights:
                pth = os.path.join(run_dir, f"{name}.pth")
                if not os.path.isfile(pth):
                    raise ValueError(f"EvalISSM missing weight file: {pth}")
            cfg_path = os.path.join(run_dir, "config.json")
            if not os.path.isfile(cfg_path):
                raise ValueError(f"EvalISSM missing config file: {cfg_path}")
            cfg = load_json(cfg_path)
            configs.append(cfg)
            if self._latent_alignment() == "B" and _n_stim_from_config(cfg) == 0:
                raise ValueError(
                    f"settings.latent_alignment='B' but {seed_dir} has no stimulation "
                    f"channels (n_stim = M - n_task_channels = 0), so its B is (D, 0) "
                    f"and B-based alignment is undefined. Set latent_alignment='corr'.")

        ref = configs[0]
        for i, cfg in enumerate(configs[1:], start=1):
            if "settings" not in ref or "settings" not in cfg:
                raise ValueError(
                    f"EvalISSM config must have 'settings'; "
                    f"mismatch in {seed_dirs[0]} or {seed_dirs[i]}."
                )
            if not nested_dicts_equal(ref["settings"], cfg["settings"]):
                raise ValueError(
                    f"EvalISSM config['settings'] must match across seed dirs; "
                    f"mismatch between {seed_dirs[0]} and {seed_dirs[i]}."
                )
            ref_comp = ref.get("components", {})
            cfg_comp = cfg.get("components", {})
            if not nested_dicts_equal(ref_comp, cfg_comp):
                raise ValueError(
                    f"EvalISSM config['components'] must match across seed dirs; "
                    f"mismatch between {seed_dirs[0]} and {seed_dirs[i]}."
                )

    # -----------------------------------------------------------------------
    # Data generation
    # -----------------------------------------------------------------------

    def _stim_rate_grid(self) -> list:
        """Return the grid of stimulation rates spanned by the test set."""
        s = self.fetch_settings()
        return [round(float(r), 6)
                for r in np.linspace(s.stim_rate_min, s.stim_rate_max, s.stim_rate_n)]

    @torch.no_grad()
    def _generate_stim_inputs(self, stimulator, n_samples: int, T_total: float, dt: float):
        """Generate stimulation inputs, distributing trials across the stim-rate grid.

        Trials are split approximately equally over the rate grid.
        """
        if not hasattr(stimulator, "stim_rate"):
            return stimulator.generate(n_samples, T_total, dt)

        rates = self._stim_rate_grid()
        counts = dict(zip(rates, [len(c) for c in np.array_split(np.arange(n_samples), len(rates))]))

        orig_rate = stimulator.stim_rate
        chunks = []
        for rate in rates:
            if counts[rate] == 0:
                continue
            stimulator.stim_rate = float(rate)
            chunks.append(stimulator.generate(counts[rate], T_total, dt))
        stimulator.stim_rate = orig_rate
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _generate_data(self, system, stimulator, dt: float, T_burn: float):
        """Simulate T_burn + T_encode + T seconds and discard the burn-in.

        T_burn comes from the training config so eval data is generated like the
        training data. Returns (u, x, y), each (n, steps_encode + steps_eval, *).
        """
        s = self.fetch_settings()
        T_total = T_burn + s.T_encode + s.T
        u = self._generate_stim_inputs(stimulator, s.n_samples, T_total, dt)
        x0 = system.rand_state(s.n_samples)

        _, x_ = system.run(T=T_total, dt=dt, x0=x0, u=u)
        y_ = system.obs(x_)

        # Discard burn, permute to (n_samples, steps_encode + steps_eval, ...)
        steps_burn = int(T_burn / dt)
        y = y_[steps_burn:].permute(1, 0, 2)
        x = x_[steps_burn:].permute(1, 0, 2)
        u_out = u[:, steps_burn:, :]
        return u_out, x, y

    @staticmethod
    def _scale_stim_amplitude(stimulator, scale: float) -> None:
        """Scale the stimulator's amplitude range in place (probes interventions
        stronger than those seen in training). RegularStimulator's min/max are
        read-only properties, so its amplitude arrays are scaled instead.
        """
        if scale == 1.0:
            return
        if "min_amplitude" in vars(stimulator):          # RandomStimulator
            stimulator.min_amplitude *= scale
            stimulator.max_amplitude *= scale
        else:                                            # RegularStimulator
            if getattr(stimulator, "_amplitude_values", None) is not None:
                stimulator._amplitude_values = stimulator._amplitude_values * scale
            if getattr(stimulator, "_amplitude_fixed", None) is not None:
                stimulator._amplitude_fixed = stimulator._amplitude_fixed * scale

    @torch.no_grad()
    def _sample_latent_points(self, x_eval: torch.Tensor) -> torch.Tensor:
        """Sample random latent points from x_eval for Jacobian evaluation.

        x_eval: any shape (..., D) — flattened along all but last dim.
        Returns (n_points, D) tensor capped at max_jacobian_points.
        """
        s = self.fetch_settings()
        x_flat = x_eval.reshape(-1, x_eval.shape[-1])
        n_points = min(s.max_jacobian_points, x_flat.shape[0])
        idx = np.random.choice(x_flat.shape[0], size=n_points, replace=False)
        return x_flat[idx]

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_models(self, 
                     group_dir: str, 
                     seed_dirs: List[str], device: torch.device
                     ) -> Tuple[Dict, Dict]:
        """Load FlexibleLDS joint models and recognition networks for all seeds."""
        joint_models = {}
        recognition_models = {}
        for seed_dir in seed_dirs:
            seed = int(seed_dir.split("_")[-1])
            run_dir = os.path.join(group_dir, seed_dir)
            joint_components = {
                name: load_component_from_run(run_dir, name, device)
                for name in ["dynamics", "emission", "likelihood"]}
            joint_models[seed] = FlexibleLDS(**joint_components).eval()
            recognition_models[seed] = load_component_from_run(
                run_dir, "recognition", device).eval()
            recognition_models[seed].set_B_from_dynamics(joint_models[seed].dynamics)
        return joint_models, recognition_models

    # -----------------------------------------------------------------------
    # Per-seed evaluation
    # -----------------------------------------------------------------------

    @staticmethod
    def _model_mean_eval(model, u: torch.Tensor, x0: torch.Tensor):
        """Run model.mean and drop the initial (x0) output step.

        Returns (x_pred, y_pred) each of shape (B, T, *).
        """
        x_full, y_full = model.mean(u=u, x0=x0)
        return x_full[:, 1:, :], y_full[:, 1:, :]

    @torch.no_grad()
    def _reconstruct(self, model, recognition, y_true: torch.Tensor, u: torch.Tensor):
        """Encoder-decoder reconstruction of the full trial (no dynamics integrated),
        built as in EvalRealISSM._collect_predictions. Returns (y_rec, x_post).
        """
        recognition.set_B_from_dynamics(model.dynamics)
        x_post = recognition.encode_mean(y_true, u)
        return model.likelihood.mean(model.emission(x_post)), x_post

    # -----------------------------------------------------------------------
    # _fit_affine  /  _evaluate_condition  /  _evaluate_seed
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _fit_affine(self,
                    model,
                    recognition,
                    y_true: torch.Tensor,
                    u: torch.Tensor,
                    x_true: torch.Tensor,
                    perm: np.ndarray,
                    D: int,
                    seed: int):
        """Fit per-dim affine maps from the permutation-aligned full-trial posterior
        to the true latents.

        Only non-interventional timesteps are used (the recognition net sets the
        posterior to Bu at stim steps). The map is a coordinate calibration, so
        using the full trial does not leak into the rollout metrics, which use
        only x0. Returns (d, c, fit_stats); identity if too few valid points.
        """
        recognition.set_B_from_dynamics(model.dynamics)
        x_post = recognition.encode_mean(y_true, u)            # (B, T, D), full trial
        x_post_p = apply_perm_latents(x_post, perm)

        # Mask over the full width of u (a property of the data, not the model), so
        # every model calibrates on the same timesteps as compute_mean_corr scores.
        zero_mask = ~np.any(_to_np(u) != 0, axis=2)   # (B, T)

        _x_ref = _to_np(x_true)[zero_mask]                     # (Npts, D)
        _x_tgt = _to_np(x_post_p)[zero_mask]
        finite = np.isfinite(_x_tgt).all(axis=1) & np.isfinite(_x_ref).all(axis=1)
        _x_ref, _x_tgt = _x_ref[finite], _x_tgt[finite]

        if _x_ref.shape[0] < _MIN_AFFINE_FIT_POINTS:
            warnings.warn(
                f"Seed {seed}: only {_x_ref.shape[0]} non-interventional points for "
                f"affine fit (< {_MIN_AFFINE_FIT_POINTS}); using identity map.")
            return np.ones(D), np.zeros(D), {"r2": np.full(D, np.nan)}

        return fit_affine_per_dim(_x_ref, _x_tgt)

    def _latent_alignment(self) -> str:
        """Validated alignment mode for this run: 'B' or 'corr'."""
        return resolve_latent_alignment(
            getattr(self.fetch_settings(), "latent_alignment", "B"))

    @staticmethod
    def _warn_invalid(seed: int, site: str, counts: Dict[str, int], n_total: int) -> None:
        """Report the per-output breakdown behind an OR mask, if anything is invalid."""
        if not counts["total"]:
            return
        breakdown = ", ".join(f"{k}={v}" for k, v in counts.items() if k != "total")
        warnings.warn(
            f"Seed {seed} [{site}]: {counts['total']}/{n_total} trials "
            f"invalid ({breakdown}). Correlations score them 0.")

    @torch.no_grad()
    def _evaluate_condition(self,
                            model,
                            recognition,
                            u: torch.Tensor,
                            x_true: torch.Tensor,
                            y_true: torch.Tensor,
                            steps_encode: int,
                            x_points: torch.Tensor,
                            J_true: np.ndarray,
                            perm: np.ndarray,
                            d: np.ndarray,
                            c: np.ndarray,
                            D: int,
                            device: torch.device,
                            seed: int = 0):
        """Rollout, reconstruction and Jacobian metrics for one seed.

        The rollout gets x0 from the first steps_encode timesteps and runs
        open-loop; the reconstruction lets the encoder see y throughout. Both are
        scored on the same timesteps. Inputs are (B, steps_encode + eval_steps, *).

        Returns (metrics, y_pred_ex, x_pred_pdc_ex): trial 0's rollout, (rollout_len,
        N) and (rollout_len, D), NaN if trial 0 is invalid.
        """
        total_n_samples = y_true.shape[0]
        N_obs = y_true.shape[2]
        rollout_len = y_true.shape[1] - steps_encode
        inv_perm = np.argsort(perm)

        _nan_ex = (np.full((rollout_len, N_obs), np.nan),
                   np.full((rollout_len, D),     np.nan))

        D_mismatch = model.dynamics.D != D

        recognition.set_B_from_dynamics(model.dynamics)
        x0_all = recognition.encode_mean(y_true[:, :steps_encode, :],
                                         u[:, :steps_encode, :])[:, -1, :]

        u_ext = u[:, steps_encode - 1:, :]
        x_pred_all, y_pred_all = self._model_mean_eval(model, u_ext, x0_all)

        # Reconstruction has its own validity mask: it should not be scored down
        # for a rollout that diverged.
        y_rec_all, x_post = self._reconstruct(model, recognition, y_true, u)
        invalid_rec, counts_rec = invalid_trial_mask(
            posterior=x_post, emissions=y_rec_all)
        self._warn_invalid(seed, "recon", counts_rec, total_n_samples)

        # One per-trial validity mask for every metric. Invalid trials are scored,
        # not deleted: correlations get 0.
        invalid, counts = invalid_trial_mask(
            posterior=x0_all, latents=x_pred_all, emissions=y_pred_all)

        y_true_eval = y_true[:, steps_encode:, :]
        u_eval      = u[:,      steps_encode:, :]
        y_rec_eval  = y_rec_all[:, steps_encode:, :]

        # Same timesteps as the rollout, so the two are paired.
        y_rec_corr = compute_mean_corr(y_true_eval, y_rec_eval, u=u_eval,
                                       invalid_trials=invalid_rec)
        _recon_metrics = {"y_corr": y_rec_corr[0], "y_corr_std": y_rec_corr[1]}

        # Latent alignment and Jacobian — require system.D == model.D ---------
        if D_mismatch:
            self._warn_invalid(seed, "interv", counts, total_n_samples)
            y_corr = compute_mean_corr(y_true_eval, y_pred_all, u=u_eval,
                                       invalid_trials=invalid)
            metrics = {
                "y_corr_rollout":    y_corr[0], "y_corr_rollout_std": y_corr[1],
                **_recon_metrics,
                "x_pdc_aligned_corr": np.nan, "x_pdc_aligned_corr_std": np.nan,
                "jac_cosine": np.nan, "jac_cosine_std": np.nan,
            }
            return metrics, _nan_ex[0], _nan_ex[1]

        # Align: permutation → p_aligned; affine → pdc_aligned
        x_pred_p_aligned = apply_perm_latents(x_pred_all, perm)
        x_pred_pdc_aligned = canonicalize_to_ref(
            _to_np(x_pred_p_aligned).reshape(-1, D), d, c
        ).reshape(total_n_samples, rollout_len, D)

        # If the affine fit itself fails, the whole seed scores 0 (with a warning).
        invalid_aligned, _ = invalid_trial_mask(aligned=x_pred_pdc_aligned)
        invalid = invalid | invalid_aligned
        counts["aligned"] = int(invalid_aligned.sum())
        counts["total"]   = int(invalid.sum())
        self._warn_invalid(seed, "interv", counts, total_n_samples)

        y_corr = compute_mean_corr(y_true_eval, y_pred_all, u=u_eval,
                                   invalid_trials=invalid)

        x_true_eval = x_true[:, steps_encode:, :]
        x_pdc_corr = compute_mean_corr(x_true_eval, x_pred_pdc_aligned, u=u_eval,
                                       invalid_trials=invalid)

        # Jacobian similarity
        x_pts_np      = _to_np(x_points)
        x_pts_learned = (x_pts_np * d[None, :] + c[None, :])[:, inv_perm]
        J_pred   = compute_jacobians(model.dynamics, x_pts_learned, device=device)
        J_pred_t = transform_jacobians_to_ref(J_pred, perm_ref_to_tgt=perm, d=d)
        n_jac    = J_true.shape[0]
        jac_cos  = cosine_similarity(J_true.reshape(n_jac, -1), J_pred_t.reshape(n_jac, -1))

        metrics = {
            "y_corr_rollout":        y_corr[0],  "y_corr_rollout_std":        y_corr[1],
            **_recon_metrics,
            "x_pdc_aligned_corr":    x_pdc_corr[0], "x_pdc_aligned_corr_std":    x_pdc_corr[1],
            "jac_cosine":            float(np.nanmean(jac_cos)),
            "jac_cosine_std":        float(np.nanstd(jac_cos)),
        }

        # Examples are always trial 0, so the plotted prediction and the plotted
        # ground truth (which run() takes from trial 0) are the same trial.
        if invalid[0]:
            warnings.warn(
                f"Seed {seed}: trial 0 is invalid; example trajectories set to NaN.")
            return metrics, _nan_ex[0], _nan_ex[1]
        return metrics, _to_np(y_pred_all[0]), x_pred_pdc_aligned[0]

    @torch.no_grad()
    def _evaluate_seed(self,
                       seed: int,
                       model,
                       recognition,
                       y_true: torch.Tensor,
                       u: torch.Tensor,
                       x_true: torch.Tensor,
                       B_true: np.ndarray,
                       D: int,
                       align_mode: str):
        """Align latents to the true system, fit affine maps and score B.

        Returns (metrics, perm, (d, c)).
        """
        # Guard: latent evaluation requires system.D == model.D ---------------
        model_D = model.dynamics.D
        if model_D != D:
            warnings.warn(
                f"Seed {seed}: system.D={D} != model.D={model_D}. "
                "Skipping all latent metrics (B alignment, Jacobian, affine fit).")
            return _nan_metrics_row(seed), np.arange(D), (np.ones(D), np.zeros(D))

        # Latent alignment ---------------------------------------------------
        # Set by settings.latent_alignment, not the model, so all models under one
        # config are aligned the same way.
        train_B = model.dynamics.train_B
        x_post_align = None
        if align_mode == "corr":
            recognition.set_B_from_dynamics(model.dynamics)
            x_post_align = recognition.encode_mean(y_true, u)   # deterministic, RNG-free
        perm = latent_perm(
            align_mode, D=D, n_stim=model.dynamics.n_stim, train_B=train_B,
            B_ref=B_true, B_tgt=_to_np(model.dynamics.B_effective),
            x_ref=x_true, x_tgt=x_post_align, u=u)

        # Affine fitting (driven full-trial posterior) -----------------------
        d, c, fit_stats = self._fit_affine(
            model, recognition, y_true, u, x_true, perm, D, seed)

        metrics = {
            "seed": seed,
            "r2_mean": float(np.nanmean(fit_stats["r2"])),
        }

        # B faithfulness ----------------------------------------------------
        # Undefined when the learned B has a different shape than the system's,
        # e.g. a model with no stim channels (n_stim=0) whose B is (D, 0).
        B_learned = model.dynamics.B.detach().cpu().numpy()
        B_aligned_seed = B_learned[perm, :]
        if B_aligned_seed.shape == B_true.shape:
            bt_flat = B_true.flatten()
            bl_flat = B_aligned_seed.flatten()
            denom = np.linalg.norm(bt_flat) * np.linalg.norm(bl_flat)
            metrics["b_cosine"] = float(np.dot(bt_flat, bl_flat) / denom) if denom > 1e-12 else np.nan
            zero_mask = B_true == 0
            metrics["b_zero_mae"] = (
                float(np.mean(np.abs(B_aligned_seed[zero_mask]))) if zero_mask.any() else np.nan)
        else:
            metrics["b_cosine"] = np.nan
            metrics["b_zero_mae"] = np.nan

        return metrics, perm, (d, c)

    # -----------------------------------------------------------------------
    # Tabular output
    # -----------------------------------------------------------------------

    def _save_tabular_outputs(self,
                              metrics_df: pd.DataFrame,
                              permutations: Dict,
                              affine_maps: Dict,
                              run_dirs: Dict,
                              output_dir: str) -> None:
        """Save the metrics CSV and the per-seed affine maps."""
        metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

        # Per-seed affine params as JSON (all in TRUE latent order):
        #   d[j], c_diag[j]: x_pred_aligned[:, j] ≈ d[j] * x_true[:, j] + c_diag[j]
        #   perm[j]:          learned dim index that corresponds to true dim j
        # run_dir is relative to the repo root.
        affine_json = {}
        for s in sorted(affine_maps.keys()):
            d_s, c_s = affine_maps[s]
            affine_json[f"seed_{s}"] = {
                "run_dir": os.path.relpath(run_dirs[s], get_root()),
                "perm": permutations[s].tolist(),
                "d": d_s.tolist(),
                "c_diag": c_s.tolist(),
            }
        save_json(affine_json, os.path.join(output_dir, "affine_maps_per_seed.json"))

    # -----------------------------------------------------------------------
    # Main run
    # -----------------------------------------------------------------------

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        device = self.get_device()
        s = self.fetch_settings()

        # Load models -------------------------------------------------------
        group_dir = add_root(s.group_dir)
        seed_dirs = sorted([d for d in os.listdir(group_dir) if d.startswith("seed_")])
        run_dirs = {int(d.split("_")[-1]): os.path.join(group_dir, d) for d in seed_dirs}
        print(f"Loading {len(seed_dirs)} seed models from {group_dir}")
        joint_models, recognition_models = self._load_models(group_dir, seed_dirs, device)
        n_models = len(joint_models)
        align_mode = self._latent_alignment()

        # System and stimulator ---------------------------------------------
        first_seed_run = os.path.join(group_dir, seed_dirs[0])
        system = build_component_from_run(first_seed_run, "system")
        stimulator = self.fetch_components().stimulator.build()
        self._scale_stim_amplitude(stimulator, s.stim_amp_scaling)
        print(f"  Eval stimulator: {type(stimulator).__name__}, amplitude "
              f"[{stimulator.min_amplitude:.3f}, {stimulator.max_amplitude:.3f}], "
              f"stim_d={stimulator.stim_d}, sigma={stimulator.sigma} "
              f"(stim_amp_scaling={s.stim_amp_scaling}).")

        # Read dt and T_burn from training config (must match the trained model)
        train_cfg = load_json(os.path.join(first_seed_run, "config.json"))
        dt = train_cfg["settings"]["dt"]
        if "T_burn" not in train_cfg["settings"]:
            raise ValueError(
                f"Training config {os.path.join(first_seed_run, 'config.json')!r} "
                "has no settings.T_burn; cannot generate eval data.")
        T_burn = train_cfg["settings"]["T_burn"]
        print(f"  Using dt={dt}, T_burn={T_burn} from training config.")

        D = system.D
        B_true = system.B.detach().cpu().numpy()  # (D, M)

        # Generate test data ------------------------------------------------
        max_seed = max(int(d.split("_")[-1]) for d in seed_dirs)
        eval_seed = max_seed + self.seed + 1
        fix_random_seed(eval_seed)
        stimulator.seed = eval_seed
        system.seed = eval_seed
        u, x_true, y_true = self._generate_data(system, stimulator, dt, T_burn)

        # Every correlation below assumes scorable ground truth.
        assert_true_trials_scorable(y_true, u=u, name="interventional y_true")
        assert_true_trials_scorable(x_true, u=u, name="interventional x_true")

        steps_encode = int(s.T_encode / dt)
        u = u.to(device)
        x_true = x_true.to(device)
        y_true = y_true.to(device)

        # Latent points for Jacobian analysis
        x_points = self._sample_latent_points(x_true[:, steps_encode:, :])
        J_true   = compute_jacobians(system, x_points, device=device)

        # Per-seed evaluation -----------------------------------------------
        metrics = []
        y_preds, x_preds_pdc_aligned = {}, {}
        permutations = {s_: None for s_ in joint_models.keys()}
        affine_maps  = {s_: None for s_ in joint_models.keys()}

        for i, (seed, model) in enumerate(joint_models.items()):
            print(f"  Evaluating seed {seed} ({i+1}/{n_models})...")
            rec = recognition_models[seed]

            seed_metrics, perm, affine_map = self._evaluate_seed(
                seed, model, rec, y_true, u, x_true, B_true, D, align_mode=align_mode)
            d, c = affine_map

            cond_metrics, y_pred_ex, x_pred_pdc_ex = self._evaluate_condition(
                model, rec, u, x_true, y_true, steps_encode,
                x_points, J_true, perm, d, c, D, device, seed=seed)

            metrics.append({**seed_metrics, **cond_metrics})
            y_preds[seed]             = y_pred_ex
            x_preds_pdc_aligned[seed] = x_pred_pdc_ex
            permutations[seed]        = perm
            affine_maps[seed]         = affine_map

        metrics_df = pd.DataFrame(metrics)

        # With system.D != model.D (e.g. a latent-dim sweep) only metrics.csv is written.
        first_model = next(iter(joint_models.values()))
        D_model = first_model.dynamics.D
        D_mismatch = D_model != D
        if D_mismatch:
            warnings.warn(
                f"system.D={D} != model.D={D_model}. "
                "Skipping example and B outputs; metrics.csv still saved.")

        # Tabular outputs ---------------------------------------------------
        self._save_tabular_outputs(
            metrics_df, permutations, affine_maps, run_dirs, self.output_dir)

        if D_mismatch:
            print(f"Done. Outputs saved to {self.output_dir}")
            return

        # Eval portions for saving (encoder-unseen steps only)
        y_eval = y_true[:, steps_encode:, :]
        x_eval = x_true[:, steps_encode:, :]
        u_eval = u[:, steps_encode:, :]

        seeds_sorted = sorted(y_preds.keys())
        y_stack = np.stack([_to_np(y_preds[s_]) for s_ in seeds_sorted])
        x_stack = np.stack([_to_np(x_preds_pdc_aligned[s_]) for s_ in seeds_sorted])
        np.savez(
            os.path.join(self.output_dir, "example_plot_data.npz"),
            u=_to_np(u_eval[0]),
            y_true=_to_np(y_eval[0]),
            y_mean=y_stack.mean(axis=0),
            y_std=y_stack.std(axis=0),
            x_true=_to_np(x_eval[0]),
            x_pdc_aligned_mean=x_stack.mean(axis=0),
            x_pdc_aligned_std=x_stack.std(axis=0),
            dt=np.float64(dt))

        # B matrix summary (only when train_B=True)
        if first_model.dynamics.train_B:
            B_arr = np.stack([
                joint_models[s_].dynamics.B.detach().cpu().numpy()[permutations[s_], :]
                for s_ in sorted(joint_models.keys())
            ])
            B_abs_arr = np.abs(B_arr)
            np.savez(
                os.path.join(self.output_dir, "B_matrix_summary.npz"),
                B_abs_mean=B_abs_arr.mean(axis=0),
                B_abs_std=B_abs_arr.std(axis=0),
                B_true=B_true,
            )

        print(f"Done. Outputs saved to {self.output_dir}")
