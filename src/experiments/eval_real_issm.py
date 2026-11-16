"""
Abstract base experiment for evaluating an iSSM trained on real neural recordings.

Rollouts start from an x0 that the encoder infers from the first steps_encode timesteps
only, chosen to end before the stimulation, so the dynamics alone must predict the
stimulation response.
"""

import os
import warnings
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models import FlexibleLDS
from src.utils.io import add_root, load_json, load_component_from_run
from src.utils.metrics import (
    compute_mean_corr, compute_corr_per_trial,
    invalid_trial_mask, assert_true_trials_scorable,
)
from .base import Experiment


# Must match the split seed in train_real_issm.py.
_SPLIT_SEED = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_stim_index(u_stim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """First timestep at which any stim channel is non-zero, per trial.

    Args:
        u_stim: (B, T, n_stim) stimulation channels only.

    Returns:
        (onset, has_stim): onset is (B,) long — the first non-zero index, or T for
        trials that are never stimulated; has_stim is (B,) bool.
    """
    T = u_stim.shape[1]
    nz = (u_stim != 0).any(dim=-1)                                    # (B, T)
    idx = torch.arange(T, device=nz.device).expand_as(nz)
    # min over masked indices rather than argmax, which is unspecified on ties
    onset = torch.where(nz, idx, torch.full_like(idx, T)).min(dim=1).values
    return onset, nz.any(dim=1)


def _load_standardization(
    run_dir: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load y_mean / y_std saved during training, if they exist."""
    mean_path = os.path.join(run_dir, "y_mean.npy")
    std_path  = os.path.join(run_dir, "y_std.npy")
    if os.path.isfile(mean_path) and os.path.isfile(std_path):
        y_mean = torch.tensor(np.load(mean_path), dtype=torch.float32)
        y_std  = torch.tensor(np.load(std_path),  dtype=torch.float32)
        return y_mean, y_std
    return None, None


def _load_zero_fill_marker(run_dir: str) -> bool:
    """True if training zero-filled masked y cells (splits.json marker; absent = False)."""
    path = os.path.join(run_dir, "splits.json")
    if not os.path.isfile(path):
        return False
    return bool(load_json(path).get("zero_fill_masked_obs", False))


# Per-seed columns produced by _unordered_pairs_stats, merged into metrics.csv.
_UP_COLUMNS = [
    "up_frac", "up_pairs_separated", "up_pairs_total",
    "up_n_blocks", "up_max_block", "up_n_never_intervened",
    "up_n_environments", "B_support_frac", "B_mean_set_size",
]

# Multipliers on B_threshold for the threshold-sensitivity sweep of the audit.
_UP_TAU_MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 10.0, 50.0]


def _unordered_pairs_stats(B: np.ndarray, tau: float) -> dict:
    """Audit the unordered pairs condition on a learned B.

    Intervention environment k drives the latent set S_k = {d : |B[d, k]| > tau},
    the same support the model uses for interventional masking
    (TransitionModel.get_inputs). Latent d's signature is row d of Z = |B| > tau,
    and a pair (i, j) is separated iff some environment contains exactly one of
    them, i.e. iff Z[i] != Z[j]. The condition holds iff all D signatures are
    distinct; latents sharing a signature form one block, and latents are
    identifiable only up to such blocks.

    Args:
        B:   (D, n_stim) intervention matrix.
        tau: magnitude below which an entry is treated as no intervention.

    Returns:
        dict of the _UP_COLUMNS scalars, plus 'blocks' (list of latent-index lists,
        largest first) and 'Z' (the (D, n_stim) bool signature matrix).
    """
    Z = np.abs(B) > tau                                    # (D, n_stim)
    D = Z.shape[0]

    # Group latents by signature; each group is one identifiability block.
    groups: Dict[bytes, List[int]] = {}
    for d in range(D):
        groups.setdefault(Z[d].tobytes(), []).append(d)
    blocks = sorted(groups.values(), key=lambda g: (-len(g), g[0]))

    # Pairs within a block share a signature and are exactly the unseparated ones.
    n_pairs = D * (D - 1) // 2
    n_unseparated = sum(len(g) * (len(g) - 1) // 2 for g in blocks)
    n_separated = n_pairs - n_unseparated

    return {
        "up_frac": float(n_separated / n_pairs) if n_pairs else float("nan"),
        "up_pairs_separated": int(n_separated),
        "up_pairs_total": int(n_pairs),
        "up_n_blocks": int(len(blocks)),
        "up_max_block": int(max(len(g) for g in blocks)) if blocks else 0,
        "up_n_never_intervened": int((~Z.any(axis=1)).sum()),
        "up_n_environments": int(Z.shape[1]),
        "B_support_frac": float(Z.mean()),
        "B_mean_set_size": float(Z.sum(axis=0).mean()),
        "blocks": blocks,
        "Z": Z,
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EvalRealISSM(Experiment):
    """Abstract base: evaluate a multi-seed iSSM trained on real neural recordings.

    Subclasses must implement:
        _load_test_loader(train_config)  — return a DataLoader for test data.

    Everything else (model loading, metric computation, CSV outputs)
    is handled here and shared across all real-data evaluation variants.
    """

    @abstractmethod
    def _load_test_loader(self, train_config: dict) -> Optional[DataLoader]:
        """Return a DataLoader for the test set.

        Subclasses either:
        - Extract held-out test trials from the same data pool used at training
          time (reproducing the split with _SPLIT_SEED), or
        - Load a different data pool when eval-override settings are
          provided in the eval config.

        Must return None if no test data is available (e.g. n_test_samples=0
        and no override was supplied).
        """
        ...

    # -----------------------------------------------------------------------
    # preprocess_config / check_config
    # -----------------------------------------------------------------------

    @classmethod
    def preprocess_config(cls, config: dict) -> dict:
        settings = config.get("settings", {})
        if settings.get("use_default_output_dir", True):
            group_dir = settings.get("group_dir", "")
            if group_dir:
                default_output = os.path.join(group_dir, "eval")
                provided = config.get("output_dir", "./outputs/default")
                if provided not in ("./outputs", default_output):
                    print(f"Warning: use_default_output_dir=True; ignoring output_dir={provided!r}, using {default_output!r}")
                config = {**config, "output_dir": default_output}
        return config

    def check_config(self):
        self.match_settings(strict=False)
        s = self.fetch_settings()
        group_dir = add_root(s.group_dir)
        if not os.path.isdir(group_dir):
            raise ValueError(f"group_dir is not a directory: {group_dir}")
        seed_dirs = sorted([d for d in os.listdir(group_dir) if d.startswith("seed_")])
        if not seed_dirs:
            raise ValueError(f"group_dir has no seed_* subdirectories: {group_dir}")
        for seed_dir in seed_dirs:
            run_dir = os.path.join(group_dir, seed_dir)
            for name in ["dynamics", "emission", "likelihood", "recognition"]:
                pth = os.path.join(run_dir, f"{name}.pth")
                if not os.path.isfile(pth):
                    raise ValueError(f"Missing weight file: {pth}")
            cfg_path = os.path.join(run_dir, "config.json")
            if not os.path.isfile(cfg_path):
                raise ValueError(f"Missing config.json in {run_dir}")
            # A run whose config declares a b_net component produces B from the
            # net (dynamics.B is None), so b_net.pth is required to reconstruct it.
            run_cfg = load_json(cfg_path)
            if "b_net" in run_cfg.get("components", {}) and not os.path.isfile(
                    os.path.join(run_dir, "b_net.pth")):
                raise ValueError(
                    f"config declares a 'b_net' component but b_net.pth is missing in {run_dir}")

    # -----------------------------------------------------------------------
    # Rollout encode window
    # -----------------------------------------------------------------------

    def _resolve_steps_encode(self, u_stim: torch.Tensor) -> int:
        """Length of the encoder's input window for the rollout x0.

        The window is a single scalar for the whole test set, so every trial's x0 is
        inferred from the same amount of data. It is resolved from the settings:

            steps_encode=None  -> the earliest stim onset over all test trials, i.e.
                                  the pre-stim timesteps of the trial stimulated first.
                                  No trial can then leak, whatever its own onset.
            steps_encode=<int> -> used as given.

        min_steps_encode is then applied as a floor to that value.

        Args:
            u_stim: (B, T, n_stim) stimulation channels of the whole test set.
        """
        T = int(u_stim.shape[1])
        s = self.fetch_settings()
        requested = getattr(s, "steps_encode", None)
        min_steps = int(getattr(s, "min_steps_encode", 25))

        onset, has_stim = _first_stim_index(u_stim)
        stim_onsets = onset[has_stim]

        if requested is None:
            if stim_onsets.numel() == 0:
                raise ValueError(
                    "No stimulated trial in the test set, so the rollout encode window "
                    "cannot be inferred. Set settings.steps_encode explicitly.")
            c = int(stim_onsets.min())
            source = f"Inferred steps_encode={c} (earliest stim onset)"
        else:
            c = int(requested)
            if not 1 <= c < T:
                raise ValueError(
                    f"settings.steps_encode={c} out of range: must be >= 1 and < T={T}.")
            source = f"settings.steps_encode={c}"

        # Floor. Applied to whichever value was resolved above, so `min` means the same
        # thing whether the window was inferred or set by hand.
        if c < min_steps:
            if min_steps >= T:
                raise ValueError(
                    f"min_steps_encode={min_steps} leaves no room in a trial of T={T}: "
                    f"the encoder would see the whole trial. Lower min_steps_encode "
                    f"below {T}.")
            earliest = torch.nonzero(has_stim & (onset == int(stim_onsets.min()))
                                     ).flatten().tolist()
            warnings.warn(
                f"{source} is below min_steps_encode={min_steps}; running with "
                f"{min_steps} instead. Only {c} pre-stim timesteps exist in trial(s) "
                f"{earliest} (median onset {int(stim_onsets.median())} of T={T}), which "
                f"is too few to infer a usable x0 from. Lower min_steps_encode to score "
                f"the leak-free window instead.")
            c = min_steps

        # Computed on the final window, so it also covers trials made leaky by the floor.
        n_leaky = int((stim_onsets < c).sum())
        if n_leaky:
            warnings.warn(
                f"steps_encode={c} is later than the stim onset of {n_leaky}/"
                f"{int(has_stim.sum())} stimulated test trials: the encoder sees their "
                f"stimulation response, so their rollout leaks. Earliest "
                f"onset is {int(stim_onsets.min())}.")

        print(f"  Rollout: encoder sees y[:, :{c}] of T={T}, x0 at t=0; stim onsets "
              f"min={int(stim_onsets.min()) if stim_onsets.numel() else None}, "
              f"median={int(stim_onsets.median()) if stim_onsets.numel() else None}"
              f"{f'; {n_leaky} trial(s) leak' if n_leaky else ''}")
        return c

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_models(self, group_dir: str, seed_dirs: List[str]
    ) -> Tuple[Dict[int, FlexibleLDS], Dict]:
        """Load FlexibleLDS joint models and recognition networks for all seeds."""
        joint_models: Dict[int, FlexibleLDS] = {}
        recognition_models: Dict = {}
        for seed_dir in seed_dirs:
            seed = int(seed_dir.split("_")[-1])
            run_dir = os.path.join(group_dir, seed_dir)
            dynamics = load_component_from_run(run_dir, "dynamics")
            # Optional B-embedding net: must be wired before any B_effective read;
            # B_net's init_scale_set/features_set are restored from b_net.pth.
            run_cfg = load_json(os.path.join(run_dir, "config.json"))
            if "b_net" in run_cfg.get("components", {}):
                dynamics.set_B_net(load_component_from_run(run_dir, "b_net"))
            joint_models[seed] = FlexibleLDS(
                dynamics=dynamics,
                emission=load_component_from_run(run_dir, "emission"),
                likelihood=load_component_from_run(run_dir, "likelihood"),
            ).eval()
            rec = load_component_from_run(run_dir, "recognition")
            rec.set_B_from_dynamics(joint_models[seed].dynamics)
            recognition_models[seed] = rec.eval()
        return joint_models, recognition_models

    # -----------------------------------------------------------------------
    # Forward pass helper
    # -----------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _collect_predictions(joint, recognition, loader, steps_encode: int
    ) -> Tuple[torch.Tensor, ...]:
        """One forward pass over loader.

        The posterior and its reconstruction see the whole trial; the rollout's x0 is
        inferred from the first steps_encode timesteps only.

        Returns:
            y:         (B, T, N) — raw observations from loader
            u:         (B, T, M) — inputs from loader
            y_pred:    (B, T, N) — emission mean (encoder-decoded)
            x_encoded: (B, T, D) — encoder posterior mean
            x_rollout: (B, T, D) — rollout latents (x0 at index 0)
            y_rollout: (B, T, N) — rollout emission mean
        """
        y_list, u_list, y_pred_list, x_enc_list, x_roll_list, y_roll_list = [], [], [], [], [], []

        for y_batch, u_batch, _ in loader:
            x_enc = recognition.encode_mean(y_batch, u_batch)
            y_pred = joint.likelihood.mean(joint.emission(x_enc))
            x0 = recognition.encode_mean(
                y_batch[:, :steps_encode], u_batch[:, :steps_encode])[:, 0]
            x_r, y_r = joint.mean(u=u_batch, x0=x0)

            y_list.append(y_batch)
            u_list.append(u_batch)
            y_pred_list.append(y_pred)
            x_enc_list.append(x_enc)
            x_roll_list.append(x_r)
            y_roll_list.append(y_r)

        return tuple(torch.cat(lst, dim=0) for lst in
                     (y_list, u_list, y_pred_list, x_enc_list, x_roll_list, y_roll_list))

    # -----------------------------------------------------------------------
    # Unordered pairs condition
    # -----------------------------------------------------------------------

    @staticmethod
    def _unordered_pairs_analysis(
        joint_models: Dict,
        u_all: torch.Tensor,
        seeds_sorted: List[int],
        output_dir: str,
    ) -> Dict[int, dict]:
        """Check the unordered pairs condition on each seed's learned B.

        The intervention environments are the n_stim single-channel interventions the
        experimental design offers, i.e. the columns of B_effective — not the u patterns
        that happen to land in the eval split, which can miss a stim site entirely.
        up_n_environments_observed records how many distinct non-empty driven sets the
        eval data actually realises, so that gap stays visible.

        Saves unordered_pairs.csv, unordered_pairs_threshold_sweep.csv and B_signatures.csv.

        Returns:
            {seed: dict of _UP_COLUMNS scalars + up_n_environments_observed}
        """
        per_seed: Dict[int, dict] = {}
        sweep_rows: List[dict] = []
        signature_rows: List[dict] = []

        for seed in seeds_sorted:
            dynamics = joint_models[seed].dynamics
            B = dynamics.B_effective.detach().cpu().numpy()
            tau = float(dynamics.B_threshold)
            stats = _unordered_pairs_stats(B, tau)

            # Distinct driven sets the eval data realises, via the model's own masking.
            with torch.no_grad():
                driven = (dynamics.get_inputs(u_all).abs() > 0).reshape(-1, B.shape[0])
            driven = torch.unique(driven, dim=0)
            n_observed = int(driven.any(dim=1).sum().item())

            row = {c: stats[c] for c in _UP_COLUMNS}
            row["up_n_environments_observed"] = n_observed
            per_seed[seed] = row

            for m in _UP_TAU_MULTIPLIERS:
                s_m = _unordered_pairs_stats(B, tau * m)
                sweep_rows.append({
                    "seed": seed, "tau_multiplier": m, "tau": tau * m,
                    "up_frac": s_m["up_frac"],
                    "up_pairs_separated": s_m["up_pairs_separated"],
                    "up_n_never_intervened": s_m["up_n_never_intervened"],
                    "B_support_frac": s_m["B_support_frac"],
                })

            Z = stats["Z"]
            for block_id, block in enumerate(stats["blocks"]):
                for d in block:
                    driving = np.where(Z[d])[0]
                    signature_rows.append({
                        "seed": seed, "latent": d, "block_id": block_id,
                        "block_size": len(block),
                        # "." rather than "0": keeps the column non-numeric, so readers
                        # cannot silently parse it as an int and strip leading zeros.
                        "signature": "".join("1" if z else "." for z in Z[d]),
                        "n_driving_environments": int(len(driving)),
                        "driving_environments": " ".join(str(k) for k in driving),
                    })

        up_df = pd.DataFrame(
            [{"seed": s, **per_seed[s]} for s in seeds_sorted])
        up_df.to_csv(os.path.join(output_dir, "unordered_pairs.csv"), index=False)
        pd.DataFrame(sweep_rows).to_csv(
            os.path.join(output_dir, "unordered_pairs_threshold_sweep.csv"), index=False)
        pd.DataFrame(signature_rows).to_csv(
            os.path.join(output_dir, "B_signatures.csv"), index=False)

        return per_seed

    # -----------------------------------------------------------------------
    # Multi-tier evaluation
    # -----------------------------------------------------------------------

    def _get_trial_tiers(
        self,
        train_config: dict,
        u_all: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """Return (n_trials,) string array of tier labels, or None to skip.

        Tier names: "Control", "Stim", "Stim_OOD".
        Base class returns None. Override in dataset-specific subclasses.
        """
        return None

    def _run_multi_tier_eval(
        self,
        train_config: dict,
        y_true: torch.Tensor,
        u_all: torch.Tensor,
        y_pred_enc_mean: torch.Tensor,
        y_pred_rollout_mean: torch.Tensor,
        trial_indices: Optional[np.ndarray] = None,
        u_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Compute per-trial metrics labelled by trial tier and save them as CSV.

        Args:
            train_config:        training config dict (passed to _get_trial_tiers).
            y_true:              (B, T, N) un-standardized observations.
            u_all:               (B, T, M) inputs.
            y_pred_enc_mean:     (B, T, N) seed-mean encoder predictions.
            y_pred_rollout_mean: (B, T, N) seed-mean rollout predictions.
            trial_indices:       (B,) pool indices, a stable trial id across runs
                                 for paired comparisons. Defaults to 0..B-1.
        """
        tier_labels = self._get_trial_tiers(train_config, u_all)
        if tier_labels is None:
            return

        n = y_true.shape[0]
        tidx = trial_indices if trial_indices is not None else np.arange(n)
        _u = u_mask if u_mask is not None else u_all

        # These are seed-MEAN predictions, so a trial is invalid here only when every
        # seed diverged on it (the NaN-aware stacking already averages over the rest).
        invalid, _ = invalid_trial_mask(
            enc=y_pred_enc_mean, rollout=y_pred_rollout_mean)
        _inv = dict(invalid_trials=invalid)

        cols = {
            "trial_idx":      tidx,
            "trial_tier":     tier_labels,
            "y_corr":         compute_corr_per_trial(
                y_true, y_pred_enc_mean, u=_u, mask=mask, **_inv),
            "y_corr_rollout": compute_corr_per_trial(
                y_true, y_pred_rollout_mean, u=_u, mask=mask, **_inv),
        }

        pd.DataFrame(cols).to_csv(
            os.path.join(self.output_dir, "multi_tier_eval_metrics.csv"), index=False)

    # -----------------------------------------------------------------------
    # run()
    # -----------------------------------------------------------------------

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        s = self.fetch_settings()

        # Discover seed directories -----------------------------------------
        group_dir = add_root(s.group_dir)
        seed_dirs = sorted([d for d in os.listdir(group_dir) if d.startswith("seed_")])
        first_run  = os.path.join(group_dir, seed_dirs[0])
        train_config = load_json(os.path.join(first_run, "config.json"))

        print(f"Evaluating {len(seed_dirs)} seed(s) from {group_dir}")

        # Load models --------------------------------------------------------
        joint_models, recognition_models = self._load_models(group_dir, seed_dirs)
        seeds_sorted = sorted(joint_models.keys())
        train_B = next(iter(joint_models.values())).dynamics.train_B
        n_stim  = next(iter(joint_models.values())).dynamics.n_stim

        # Expose first-run dir to subclasses so they can load splits.json.
        self._first_run_dir = first_run

        # Load test data (built once, shared across seeds) -------------------
        test_loader = self._load_test_loader(train_config)
        if test_loader is None:
            raise RuntimeError(
                "No test data available. Either set n_test_samples > 0 in the "
                "training config, or provide eval override settings to load a "
                "separate evaluation dataset.")

        # Apply training-time standardization to test data ------------------
        y_mean, y_std = _load_standardization(first_run)
        if y_mean is not None:
            test_loader.dataset.y = (test_loader.dataset.y - y_mean) / y_std

        # Observation mask (True = ignore), or None for datasets without one.
        # Applied only to y-space metrics.
        test_mask = getattr(test_loader.dataset, "mask", None)

        # Match training: masked cells are zero-filled iff training did so.
        if test_mask is not None and _load_zero_fill_marker(first_run):
            test_loader.dataset.y = test_loader.dataset.y.masked_fill(test_mask, 0.0)

        # Hard error on a non-finite or fully flat trial.
        _n_stim_guard = next(iter(joint_models.values())).dynamics.n_stim
        _y_ds = test_loader.dataset.y
        _u_ds = test_loader.dataset.u
        _u_ds_stim = (_u_ds[..., :_n_stim_guard]
                      if _n_stim_guard < _u_ds.shape[-1] else _u_ds)
        assert_true_trials_scorable(
            _y_ds, u=_u_ds_stim, mask=test_mask, name="test y")

        steps_encode = self._resolve_steps_encode(_u_ds_stim)

        # Per-seed evaluation ------------------------------------------------
        metrics: List[dict] = []
        y_preds: Dict[int, torch.Tensor] = {}
        y_preds_rollout: Dict[int, torch.Tensor] = {}

        for seed_dir in seed_dirs:
            seed  = int(seed_dir.split("_")[-1])
            joint = joint_models[seed]
            rec   = recognition_models[seed]
            print(f"  Seed {seed}...")

            y_all, u_all, y_pred_all, x_enc, x_roll, y_roll = self._collect_predictions(
                joint, rec, test_loader, steps_encode)

            # Stim-only slice of u for masking (excludes task-feature tail channels).
            u_for_mask = u_all[..., :n_stim] if n_stim < u_all.shape[-1] else u_all

            # A trial whose posterior or rollout diverged is scored 0 on
            # correlations, never dropped from the denominator.
            invalid, counts = invalid_trial_mask(
                posterior=x_enc, encoder_emissions=y_pred_all, latents=x_roll,
                rollout_emissions=y_roll)
            if counts["total"]:
                breakdown = ", ".join(f"{k}={v}" for k, v in counts.items() if k != "total")
                warnings.warn(
                    f"Seed {seed}: {counts['total']}/{len(invalid)} trials invalid "
                    f"({breakdown}). Correlations score them 0.")

            # Reverse standardisation before computing observation metrics
            if y_mean is not None:
                y_eval, y_pred_eval, y_roll_eval = (
                    t * y_std + y_mean for t in (y_all, y_pred_all, y_roll))
            else:
                y_eval, y_pred_eval, y_roll_eval = y_all, y_pred_all, y_roll

            y_corr, y_corr_std = compute_mean_corr(
                y_eval, y_pred_eval, u=u_for_mask, mask=test_mask, invalid_trials=invalid)
            corr_r, corr_r_std = compute_mean_corr(
                y_eval, y_roll_eval, u=u_for_mask, mask=test_mask, invalid_trials=invalid)
            metrics.append({
                "seed":               seed,
                "y_corr":             y_corr,
                "y_corr_std":         y_corr_std,
                "y_corr_rollout":     corr_r,
                "y_corr_rollout_std": corr_r_std,
            })
            y_preds[seed] = y_pred_eval
            y_preds_rollout[seed] = y_roll_eval

        # u_all and y_all are identical across seeds; the last iteration's are used.

        # Unordered pairs condition (per seed) -------------------------------
        up_per_seed: Dict[int, dict] = {}
        if train_B and n_stim > 0:
            print("  Auditing unordered-pairs condition...")
            up_per_seed = self._unordered_pairs_analysis(
                joint_models, u_all, seeds_sorted, self.output_dir)

        # Tabular outputs ----------------------------------------------------
        metrics_df = pd.DataFrame(metrics)

        for col in (_UP_COLUMNS + ["up_n_environments_observed"]) if up_per_seed else []:
            metrics_df[col] = metrics_df["seed"].map(
                {s: v[col] for s, v in up_per_seed.items()})

        metrics_df.to_csv(
            os.path.join(self.output_dir, "metrics.csv"), index=False)

        # Seed-mean predictions ----------------------------------------------
        y_true_plot = y_all * y_std + y_mean if y_mean is not None else y_all

        # Pick a stim trial for the example prediction
        trial_to_plot = 0
        for i in range(u_for_mask.shape[0]):
            if u_for_mask[i].abs().sum() > 0:
                trial_to_plot = i
                break

        def _seed_mean_std(stack: torch.Tensor, ddof: int = 1
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
            if torch.isfinite(stack).all():
                if ddof == 0:
                    arr = stack.detach().cpu().numpy()
                    return (torch.from_numpy(arr.mean(axis=0)),
                            torch.from_numpy(arr.std(axis=0)))
                return stack.mean(dim=0), stack.std(dim=0, correction=ddof)
            warnings.warn("Non-finite predictions excluded from the seed-mean "
                          "predictions.")
            arr = stack.detach().cpu().numpy().astype(np.float64)
            arr[~np.isfinite(arr)] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
                return (torch.from_numpy(np.nanmean(arr, axis=0)),
                        torch.from_numpy(np.nanstd(arr, axis=0, ddof=ddof)))

        # Rollouts
        y_pred_roll_stack = torch.stack([y_preds_rollout[s_] for s_ in seeds_sorted])  # (S, B, T, N)
        y_pred_roll_mean, _ = _seed_mean_std(y_pred_roll_stack)

        # Encoder reconstructions
        y_pred_enc_stack = torch.stack([y_preds[s_] for s_ in seeds_sorted])  # (S, B, T, N)
        y_pred_enc_mean, _ = _seed_mean_std(y_pred_enc_stack)

        # Example trial across seeds (NaN-aware, so one diverged seed cannot
        # blank it).
        _enc_mean_0, _enc_std_0 = _seed_mean_std(y_pred_enc_stack, ddof=0)
        _ro_mean_0,  _ro_std_0  = _seed_mean_std(y_pred_roll_stack, ddof=0)
        _npz_kwargs = dict(
            u              = u_for_mask[trial_to_plot].cpu().numpy(),
            y_true         = y_true_plot[trial_to_plot].cpu().numpy(),
            y_mean         = _enc_mean_0[trial_to_plot].cpu().numpy(),
            y_std          = _enc_std_0[trial_to_plot].cpu().numpy(),
            y_rollout_mean = _ro_mean_0[trial_to_plot].cpu().numpy(),
            y_rollout_std  = _ro_std_0[trial_to_plot].cpu().numpy(),
            dt             = np.float64(train_config["settings"]["dt"]),
        )
        # Observation mask of the example trial, if any.
        if test_mask is not None:
            _npz_kwargs["y_mask"] = test_mask[trial_to_plot].cpu().numpy()  # (T, N)
        np.savez(os.path.join(self.output_dir, "example_plot_data.npz"), **_npz_kwargs)

        # Multi-tier evaluation (control / stim / stim-OOD)
        self._run_multi_tier_eval(
            train_config, y_true_plot, u_all,
            y_pred_enc_mean, y_pred_roll_mean,
            trial_indices=getattr(self, "_test_trial_indices", None),
            u_mask=u_for_mask,
            mask=test_mask,
        )

        print(f"Done. Outputs saved to {self.output_dir}")
