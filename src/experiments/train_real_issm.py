"""
Abstract base experiment for training an iSSM on real neural recordings.
"""

import json
import os
from abc import abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 80 # to avoid DecompressionBombError with wandb

import torch
try:
    import wandb
except ImportError:
    wandb = None

from src.datasets import get_dataloaders_real, masked_channel_stats
from src.models import FlexibleLDS
from src.training import get_optimizer, run_training_loop
from src.utils.metrics import compute_mean_corr, compute_mean_mae, compute_corr_per_trial
from src.utils.plotting import (
    plot_training_metrics, plot_x_pred_only,
    plot_y_predictions_pair, get_cmap,
)
from src.utils.stats import fix_random_seed
from .base import Experiment


# ---------------------------------------------------------------------------
# Debug plot helpers
# ---------------------------------------------------------------------------

def _plot_debug_metrics(history_df: pd.DataFrame, save_path: str) -> None:
    """Plot per-epoch encoder/likelihood diagnostics collected in debug mode."""
    logvar_cols = [
        ("logvar_mean", "mean", "C0"),
        ("logvar_min",  "min",  "C1"),
        ("logvar_max",  "max",  "C2"),
    ]
    scalar_cols = [
        ("x_mean",      "x mean (encoder)"),
        ("x_std",       "x std (encoder)"),
        ("sigma_mean",  "likelihood sigma mean"),
        ("B_grad_norm", "B gradient norm"),
    ]

    _, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs_flat = axs.flatten()

    # Subplot 0: combined logvar
    ax = axs_flat[0]
    any_lv = False
    for col, label, color in logvar_cols:
        if col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[col], label=label, color=color)
            any_lv = True
    if any_lv:
        ax.legend(fontsize=8)
    ax.set_title("logvar (clamped)")
    ax.set_xlabel("epoch")
    ax.grid(True)

    # Subplots 1–3: individual scalar diagnostics
    for i, (col, title) in enumerate(scalar_cols, start=1):
        ax = axs_flat[i]
        if col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[col])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True)

    for j in range(len(scalar_cols) + 1, len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.suptitle("Debug diagnostics")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _plot_B_evolution(
    b_snapshots: list,
    snapshot_epochs: list,
    B_threshold: float,
    save_path: str,
) -> None:
    """Plot the support |B| > B_threshold over training, one (M x n_snapshots)
    binary heatmap per latent.
    """
    if not b_snapshots:
        return

    D, M = b_snapshots[0].shape
    n = len(b_snapshots)

    B_binary = np.zeros((D, M, n), dtype=np.float32)
    for t, B in enumerate(b_snapshots):
        B_binary[:, :, t] = (np.abs(B.numpy()) > B_threshold).astype(np.float32)

    row_h = min(1.5, max(0.6, M * 0.35)) # clamp for large D*M
    fig, axs = plt.subplots(
        D, 1,
        figsize=(max(8, n * 0.4 + 3), D * row_h + 1.5),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    if D == 1:
        axs = [axs]

    step = max(1, n // 10)
    for d in range(D):
        ax = axs[d]
        ax.imshow(
            B_binary[d],  # (M, n)
            aspect="auto", cmap="binary", vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax.set_yticks(range(M))
        ax.set_yticklabels([f"u{m}" for m in range(M)], fontsize=6)
        ax.set_ylabel(f"lat {d}", fontsize=7, rotation=0, labelpad=24, va="center")

    axs[-1].set_xticks(range(0, n, step))
    axs[-1].set_xticklabels(
        [str(snapshot_epochs[i]) for i in range(0, n, step)], fontsize=7,
    )
    axs[-1].set_xlabel("epoch")

    fig.suptitle(f"B matrix evolution  (|B| > {B_threshold})", y=1.01, fontsize=10)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _plot_gamma_evolution(
    gamma_snapshots: list,
    snapshot_epochs: list,
    save_path: str,
) -> None:
    """Plot the per-latent leak rate gamma = softplus(theta) over training.

    gamma is the contraction rate in x + dt*(drift - gamma*x).
    """
    if not gamma_snapshots:
        return

    gammas = np.stack([g.numpy() for g in gamma_snapshots], axis=0)  # (n, D)
    D = gammas.shape[1]
    cmap = get_cmap(D)

    ax = plt.subplots(figsize=(8, 5))[1]
    for d in range(D):
        ax.plot(snapshot_epochs, gammas[:, d],
                color=cmap(d / max(D - 1, 1)), label=f"latent {d}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("gamma = softplus(theta)")
    ax.set_title("Per-latent leak rate evolution")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="best")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _pick_best_worst(pool: list, corr: np.ndarray) -> tuple[int, int]:
    """Return (best_idx, worst_idx) from *pool* by correlation value.

    NaN-valued trials are excluded; if all are NaN the full pool is used.
    """
    valid = [i for i in pool if not np.isnan(corr[i])]
    if not valid:
        valid = pool
    sorted_pool = sorted(valid, key=lambda i: corr[i])
    return sorted_pool[-1], sorted_pool[0]  # (best=highest corr, worst=lowest)


# ---------------------------------------------------------------------------
# Experiment base class
# ---------------------------------------------------------------------------

class TrainRealDataISSM(Experiment):
    """Abstract base: train an iSSM on real neural recordings.

    Subclasses implement check_config() and _load_data() -> (y, u, strata).
    """

    # Zero-fill masked y cells so the encoder never sees them. Only for datasets
    # whose eval rebuilds the same mask (else train/eval inputs differ).
    zero_fill_masked_obs = False

    @abstractmethod
    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Load neural activity and stimulation inputs.

        Returns:
            y:      (total_trials, T, N) float32 tensor.
            u:      (total_trials, T, M) float32 tensor.
            strata: (total_trials,) int32 array of stratum labels for
                    stratified splitting.
        """
        ...

    def _on_data_loaded_verbose(self) -> None:
        """Hook for extra verbose output after the data summary line."""
        pass

    def _get_site_features(self) -> torch.Tensor | None:
        """(M, n_features) per-stim-site features for a b_net (row k = stim
        channel k of u), or None. Called after _load_data().
        """
        return None

    def _select_trials_to_plot(
        self,
        corr_per_trial: np.ndarray,
        u_val: torch.Tensor,
        _val_strata: np.ndarray | None,
    ) -> tuple[list[int], list[int]]:
        """Pick the two best and two worst trials by correlation, preferring stim
        trials. Returns ([best, 2nd_best], [worst, 2nd_worst]).
        """
        is_stim = (u_val.abs().sum(dim=(1, 2)) > 0).numpy()
        pool = np.where(is_stim)[0].tolist()
        if len(pool) < 2:
            pool = list(range(len(corr_per_trial)))

        best, worst = _pick_best_worst(pool, corr_per_trial)

        # Second-best / second-worst from the remaining candidates
        pool_without_best  = [i for i in pool if i != best]
        pool_without_worst = [i for i in pool if i != worst]
        best2  = _pick_best_worst(pool_without_best,  corr_per_trial)[0] if pool_without_best  else best
        worst2 = _pick_best_worst(pool_without_worst, corr_per_trial)[1] if pool_without_worst else worst

        return [best, best2], [worst, worst2]

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        img_to_log = []
        fix_random_seed(self.seed)
        s = self.fetch_settings()
        components = self.fetch_components()
        debug = getattr(s, "debug", False)

        # Load real data --------------------------------------------------------
        y, u, strata = self._load_data()

        # b_net shares B across stim sites, so per-site strata aren't needed:
        # collapse to one group so the split honours n_val/n_test exactly.
        if "b_net" in components:
            strata = np.zeros_like(strata)

        # Split seed is fixed so every training seed gets the same val/test trials.
        # OOD holdout trials (_force_test) always go to test.
        _SPLIT_SEED = 0
        _train_mask = getattr(self, '_artefact_mask', None)
        _force_test = getattr(self, '_force_test_idx', None)
        train_loader, val_loader, _, _split_indices = get_dataloaders_real(
            y, u, strata,
            batch_size=s.batch_size,
            n_val_samples=s.n_val_samples,
            n_test_samples=s.n_test_samples,
            seed=_SPLIT_SEED,
            mask=_train_mask,
            force_test_idx=_force_test,
        )
        _splits_to_save = {k: [int(i) for i in v] for k, v in _split_indices.items()}
        _splits_to_save["pool_source"] = "train"
        if _force_test is not None:
            _splits_to_save["ood_test_indices"] = [int(i) for i in _force_test]
        # Lets eval verify it rebuilds the same trial pool.
        _pool_ids = getattr(self, "_pool_trial_ids", None)
        if _pool_ids is not None:
            _splits_to_save["pool_trial_ids"] = [int(i) for i in _pool_ids]
        _zero_fill = self.zero_fill_masked_obs and _train_mask is not None
        _splits_to_save["zero_fill_masked_obs"] = bool(_zero_fill)
        _val_idx = _split_indices["val_indices"]
        _val_strata = strata[np.array(_val_idx)] if _val_idx else None

        with open(os.path.join(self.output_dir, "splits.json"), "w") as _f:
            json.dump(_splits_to_save, _f)

        # Val artefact mask: the same (t, ch) cells as masked during training.
        _val_mask = val_loader.dataset.mask if val_loader is not None else None

        if self.verbose:
            n_val_actual = len(_splits_to_save["val_indices"])
            n_test_actual = len(_splits_to_save["test_indices"])
            n_train_actual = len(_splits_to_save["train_indices"])
            val_str = f"{n_val_actual} val" if n_val_actual > 0 else "no val"
            print(
                f"Loaded {len(strata)} trials total "
                f"({n_train_actual} train / {val_str} / {n_test_actual} test held out)"
            )
            self._on_data_loaded_verbose()

        # Optionally standardize y (stats over observed cells only) -------------
        y_mean, y_std = None, None
        _loaders = [train_loader] + ([val_loader] if val_loader is not None else [])
        if getattr(s, "standardize", False):
            y_mean, y_std = masked_channel_stats(train_loader.dataset.y,
                                                 train_loader.dataset.mask)
            for ldr in _loaders:
                ldr.dataset.y = (ldr.dataset.y - y_mean) / y_std

        # Masked cells never reach the encoder (the likelihood already ignores them).
        if _zero_fill:
            for ldr in _loaders:
                ldr.dataset.y = ldr.dataset.y.masked_fill(ldr.dataset.mask, 0.0)

        # Build models ----------------------------------------------------------
        dynamics = components.dynamics.build()
        emission = components.emission.build()
        likelihood = components.likelihood.build()
        joint = FlexibleLDS(dynamics, emission, likelihood)
        recognition = components.recognition.build()

        # Optional b_net: computes B from per-site features
        b_net = None
        if "b_net" in components:
            b_net = components.b_net.build()
            site_features = self._get_site_features()
            if site_features is None:
                raise ValueError(
                    "A 'b_net' component is configured but this experiment provides "
                    "no site features (_get_site_features returned None).")
            b_net.set_site_features(site_features)
            dynamics.set_B_net(b_net)

        # Train -----------------------------------------------------------------
        optimizer = get_optimizer(
            models=[joint, recognition] + ([b_net] if b_net is not None else []),
            learning_rate=s.lr,
            weight_decay=s.weight_decay,
        )
        history_df, early_stopper, b_snapshots, gamma_snapshots = run_training_loop(
            joint, recognition, optimizer,
            train_loader, val_loader,
            s, verbose=self.verbose, debug=debug,
        )

        # Save standardization stats (used by eval) ----------------------------
        if y_mean is not None and y_std is not None:
            np.save(os.path.join(self.output_dir, "y_mean.npy"), y_mean.numpy())
            np.save(os.path.join(self.output_dir, "y_std.npy"),  y_std.numpy())

        # Save weights ----------------------------------------------------------
        if s.save_weights:
            torch.save(dynamics.state_dict(),
                       os.path.join(self.output_dir, "dynamics.pth"))
            torch.save(emission.state_dict(),
                       os.path.join(self.output_dir, "emission.pth"))
            torch.save(likelihood.state_dict(),
                       os.path.join(self.output_dir, "likelihood.pth"))
            torch.save(recognition.state_dict(),
                       os.path.join(self.output_dir, "recognition.pth"))
            if b_net is not None:
                torch.save(b_net.state_dict(),
                           os.path.join(self.output_dir, "b_net.pth"))

        # Plot loss curves ------------------------------------------------------
        best_epoch = early_stopper.best_epoch if early_stopper is not None else None
        history_df.to_csv(os.path.join(self.output_dir, "history.csv"), index=False)
        save_path = os.path.join(self.output_dir, "history.png")
        plot_training_metrics(history_df, save_path=save_path, best_epoch=best_epoch)
        img_to_log.append(save_path)

        # Debug plots -----------------------------------------------------------
        if debug:
            debug_metrics_path = os.path.join(self.output_dir, "debug_metrics.png")
            _plot_debug_metrics(history_df, save_path=debug_metrics_path)
            img_to_log.append(debug_metrics_path)

        if b_snapshots:
            snapshot_epochs = list(history_df["epoch"].iloc[:len(b_snapshots)])
            b_evo_path = os.path.join(self.output_dir, "B_evolution.png")
            _plot_B_evolution(
                b_snapshots,
                snapshot_epochs,
                B_threshold=joint.dynamics.B_threshold,
                save_path=b_evo_path,
            )
            img_to_log.append(b_evo_path)

        if gamma_snapshots:
            snapshot_epochs = list(history_df["epoch"].iloc[:len(gamma_snapshots)])
            gamma_evo_path = os.path.join(self.output_dir, "gamma_evolution.png")
            _plot_gamma_evolution(gamma_snapshots, snapshot_epochs, save_path=gamma_evo_path)
            img_to_log.append(gamma_evo_path)

        joint.eval()
        recognition.eval()

        if val_loader is not None:
            # Plot example predictions on the validation set --------------------
            val_data = val_loader.dataset
            u_val = val_data.u
            y_val = val_data.y
            _n_stim = getattr(joint.dynamics, 'n_stim', None)
            u_val_stim = u_val[..., :_n_stim]

            with torch.no_grad():
                recognition.set_B_from_dynamics(joint.dynamics)
                x_encoded_val = recognition.encode_mean(y_val, u_val)
                y_pred_val = joint.likelihood.mean(joint.emission(x_encoded_val))
                _, y_pred_rollout_val = joint.mean(u=u_val, x0=x_encoded_val[:, 0])
                y_rollout_true = y_val
                u_rollout_eval = u_val
            u_rollout_eval_stim = u_rollout_eval[..., :_n_stim]
            _val_mask_rollout = _val_mask

            def _mrow(m, i):  # None-safe per-trial mask slice
                return m[i] if m is not None else None

            # Reverse standardization before metrics and plots
            if y_mean is not None and y_std is not None:
                y_val = y_val * y_std + y_mean
                y_pred_val = y_pred_val * y_std + y_mean
                y_pred_rollout_val = y_pred_rollout_val * y_std + y_mean
                y_rollout_true = y_rollout_true * y_std + y_mean

            # Select trial pairs based on per-trial rollout correlation ----------
            _N_DIMS = 20
            _PER_PANEL = 10
            rollout_corr = compute_corr_per_trial(y_rollout_true, y_pred_rollout_val, mask=_val_mask_rollout)
            y_corr = compute_corr_per_trial(y_val, y_pred_val, mask=_val_mask)
            best_pair, worst_pair = self._select_trials_to_plot(
                rollout_corr, u_val_stim, _val_strata,
            )

            def _corr_str(corr, idx1, idx2):
                c1 = f"{corr[idx1]:.2f}" if not np.isnan(corr[idx1]) else "nan"
                c2 = f"{corr[idx2]:.2f}" if not np.isnan(corr[idx2]) else "nan"
                return f"[{c1}, {c2}]"

            # Val-set position -> splits.json trial index, for plot labels
            def _tid(pos):
                return _val_idx[pos] if (_val_idx is not None and pos < len(_val_idx)) else None

            # Rollout prediction plots (best / worst) --------------------------
            for pair, suffix in [(best_pair, "best"), (worst_pair, "worst")]:
                path = os.path.join(self.output_dir, f"y_val_rollout_predictions_{suffix}.png")
                plot_y_predictions_pair(
                    u_rollout_eval_stim, y_rollout_true, y_pred_rollout_val,
                    idx1=pair[0], idx2=pair[1],
                    trial_id1=_tid(pair[0]), trial_id2=_tid(pair[1]),
                    n_dims_per_trial=_N_DIMS, max_per_panel=_PER_PANEL,
                    pred_label="Rollout",
                    title=f"Rollout predictions ({suffix})  corr={_corr_str(rollout_corr, pair[0], pair[1])}",
                    save_path=path,
                    mask1=_mrow(_val_mask_rollout, pair[0]),
                    mask2=_mrow(_val_mask_rollout, pair[1]),
                )
                img_to_log.append(path)

            # Encoder reconstruction plots (same pairs) ------------------------
            for pair, suffix in [(best_pair, "best"), (worst_pair, "worst")]:
                path = os.path.join(self.output_dir, f"y_val_predictions_{suffix}.png")
                plot_y_predictions_pair(
                    u_val_stim, y_val, y_pred_val,
                    idx1=pair[0], idx2=pair[1],
                    trial_id1=_tid(pair[0]), trial_id2=_tid(pair[1]),
                    n_dims_per_trial=_N_DIMS, max_per_panel=_PER_PANEL,
                    pred_label="Smoothed",
                    title=f"Reconstructions ({suffix})  corr={_corr_str(y_corr, pair[0], pair[1])}",
                    save_path=path,
                    mask1=_mrow(_val_mask, pair[0]),
                    mask2=_mrow(_val_mask, pair[1]),
                )
                img_to_log.append(path)

            # One-step latent predictions (dynamics model, teacher-forced by encoder)
            with torch.no_grad():
                x_onestep_val = joint.x_onestep(x_encoded_val, u_val)
            _X_DIMS = min(40, x_encoded_val.shape[2])
            _X_PER_PANEL = 10
            x_inputs_val = joint.dynamics.get_inputs(u_val)
            onestep_path = os.path.join(self.output_dir, "x_val_onestep.png")
            plot_x_pred_only(
                x_encoded_val[:, :, :_X_DIMS],
                x_inputs_val[:, :, :_X_DIMS],
                x_dyn=x_onestep_val[:, :, :_X_DIMS],
                save_path=onestep_path,
                plot_together=True,
                max_per_panel=_X_PER_PANEL,
                trial_idx=best_pair[0],
            )
            img_to_log.append(onestep_path)

            # Compute and save performance metrics ------------------------------
            val_x_corr_onestep, _ = compute_mean_corr(x_encoded_val, x_onestep_val, u=u_val_stim)
            val_x_mae_onestep,  _ = compute_mean_mae( x_encoded_val, x_onestep_val, u=u_val_stim)
            performance = {
                "val_y_corr": compute_mean_corr(y_val, y_pred_val, u=u_val_stim, mask=_val_mask)[0],
                "val_y_corr_rollout": compute_mean_corr(y_rollout_true, y_pred_rollout_val, u=u_rollout_eval_stim, mask=_val_mask_rollout)[0],
                "val_y_mae": compute_mean_mae( y_val, y_pred_val, u=u_val_stim, mask=_val_mask)[0],
                "val_x_corr_onestep": val_x_corr_onestep,
                "val_x_mae_onestep": val_x_mae_onestep,
                "best_epoch": best_epoch,
            }

            if debug and joint.dynamics.train_B:
                B = joint.dynamics.B_effective.detach().cpu()
                B_thresh = joint.dynamics.B_threshold
                B_np = B.numpy()
                above = np.abs(B_np) > B_thresh
                performance["B_sparsity"]         = float((~above).mean())
                performance["B_nonzero_latents"]  = int(above.any(axis=1).sum())
                performance["B_frob_norm"]        = float(np.linalg.norm(B_np, "fro"))

                # Fraction of latents intervened on per stimulation timestep
                with torch.no_grad():
                    x_inputs = joint.dynamics.get_inputs(u_val)   # (trials, T, D)
                    stim_mask = (u_val_stim != 0).any(dim=-1)     # (trials, T)
                if stim_mask.any():
                    driven = (x_inputs.abs() > 0)[stim_mask]      # (n_stim_steps, D)
                    performance["B_frac_latents_intervened"] = float(driven.float().mean().item())

            if wandb is not None and wandb.run is not None:
                wandb.run.summary.update(performance)
            else:
                pd.DataFrame([performance]).to_csv(
                    os.path.join(self.output_dir, "performance.csv"), index=False
                )

        # Log collected images, downscaled to wandb's pixel limit
        if wandb is not None and wandb.run is not None:
            _WANDB_MAX_PX = 178_956_970
            from PIL import Image as _PILImage
            _PILImage.MAX_IMAGE_PIXELS = None # disable PIL's decompression-bomb guard
            for img_path in img_to_log:
                img = _PILImage.open(img_path)
                w, h = img.size
                if w * h > _WANDB_MAX_PX:
                    scale = (_WANDB_MAX_PX / (w * h)) ** 0.5
                    img = img.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)
                wandb.log({img_path: wandb.Image(img)})
