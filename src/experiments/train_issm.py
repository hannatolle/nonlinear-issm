"""
Train an iSSM on data simulated from a toy system.
"""

import os
import torch
try:
    import wandb
except ImportError:
    wandb = None
import numpy as np
import pandas as pd

from src.models import FlexibleLDS
from src.datasets import get_dataloaders
from src.training import get_optimizer, run_training_loop
from src.utils.plotting import plot_y_predictions, plot_x_predictions, plot_training_metrics
from src.utils.stats import fix_random_seed
from src.utils.metrics import compute_mean_corr, compute_mean_mae, align_by_B, apply_perm_latents
from .base import Experiment


class TrainISSM(Experiment):
    """Train an iSSM on simulated data."""
    def check_config(self):
        """Check required components and consistency of resolved settings."""
        required_components = ["system", "stimulator", "dynamics",
                               "emission", "likelihood", "recognition"]
        for comp in required_components:
            if comp not in self.components:
                raise ValueError(f"TrainISSM requires component '{comp}'.")
        # Allow system.D to differ from the global D (e.g. VanDerPol requires D=2
        # regardless of the latent dimension used by the model components).
        skip = {}
        system_D = self.components['system'].params.get('D')
        settings_D = self.settings.get('D')
        if system_D is not None and settings_D is not None and system_D != settings_D:
            print(f"Warning: system.D={system_D} differs from settings.D={settings_D}. "
                  f"Model components will use D={settings_D}.")
            skip = {'system': {'D'}}
        self.match_settings(strict=True, skip=skip)
        s = self.fetch_settings()
        if s.n_samples <= 0:
            raise ValueError("n_samples must be > 0.")
        if s.n_val_samples + s.n_test_samples >= s.n_samples:
            raise ValueError(
                f"n_val_samples ({s.n_val_samples}) + n_test_samples ({s.n_test_samples}) "
                f"must be less than n_samples ({s.n_samples}).")
        if s.n_test_samples > 0 and not s.early_stopping:
            raise ValueError(
                "n_test_samples > 0 is only meaningful when early_stopping=True "
                "(test set is reserved for post-training evaluation). "
                "Set n_test_samples=0 or enable early_stopping.")

    @torch.no_grad()
    def _generate_data(self, system, stimulator):
        """Simulate trials and discard the burn-in; returns u, x, y as (n, T, *)."""
        s = self.fetch_settings()
        n_samples = s.n_samples
        T_total = s.T + s.T_burn
        u = stimulator.generate(n_samples, T_total, s.dt) # (n_samples, T_total, M)
        x0 = system.rand_state(n_samples)
        _, x_ = system.run(T=T_total, dt=s.dt, x0=x0, u=u)
        y_ = system.obs(x_) # (T_total, n_samples, N)
        steps_to_discard = int(s.T_burn / s.dt)
        y = y_[steps_to_discard:].permute(1, 0, 2)  # (n_samples, T, N)
        x = x_[steps_to_discard:].permute(1, 0, 2)  # (n_samples, T, D)
        u = u[:,steps_to_discard:,:]
        return u, x, y

    def run(self):
        # Preliminaries -------------------------------------------------------------
        os.makedirs(self.output_dir, exist_ok=True)
        img_to_log = []  # image paths to log to wandb
        fix_random_seed(self.seed)
        s = self.fetch_settings()      
        components = self.fetch_components()

        # Simulate inputs and dynamics ----------------------------------------------
        system = components.system.build()
        stimulator = components.stimulator.build()
        if stimulator.seed is None:
            stimulator.seed = self.seed
        u, x, y = self._generate_data(system, stimulator)

        # Create datasets and dataloaders -------------------------------------------
        train_loader, val_loader, test_loader = get_dataloaders(x, y, u,
            batch_size=s.batch_size, n_val_samples=s.n_val_samples, n_test_samples=s.n_test_samples)

        # Optionally standardize y --------------------------------------------------
        y_mean, y_std = None, None
        if getattr(s, "standardize", False):
            y_mean = train_loader.dataset.y.mean(dim=(0, 1))  # (N,)
            y_std = train_loader.dataset.y.std(dim=(0, 1)).clamp(min=1e-8)
            for ldr in [train_loader, val_loader] + ([test_loader] if test_loader else []):
                ldr.dataset.y = (ldr.dataset.y - y_mean) / y_std

        # Set up the models ---------------------------------------------------------
        dynamics = components.dynamics.build()
        emission = components.emission.build()
        likelihood = components.likelihood.build()
        joint = FlexibleLDS(dynamics, emission, likelihood)
        recognition = components.recognition.build()

        # Train the model -----------------------------------------------------------
        optimizer = get_optimizer(models=[joint, recognition],
                                  learning_rate=s.lr,
                                  weight_decay=s.weight_decay)

        history_df, early_stopper, _, _ = run_training_loop(
            joint, recognition, optimizer,
            train_loader, val_loader,
            s, verbose=self.verbose, compute_encoder_mae=False)

        # Save all models -----------------------------------------------------------
        if s.save_weights:
            torch.save(dynamics.state_dict(), os.path.join(self.output_dir, "dynamics.pth"))
            torch.save(emission.state_dict(), os.path.join(self.output_dir, "emission.pth"))
            torch.save(likelihood.state_dict(), os.path.join(self.output_dir, "likelihood.pth"))
            torch.save(recognition.state_dict(), os.path.join(self.output_dir, "recognition.pth"))

        # Plot the results ----------------------------------------------------------
        # 1. Loss curves
        best_epoch = early_stopper.best_epoch if early_stopper is not None else None
        history_df.to_csv(os.path.join(self.output_dir, "history.csv"), index=False)
        save_path = os.path.join(self.output_dir, "history.png")
        plot_training_metrics(history_df, save_path=save_path, best_epoch=best_epoch);
        img_to_log.append(save_path)

        # 2. Learned latent state equations
        equations = dynamics.describe()
        true_equations = system.describe()
        equations_list = equations if isinstance(equations, list) else [equations]
        true_equations_list = true_equations if isinstance(true_equations, list) else [true_equations]
        if self.verbose:
            print("\nLearned Latent State Equations:")
            for eq in equations_list:
                print(eq)
            print("\nGround-truth Latent State Equations:")
            for eq in true_equations_list:
                print(eq)

        file_path = os.path.join(self.output_dir, "latent_state_equations.txt")
        with open(file_path, "w") as f:
            f.write("Learned Latent State Equations:\n")
            for eq in equations_list:
                f.write(eq + "\n")
            f.write("\nGround-truth Latent State Equations:\n")
            for eq in true_equations_list:
                f.write(eq + "\n")
        if wandb is not None and wandb.run is not None:
            textblob = "learned:\n" + "\n".join(equations_list) + "\n\ntrue:\n" + "\n".join(true_equations_list)
            wandb.log({"latent_state_equations": wandb.Html(f"<pre>{textblob}</pre>")})

        # 3. Plot example test predictions of the final model
        eval_loader = test_loader if test_loader is not None else val_loader
        eval_data = eval_loader.dataset
        u_val, x_val, y_val = eval_data.u, eval_data.x, eval_data.y

        joint.eval()
        recognition.eval()
        with torch.no_grad():
            recognition.set_B_from_dynamics(joint.dynamics)
            x_encoded_val = recognition.encode_mean(y_val, u_val)
            x_pred_val, y_pred_rollout_val = joint.mean(u=u_val, x0=x_encoded_val[:, 0])
            y_pred_enc_val = joint.likelihood.mean(joint.emission(x_encoded_val))

        # Reverse standardization for y before plotting and metrics (if we standardized)
        if y_mean is not None and y_std is not None:
            y_val = y_val * y_std + y_mean
            y_pred_rollout_val = y_pred_rollout_val * y_std + y_mean
            y_pred_enc_val = y_pred_enc_val * y_std + y_mean

        plot_y_predictions(u_val, y_val, y_pred_rollout_val, save_path=os.path.join(self.output_dir, "y_val_predictions.png"))
        img_to_log.append(os.path.join(self.output_dir, "y_val_predictions.png"))

        # Latent-space alignment and comparison are only meaningful when system.D == dynamics.D
        latent_dims_match = (x_val.shape[-1] == x_pred_val.shape[-1])

        if latent_dims_match:
            # Align latent indices: permute x_pred_val to match true latent order
            B_true_np = system.B.detach().cpu().numpy()
            B_learned_np = joint.dynamics.B.detach().cpu().numpy()
            perm = align_by_B(B_true_np, B_learned_np, use_abs=True) if joint.dynamics.train_B else np.arange(system.D)
            x_pred_val_p_aligned = apply_perm_latents(x_pred_val, perm)

            # Align sign: flip latents whose mean correlation with the true latent is negative
            x_pred_np = x_pred_val_p_aligned.detach().cpu().numpy()  # (B, T, D)
            x_true_np = x_val.detach().cpu().numpy()                 # (B, T, D)
            signs = np.ones(dynamics.D)
            for di in range(dynamics.D):
                corrs = [np.corrcoef(x_pred_np[b, :, di], x_true_np[b, :, di])[0, 1]
                         for b in range(x_pred_np.shape[0])
                         if np.std(x_pred_np[b, :, di]) > 0 and np.std(x_true_np[b, :, di]) > 0]
                if corrs and np.mean(corrs) < 0:
                    signs[di] = -1.0
            sign_tensor = torch.tensor(signs, dtype=x_pred_val_p_aligned.dtype,
                                       device=x_pred_val_p_aligned.device)
            x_pred_val_p_aligned = x_pred_val_p_aligned * sign_tensor

            x_inputs_val = joint.dynamics.get_inputs(u_val)
            plot_x_predictions(x_val, x_pred_val_p_aligned, x_inputs_val, save_path=os.path.join(self.output_dir, "x_val_predictions.png"))
            img_to_log.append(os.path.join(self.output_dir, "x_val_predictions.png"))
            val_x_corr = compute_mean_corr(x_val, x_pred_val_p_aligned, u=u_val, absolute=True)[0]
        else:
            val_x_corr = float('nan')

        # 4. Compute & log final performance
        performance = {'val_y_corr': compute_mean_corr(y_val, y_pred_enc_val, u=u_val)[0],
                       'val_y_corr_rollout': compute_mean_corr(y_val, y_pred_rollout_val, u=u_val)[0],
                       'val_y_mae': compute_mean_mae( y_val, y_pred_enc_val, u=u_val)[0],
                       'val_y_mae_rollout': compute_mean_mae( y_val, y_pred_rollout_val, u=u_val)[0],
                       'val_x_corr': val_x_corr,
                       'best_epoch': best_epoch}

        if wandb is not None and wandb.run is not None:
            wandb.run.summary.update(performance)
        else:
            # Save as dataframe to output directory
            performance_df = pd.DataFrame([performance])
            performance_df.to_csv(os.path.join(self.output_dir, "performance.csv"), index=False)

        # Log all images to wandb, if enabled
        if wandb is not None and wandb.run is not None:
            for img_path in img_to_log:
                wandb.log({img_path: wandb.Image(img_path)})
