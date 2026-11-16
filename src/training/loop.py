"""
Shared training loop for iSSM experiments.
"""

from typing import Optional

import torch
import pandas as pd
try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm

from .early_stopping import EarlyStopper
from .elbo import get_beta, train, test

_TRAIN_EVAL_BATCHES = 1  # batches used for train-set metrics


class _BudgetLoader:
    """Wraps a DataLoader, yielding at most n_batches batches."""
    def __init__(self, loader, n_batches: int):
        self._loader = loader
        self._n = n_batches

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._n:
                break
            yield batch

    @property
    def dataset(self):
        return self._loader.dataset


@torch.no_grad()
def _collect_debug_stats(joint, recognition, y_probe: torch.Tensor, u_probe: torch.Tensor) -> dict:
    """Debug diagnostics on a fixed probe batch: encoder logvar range, sampled
    latent mean/std and (if available) the likelihood noise level.
    """
    was_joint_training = joint.training
    was_rec_training = recognition.training
    joint.eval()
    recognition.eval()

    recognition.set_B_from_dynamics(joint.dynamics)
    _, logvar_raw = recognition._encode(y_probe, u_probe)
    lv = logvar_raw.clamp(min=recognition._logvar_min, max=recognition._logvar_max)
    x_sample, _ = recognition(y_probe, u_probe)

    stats = {
        "logvar_mean": lv.mean().item(),
        "logvar_min":  lv.min().item(),
        "logvar_max":  lv.max().item(),
        "x_mean":      x_sample.mean().item(),
        "x_std":       x_sample.std().item(),
    }
    if hasattr(joint.likelihood, "log_diag"):
        stats["sigma_mean"] = joint.likelihood.log_diag.exp().mean().item()

    joint.train(was_joint_training)
    recognition.train(was_rec_training)
    return stats


def run_training_loop(
    joint,
    recognition,
    optimizer,
    train_loader,
    val_loader,
    s,
    verbose: bool = False,
    debug: bool = False,
    compute_encoder_mae: bool = True,
) -> tuple[pd.DataFrame, Optional[EarlyStopper], list]:
    """Run the ELBO training loop shared across iSSM experiments.

    Args:
        joint: FlexibleLDS model.
        recognition: Recognition network.
        optimizer: Torch optimizer (already constructed).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        s: Settings DotDict (n_epochs, beta*, tau_warmup_fraction, grad_clip, es_*).
        debug: If True, add per-interval diagnostics to history_df.

    Returns:
        history_df: one row per evaluation interval (train_*/val_* metrics, epoch, beta).
        early_stopper: EarlyStopper with best weights restored, or None.
        b_snapshots: (D, M) CPU copies of B_effective per logged interval
            (empty unless dynamics.train_B).
        gamma_snapshots: (D,) leak rates softplus(theta) per logged interval
            (empty unless debug and the dynamics have a learnable leak).
    """
    log_every = 10 if s.n_epochs // 10 >= 2 else 1
    history = []
    b_snapshots: list = []
    gamma_snapshots: list = []

    tau_warmup_steps = int(s.tau_warmup_fraction * s.n_epochs)
    beta_warmup_steps = int(s.beta_warmup_fraction * s.n_epochs)

    # Fixed probe batch for debug stats
    if debug:
        _probe_y, _probe_u, _ = next(iter(train_loader))

    # b_net is held outside joint's state_dict, so it is checkpointed separately.
    es_models = {"joint": joint, "recognition": recognition}
    if joint.dynamics.B_net is not None:
        es_models["b_net"] = joint.dynamics.B_net

    early_stopper: Optional[EarlyStopper] = None
    if s.early_stopping:
        early_stopper = EarlyStopper(
            patience=s.es_patience,
            metric=s.es_metric,
            min_delta=s.es_min_delta,
        )

    last_B_grad_norm = 0.0

    for epoch in tqdm(range(1, s.n_epochs + 1), desc="Epoch", unit="epoch", disable=not verbose):
        current_beta = get_beta(epoch, s.beta, s.beta_schedule, beta_warmup_steps)

        *_, last_B_grad_norm = train(
            joint, recognition, train_loader, optimizer,
            beta=current_beta, grad_clip=s.grad_clip,
            track_B_grad=debug,
        )

        frac = min(epoch / tau_warmup_steps, 1.0) if tau_warmup_steps > 0 else 1.0
        joint.dynamics.apply_soft_threshold(frac)

        if (epoch % log_every == 0) or (epoch == s.n_epochs):
            eval_train = (epoch % (2 * log_every) == 0) or (epoch == s.n_epochs)

            metrics_val = test(joint, recognition, val_loader, beta=current_beta,
                               compute_encoder_mae=compute_encoder_mae)
            metrics = {**{f"val_{k}": v for k, v in metrics_val.items()}}

            if eval_train:
                metrics_train = test(joint, recognition,
                                     _BudgetLoader(train_loader, _TRAIN_EVAL_BATCHES),
                                     beta=current_beta,
                                     compute_encoder_mae=compute_encoder_mae)
                metrics.update({f"train_{k}": v for k, v in metrics_train.items()})

            if joint.dynamics.train_B:
                b_snapshots.append(joint.dynamics.B_effective.detach().cpu().clone())
            if debug:
                metrics.update(_collect_debug_stats(joint, recognition, _probe_y, _probe_u))
                if joint.dynamics.train_B:
                    metrics["B_grad_norm"] = last_B_grad_norm
                if joint.dynamics.theta is not None:
                    gamma = torch.nn.functional.softplus(joint.dynamics.theta)
                    gamma_snapshots.append(gamma.detach().cpu().clone())

            if wandb is not None and wandb.run is not None:
                wandb.log(metrics)
            metrics["epoch"] = epoch
            metrics["beta"] = current_beta
            history.append(metrics)

            if early_stopper is not None:
                if early_stopper.step(metrics, es_models):
                    if verbose:
                        print(
                            f"\nEarly stopping at epoch {epoch} "
                            f"(best {s.es_metric}: {early_stopper.best_value:.6f})"
                        )
                    break

    if early_stopper is not None:
        early_stopper.restore(es_models)

    return pd.DataFrame(history), early_stopper, b_snapshots, gamma_snapshots
