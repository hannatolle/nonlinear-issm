"""
beta-VAE ELBO loss and the per-epoch train / test loops.
"""

import math
import torch


# Training ----------------------------------------------------------------------

def get_beta(step: int, beta: float, schedule: str = "constant", warmup_steps: int = 0) -> float:
    """Return the current KL weight given the schedule.

    Args:
        step: Current step or epoch index (1-based).
        beta: Target (maximum) beta value.
        schedule: One of 'constant', 'linear', 'cyclical'.
        warmup_steps: Number of steps/epochs over which to anneal (ignored for 'constant').
    """
    if schedule == "constant" or warmup_steps <= 0:
        return beta
    elif schedule == "linear":
        return float(min(beta * step / warmup_steps, beta))
    elif schedule == "cyclical":
        t = step % warmup_steps
        half = warmup_steps / 2
        if t < half:
            return float(beta * t / half)
        else:
            return float(beta)
    else:
        raise ValueError(f"Unknown beta_schedule: {schedule!r}. Use 'constant', 'linear', or 'cyclical'.")


def elbo_loss(model, recognition, y, u, beta=1.0, mask=None):
    """beta-VAE ELBO: E_q[log p(y|x) + sparsity] - beta * KL[q(x|y,u) || p(x|u)].

    mask: optional (B, T, N) bool tensor, True = unobserved; the likelihood
    excludes masked entries.
    """
    x_sample, log_q = recognition(y, u)  # log_q: (B, T)
    log_q_total = log_q.sum(dim=1)       # (B,)

    prior_log_prob, log_likelihood, sparsity_reg = model.log_prob(x_sample, y, u, mask=mask)

    kl = log_q_total - prior_log_prob  # (B,)
    sparsity_val = sparsity_reg if sparsity_reg.dim() == 0 else sparsity_reg.mean()
    elbo = (log_likelihood + sparsity_val - beta * kl).mean()
    return -elbo, kl.mean(), log_q_total.mean(), prior_log_prob.mean(), log_likelihood.mean(), sparsity_val


def _batch_mask(third):
    """Extract the artefact mask from a loader batch's third element, or None.

    Slot 3 is overloaded: real-data loaders put a (B, T, N) bool artefact mask
    there (True = ignore), SimDynamicsDataset puts (B, T, D) float latents, and
    mask-less real loaders put a (B, 1) dummy. Only a 3-D bool tensor is a mask.
    """
    return third if (torch.is_tensor(third) and third.dim() == 3
                     and third.dtype == torch.bool) else None


def train(joint, recognition, loader, optimizer, beta=1.0, grad_clip=None,
          track_B_grad: bool = False):
    joint.train()
    recognition.train()

    total_loss = total_kl = total_log_q = 0.0
    total_prior = total_ll = total_sparsity = 0.0
    total_B_grad_norm = 0.0
    n_samples = len(loader.dataset)
    n_batches = 0

    for y_batch, u_batch, m_batch in loader:
        optimizer.zero_grad()
        recognition.set_B_from_dynamics(joint.dynamics)
        mask_batch = _batch_mask(m_batch)
        loss, kl, log_q, prior, ll, sparsity = elbo_loss(
            joint, recognition, y_batch, u_batch, beta, mask=mask_batch
        )
        total = loss
        total.backward()

        _B_param = getattr(joint.dynamics, 'B', None)
        if track_B_grad and joint.dynamics.train_B and _B_param is not None and _B_param.grad is not None:
            total_B_grad_norm += _B_param.grad.norm().item()

        if grad_clip is not None:
            all_params = list(joint.parameters()) + list(recognition.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        b = y_batch.size(0)
        total_loss += loss.item() * b
        total_kl += kl.item() * b
        total_log_q += log_q.item() * b
        total_prior += prior.item() * b
        total_ll += ll.item() * b
        total_sparsity += (sparsity.item() if sparsity.dim() == 0 else sparsity.mean().item()) * b
        n_batches += 1

    n = n_samples
    B_grad_norm = total_B_grad_norm / max(1, n_batches) if track_B_grad else 0.0
    return (total_loss / n, total_kl / n, total_log_q / n,
            total_prior / n, total_ll / n, total_sparsity / n, B_grad_norm)

# Evaluation --------------------------------------------------------------------

def _safe_div(num: float, denom: int) -> float:
    """Return num / denom if denom > 0 and num is finite, else float('nan')."""
    if denom <= 0 or not math.isfinite(num):
        return float("nan")
    return num / denom


@torch.no_grad()
def test(joint, recognition, loader, beta=1.0, compute_encoder_mae: bool = True):
    joint.eval()
    recognition.eval()
    _n_stim = getattr(joint.dynamics, 'n_stim', None)

    total_loss = total_kl = total_log_q = 0.0
    total_prior = total_ll = total_sparsity = 0.0
    n_loss = n_kl = n_log_q = 0
    n_prior = n_ll = n_sparsity = 0
    total_y_mae_rollout = 0.0
    n_y_mae_rollout = 0
    total_y_mae = 0.0
    n_y_mae = 0
    n_samples = len(loader.dataset)

    for y_batch, u_batch, m_batch in loader:
        recognition.set_B_from_dynamics(joint.dynamics)
        b = y_batch.size(0)
        artefact = _batch_mask(m_batch)  # (B, T, N) bool, True = ignore, or None

        # Only finite values are accumulated, so one bad batch does not poison metrics
        loss, kl, log_q, prior, ll, sparsity = elbo_loss(joint, recognition, y_batch, u_batch, beta, mask=artefact)
        loss_val = loss.item()
        kl_val = kl.item()
        log_q_val = log_q.item()
        prior_val = prior.item()
        ll_val = ll.item()
        sparsity_val = sparsity.item() if sparsity.dim() == 0 else sparsity.mean().item()
        if math.isfinite(loss_val):
            total_loss += loss_val * b
            n_loss += b
        if math.isfinite(kl_val):
            total_kl += kl_val * b
            n_kl += b
        if math.isfinite(log_q_val):
            total_log_q += log_q_val * b
            n_log_q += b
        if math.isfinite(prior_val):
            total_prior += prior_val * b
            n_prior += b
        if math.isfinite(ll_val):
            total_ll += ll_val * b
            n_ll += b
        if math.isfinite(sparsity_val):
            total_sparsity += sparsity_val * b
            n_sparsity += b

        x_encoded = recognition.encode_mean(y_batch, u_batch)
        _, y_pred_rollout = joint.mean(u=u_batch, x0=x_encoded[:, 0])

        # Score only non-stimulated timesteps (task-context channels are ignored)
        keep = (u_batch[..., :_n_stim] == 0).all(dim=2).unsqueeze(-1).expand(-1, -1, y_batch.size(-1))
        if artefact is not None:
            keep = keep & ~artefact

        masked_diff_rollout = torch.abs(y_pred_rollout - y_batch)[keep]
        if masked_diff_rollout.numel() > 0 and torch.isfinite(masked_diff_rollout).all():
            total_y_mae_rollout += masked_diff_rollout.mean().item() * b
            n_y_mae_rollout += b

        if compute_encoder_mae:
            y_pred_enc = joint.likelihood.mean(joint.emission(x_encoded))
            keep_enc = (u_batch[..., :_n_stim] == 0).all(dim=2).unsqueeze(-1).expand_as(y_batch)
            if artefact is not None:
                keep_enc = keep_enc & ~artefact
            masked_diff_enc = torch.abs(y_pred_enc - y_batch)[keep_enc]
            if masked_diff_enc.numel() > 0 and torch.isfinite(masked_diff_enc).all():
                total_y_mae += masked_diff_enc.mean().item() * b
                n_y_mae += b

    # Normalise by the number of finite samples per metric
    metrics = {
        "loss": _safe_div(total_loss, n_loss),
        "kl": _safe_div(total_kl, n_kl),
        "log_q": _safe_div(total_log_q, n_log_q),
        "prior_log_prob": _safe_div(total_prior, n_prior),
        "log_likelihood": _safe_div(total_ll, n_ll),
        "sparsity_reg": _safe_div(total_sparsity, n_sparsity),
        "y_mae_rollout": _safe_div(total_y_mae_rollout, n_y_mae_rollout),
        "y_mae": _safe_div(total_y_mae, n_y_mae),
    }
    return metrics
