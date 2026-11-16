"""Joint distribution p(x, y | u)."""

import torch
import torch.nn as nn


class FlexibleLDS(nn.Module):
    """p(x, y | u) = p(x | u) p(y | f_emission(x)).

    Args:
        dynamics: state transition model, x(t+1) = f_dynamics(x(t), u(t))
        emission: emission model, y_pred = f_emission(x)
        likelihood: likelihood model, p(y | f_emission(x))
    """
    def __init__(self, dynamics, emission, likelihood):
        super().__init__()
        self.dynamics = dynamics
        self.emission = emission
        self.likelihood = likelihood

    def log_prob(self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        prior_log_prob = self.dynamics.log_prob(x, u)          # (B,)
        y_pred = self.emission(x)                              # (B, T, N)
        log_likelihood = self.likelihood.log_prob(y_pred, y, mask=mask)
        sparsity_reg = self.dynamics.sparsity_priors()         # scalar
        return prior_log_prob, log_likelihood, sparsity_reg

    def sample(self, u: torch.Tensor, x0: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.dynamics.sample(u, x0)
        y = self.likelihood.sample(self.emission(x))
        return x, y
    
    def mean(self, u: torch.Tensor, x0: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.dynamics.mean(u, x0)
        y = self.likelihood.mean(self.emission(x))
        return x, y

    def x_onestep(self, x_enc: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """One-step predictions teacher-forced by encoder states.

        x_os[:, t] = dynamics.step(x_enc[:, t-1], u[:, t]) for t >= 1; x_os[:, 0] = x_enc[:, 0].
        x_enc: (B, T, D), u: (B, T, M) -> x_os: (B, T, D)
        """
        _, T, _ = x_enc.shape
        x_os = x_enc.clone()
        for t in range(1, T):
            x_os[:, t, :] = self.dynamics.step(x_enc[:, t - 1, :], u[:, t, :])
        return x_os
