"""Amortized recognition networks q(x | y, u)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch.distributions import Normal
from typing import Optional, Tuple

from ..simulations import RateModel


# -----------------------------------------------------------------------------
# Base recognition network
# -----------------------------------------------------------------------------

class RecognitionNetwork(nn.Module, ABC):
    """
    Abstract base for amortized variational encoders q(x | y, u).
    Subclasses implement _encode(y, u) -> (mu, logvar); this class handles
    logvar clamping, intervention masking, reparameterization, and log_q.
    """

    def __init__(self,
                 D: int,
                 N: int,
                 M: int,
                 *,
                 interventional: bool = True,   # set the posterior mean to Bu at driven latents
                 B_threshold: float = 1e-4,     # |B| below this does not count as driven
                 min_logstd: float = -5.0,      # lower bound of the posterior log std
                 max_logstd: float = 5.0,       # upper bound of the posterior log std
                 obs_mask_prob: float = 0.0):   # training only: zero each entry of y with this prob.
        super().__init__()
        self.D, self.N, self.M = D, N, M
        self.interventional = interventional
        self.B_threshold = B_threshold
        self._logvar_min = 2.0 * min_logstd
        self._logvar_max = 2.0 * max_logstd
        self.obs_mask_prob = obs_mask_prob
        self.B: Optional[torch.Tensor] = None
        self.n_task_channels: int = 0

    def set_B_from_dynamics(self, dynamics: RateModel) -> None:
        self.B = dynamics.B_effective
        self.n_task_channels = getattr(dynamics, "n_task_channels", 0)

    @abstractmethod
    def _encode(self, y: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) both (B, T, D). logvar may be unclamped."""
        ...

    def _apply_B_mask(self, mu: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Replace mu with Bu at driven latents (hard intervention).

        Only the stim channels of u (the first B.shape[-1]) are used; task channels are ignored.
        """
        u_stim = u[..., :self.B.shape[-1]]
        Bu = torch.einsum("bti,di->btd", u_stim, self.B)
        if self.interventional:
            with torch.no_grad():
                B_clean = torch.where(self.B.abs() < self.B_threshold,
                    torch.zeros_like(self.B), self.B)
                clean_Bu = torch.einsum("bti,di->btd", u_stim, B_clean)
                mask = (clean_Bu == 0).float()
            return mask * mu + (1 - mask) * Bu
        else:
            return mu + Bu

    def encode_mean(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """B-masked posterior mean without sampling, for deterministic evaluation.

        y: (B, T, N), u: (B, T, M) -> mu: (B, T, D)
        """
        assert self.B is not None, "Call set_B_from_dynamics() before encode_mean()"
        mu, _ = self._encode(y, u)
        mu = self._apply_B_mask(mu, u)
        return mu

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """y: (B, T, N), u: (B, T, M) -> x_sample: (B, T, D), log_q: (B, T)"""
        assert self.B is not None, "Call set_B_from_dynamics() before forward()"

        # Training-only augmentation: zero each (trial, time, neuron) entry independently.
        if self.training and self.obs_mask_prob > 0.0:
            keep = torch.bernoulli(torch.full_like(y, 1.0 - self.obs_mask_prob))
            y = y * keep

        mu, logvar = self._encode(y, u)
        logvar = torch.clamp(logvar, min=self._logvar_min, max=self._logvar_max)
        std = torch.exp(0.5 * logvar)

        mu = self._apply_B_mask(mu, u)

        eps = torch.randn_like(std)
        x_sample = mu + eps * std

        log_q = Normal(mu, std).log_prob(x_sample).sum(-1) # sum over D; (B, T)
        return x_sample, log_q


# -----------------------------------------------------------------------------
# LSTM encoder
# -----------------------------------------------------------------------------


class SharedLSTM(RecognitionNetwork):
    """q(x | y, u) from a shared LSTM backbone with separate mu and logvar heads."""

    def __init__(self,
                 D: int,
                 N: int,
                 M: int,
                 H: int = 64,                   # hidden width of the input projection and LSTM
                 bidirectional: bool = False,   # if True, the posterior also sees future observations
                 interventional: bool = True,
                 B_threshold: float = 1e-4,
                 min_logstd: float = -5.0,
                 max_logstd: float = 5.0,
                 input_dropout: float = 0.0,    # training only: zero all of y at a time step with this prob.
                 obs_mask_prob: float = 0.0):
        super().__init__(D, N, M, interventional=interventional, B_threshold=B_threshold,
                         min_logstd=min_logstd, max_logstd=max_logstd, obs_mask_prob=obs_mask_prob)
        self.input_dropout = input_dropout
        input_dim = N + M
        self.in_proj = nn.Linear(input_dim, H)
        self.in_norm = nn.LayerNorm(H)

        self.lstm = nn.LSTM(input_size=H, hidden_size=H, batch_first=True,
                            bidirectional=bidirectional)
        lstm_out_dim = 2 * H if bidirectional else H

        # Output heads: LN -> projection -> GELU -> final
        self.norm_mu = nn.LayerNorm(lstm_out_dim)
        self.proj_mu = nn.Linear(lstm_out_dim, lstm_out_dim)
        self.to_mu = nn.Linear(lstm_out_dim, D)

        self.norm_logvar = nn.LayerNorm(lstm_out_dim)
        self.proj_logvar = nn.Linear(lstm_out_dim, lstm_out_dim)
        self.to_logvar = nn.Linear(lstm_out_dim, D)

        # Near-zero init so logvar starts near 0 (std near 1).
        nn.init.normal_(self.to_logvar.weight, std=1e-3)
        nn.init.zeros_(self.to_logvar.bias)

    def _encode(self, y: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.input_dropout > 0.0:
            keep = torch.rand(y.size(0), y.size(1), 1, device=y.device) > self.input_dropout
            y = y * keep
        data = torch.cat([y, u], dim=-1)                         # (B, T, N+M)
        data = F.gelu(self.in_norm(self.in_proj(data)))          # (B, T, H)
        h, _ = self.lstm(data)                                   # (B, T, H or 2H)
        mu = self.to_mu(F.gelu(self.proj_mu(self.norm_mu(h))))
        logvar = self.to_logvar(F.gelu(self.proj_logvar(self.norm_logvar(h))))
        return mu, logvar
