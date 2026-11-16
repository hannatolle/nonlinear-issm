"""Emission models and observation likelihoods p(y | x)."""

import math

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Poisson


# -----------------------------------------------------------------------------
# Likelihood models
# -----------------------------------------------------------------------------

def _marginal_scale_tril(L: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Lower-triangular factor of Sigma_OO for Sigma = L L^T.

    L[obs][:, obs] is not a factor of Sigma_OO. Instead, Sigma_OO = A A^T with
    A = L[obs, :], and the QR of A^T gives A A^T = R^T R without forming Sigma.
    The diagonal of R^T may be negative; callers use |diag|.
    """
    return torch.linalg.qr(L[obs, :].mT, mode="reduced").R.mT


def _gaussian_log_prob_tril(diff: torch.Tensor, tril: torch.Tensor) -> torch.Tensor:
    """log N(diff; 0, tril tril^T) for diff (n, d); tril (d, d) lower-triangular."""
    z = torch.linalg.solve_triangular(tril, diff.unsqueeze(-1), upper=False).squeeze(-1)
    d = diff.shape[-1]
    return (-0.5 * (z ** 2).sum(-1)
            - torch.log(torch.diagonal(tril).abs()).sum()
            - 0.5 * d * math.log(2 * math.pi))


class NormalConditionalLikelihood(nn.Module):
    """Gaussian likelihood p(y | f_emission(x)) with Cholesky-parameterised covariance.

    diagonal_only drops the off-diagonal parameter (cheaper for large N).
    """
    def __init__(self, N: int,
                 y_init_sigma: float = 0.1,    # initial observation noise std
                 min_logstd: float = -5.0,     # lower bound of the noise log std
                 max_logstd: float = 5.0,      # upper bound of the noise log std
                 diagonal_only: bool = False):
        super().__init__()
        self.N = N
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.diagonal_only = diagonal_only
        self.log_diag = nn.Parameter(torch.log(torch.ones(N) * y_init_sigma))
        if not diagonal_only:
            self.off_diag = nn.Parameter(torch.zeros(N, N))

    @property
    def no_decay_params(self) -> set[str]:
        """Parameter names excluded from weight decay."""
        params = {"log_diag"}
        if not self.diagonal_only:
            params.add("off_diag")
        return params

    @property
    def scale_tril(self):
        L = torch.diag(torch.exp(torch.clamp(self.log_diag, min=self.min_logstd, max=self.max_logstd)))
        if not self.diagonal_only:
            L = L + torch.tril(self.off_diag, diagonal=-1)
        return L
        
    def log_prob(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        """Log-likelihood summed over time, shape (B,).

        y_pred, y_true: (B, T, N); mask: optional (B, T, N) bool, True = unobserved.
        At each (b, t) the contribution is the Gaussian marginal over the observed
        channels O, log N(y_O; mu_O, Sigma_OO); masked values are never used.
        """
        L = self.scale_tril
        if mask is None or not mask.any():
            return MultivariateNormal(y_pred, scale_tril=L).log_prob(y_true).sum(dim=1)

        # Masked targets are overwritten so non-finite values cannot reach autograd.
        y_true = y_true.masked_fill(mask, 0.0)
        B, T, N = y_pred.shape

        if self.diagonal_only:
            std = torch.diagonal(L)
            lp = -0.5 * ((y_true - y_pred) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)
            return lp.masked_fill(mask, 0.0).sum(dim=(1, 2))

        # Full covariance: one marginal factor per unique mask pattern.
        diff = (y_true - y_pred).reshape(B * T, N)
        patterns, inverse = torch.unique(mask.reshape(B * T, N).to(torch.uint8),
                                         dim=0, return_inverse=True)
        lp = diff.new_zeros(B * T)
        for k in range(patterns.shape[0]):
            obs = (patterns[k] == 0).nonzero(as_tuple=True)[0]
            if obs.numel() == 0:
                continue
            rows = (inverse == k).nonzero(as_tuple=True)[0]
            tril = L if obs.numel() == N else _marginal_scale_tril(L, obs)
            lp = lp.index_add(0, rows, _gaussian_log_prob_tril(diff[rows][:, obs], tril))
        return lp.view(B, T).sum(dim=1)

    def sample(self, y_pred: torch.Tensor) -> torch.Tensor:
        dist = MultivariateNormal(y_pred, scale_tril=self.scale_tril)
        return dist.rsample() # (B, T, N)
    
    def mean(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred


class PoissonConditionalLikelihood(nn.Module):
    """Poisson likelihood y ~ Poisson(exp(f_emission(x))); the emission outputs log-rates."""
    def __init__(self, N: int):
        super().__init__()
        self.N = N

    def log_prob(self, log_rate: torch.Tensor, y_true: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        # log_rate, y_true: (B, T, N); mask: (B, T, N) bool, True = masked out
        rate = torch.exp(log_rate)
        lp = Poisson(rate).log_prob(y_true)
        if mask is not None:
            lp = lp.masked_fill(mask, 0.0)
        return lp.sum(dim=(1, 2))  # (B,)

    def sample(self, log_rate: torch.Tensor) -> torch.Tensor:
        rate = torch.exp(log_rate)
        dist = Poisson(rate)
        return dist.sample()  # (B, T, N)

    def mean(self, log_rate: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_rate)


# -----------------------------------------------------------------------------
# Emission models
# -----------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> GELU -> Dropout -> Linear, with skip."""

    def __init__(self, H: int, p_dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        self.fc1 = nn.Linear(H, H)
        self.fc2 = nn.Linear(H, H)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dropout(torch.nn.functional.gelu(self.fc1(h)))
        h = self.fc2(h)
        return x + h


class ResMLPEmission(nn.Module):
    """Emission y_hat = W_skip x + MLP(x): linear readout plus a residual-MLP correction."""

    def __init__(self,
                 D: int,
                 N: int,
                 H: int = 64,               # hidden width
                 n_blocks: int = 2,         # number of residual blocks
                 p_dropout: float = 0.0):   # dropout probability in each block
        super().__init__()
        self.linear_skip = nn.Linear(D, N)
        self.in_proj = nn.Linear(D, H)
        self.blocks = nn.Sequential(*[ResBlock(H, p_dropout=p_dropout) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(H, N)

        # Small init so the emission starts near-linear.
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, N)"""
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        linear = self.linear_skip(x_flat)
        h = torch.nn.functional.gelu(self.in_proj(x_flat))
        h = self.blocks(h)
        nonlinear = self.out_proj(h)
        return (linear + nonlinear).view(B, T, -1)
