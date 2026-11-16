"""Latent transition models p(x | u) and the site-embedding input map."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import math
from abc import ABC, abstractmethod
from typing import Optional


class InitialCondition(nn.Module):
    """Initial condition x0 ~ N(mu, L L^T)."""
    def __init__(self, D: int,
                 x_init_sigma: float = 0.05,
                 min_logstd: float = -5.0,
                 max_logstd: float = 5.0,
                 diagonal_cov: bool = False):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(D))
        self.log_diag = nn.Parameter(torch.log(torch.ones(D) * x_init_sigma))
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.diagonal_cov = diagonal_cov

        if not self.diagonal_cov:
            self.off_diag = nn.Parameter(torch.zeros(D, D))
        else:
            self.off_diag = None

    @property
    def no_decay_params(self) -> set[str]:
        """Parameter names excluded from weight decay."""
        if self.diagonal_cov:
            return {"mu", "log_diag"}
        else:
            return {"mu", "log_diag", "off_diag"}

    @property
    def scale_tril(self):
        L = torch.diag(torch.exp(torch.clamp(self.log_diag, min=self.min_logstd, max=self.max_logstd)))
        if not self.diagonal_cov:
            L = L + torch.tril(self.off_diag, diagonal=-1)
        return L

    def sample(self, batch_size: int) -> torch.Tensor:
        dist = MultivariateNormal(self.mu, scale_tril=self.scale_tril)
        return dist.rsample((batch_size,))

    def log_prob(self, x0: torch.Tensor) -> torch.Tensor:
        dist = MultivariateNormal(self.mu, scale_tril=self.scale_tril)
        return dist.log_prob(x0) # (B, D) -> (B,)


class TransitionModel(nn.Module, ABC):
    """Base class for latent transition models with a (D, n_stim) input matrix B."""
    def __init__(self, D: int,
                 M: int,                              # input channels (stim + task)
                 dt: float = 0.1,
                 interventional: bool = True,         # hard interventions (replace mu by Bu)
                 discrete: bool = False,              # if False, noise is scaled by sqrt(dt)
                 train_leak: bool = False,            # learnable per-latent leak rate
                 train_B: bool = True,                # if False, B stays fixed at its initial value
                 B_init: str = "randn",               # "randn", "eye", "eye_noisy" or "none"
                 B_init_scale: float = 1.0,           # scale of the initial B
                 sparsity: float = 0.0,               # weight of the B sparsity prior
                 B_prior: str = "none",               # "none" or "l1"
                 B_threshold: float = 1e-4,           # |B| below this does not count as driven
                 max_tau_B: float = 0.05,             # max hard-threshold magnitude for B
                 diagonal_cov: bool = False,          # diagonal initial-state and process noise covariance
                 x_init_sigma: float = 0.05,          # initial std of the initial-state and process noise
                 min_logstd: float = -5.0,            # lower bound of the noise log std
                 max_logstd: float = 5.0,             # upper bound of the noise log std
                 n_task_channels: int = 0):           # trailing channels of u used as context
        super().__init__()
        _valid_B_priors = ("none", "l1")
        if B_prior not in _valid_B_priors:
            raise ValueError(f"B_prior must be one of {_valid_B_priors}, got '{B_prior}'")
        self.D = D
        self.M = M
        self.dt = dt
        self.interventional = interventional
        self.discrete = discrete
        self.train_B = train_B
        self.B_init = B_init
        self.B_init_scale = B_init_scale
        self.sparsity = sparsity
        self.B_prior = B_prior
        self.B_threshold = B_threshold
        self.max_tau_B = max_tau_B
        self.diagonal_cov = diagonal_cov
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.n_task_channels = n_task_channels
        self.n_stim = M - n_task_channels
        self.initial = InitialCondition(D, x_init_sigma=x_init_sigma,
                                        min_logstd=min_logstd, max_logstd=max_logstd,
                                        diagonal_cov=diagonal_cov)

        self._init_B()

        # Optional SiteEmbeddingB, held in a list so it is NOT registered as a submodule
        # and its params stay out of this module's state_dict / parameters().
        self._b_net = []

        self.log_diag = nn.Parameter(torch.log(torch.ones(D) * x_init_sigma))
        if not self.diagonal_cov:
            self.off_diag = nn.Parameter(torch.zeros(D, D))
        else:
            self.off_diag = None

        # Per-latent leak gamma = softplus(theta), initialised to 1. theta is None for
        # the fixed gamma = 1 case (no state_dict entry).
        self.train_leak = train_leak
        if train_leak:
            self.theta = nn.Parameter(torch.full((D,), math.log(math.e - 1)))
        else:
            self.theta = None

    @property
    def no_decay_params(self) -> set[str]:
        """Parameter names excluded from weight decay; subclasses extend this set."""
        return {"theta"} if self.theta is not None else set()

    def _init_B(self):
        """Allocate self.B (D, n_stim) as a parameter or buffer.

        B_init="none" leaves self.B as None; B must then come from set_B_net.
        """
        if self.B_init == "none":
            self.B = None
            return
        if self.B_init == "eye":
            B_val = torch.eye(self.D, self.n_stim) * self.B_init_scale
        elif self.B_init == "eye_noisy":
            # Off-diagonal entries of random sign in [2, 10] * B_threshold, so all entries
            # start above the gradient dead zone |B| < B_threshold.
            B_val = torch.eye(self.D, self.n_stim) * self.B_init_scale
            off_mask = 1.0 - torch.eye(self.D, self.n_stim)
            lo, hi = self.B_threshold * 2, self.B_threshold * 10
            magnitudes = torch.empty(self.D, self.n_stim).uniform_(lo, hi)
            signs = 2 * torch.randint(0, 2, (self.D, self.n_stim)).float() - 1
            B_val = B_val + off_mask * magnitudes * signs
        elif self.B_init == "randn":
            B_val = torch.randn(self.D, self.n_stim) * self.B_init_scale
        else:
            raise ValueError(f"Invalid B_init: {self.B_init}")
            
        if self.train_B:
            self.B = nn.Parameter(B_val)
        else:
            self.register_buffer("B", B_val)

    @property
    def B_net(self):
        """The attached SiteEmbeddingB, or None."""
        return self._b_net[0] if self._b_net else None

    def set_B_net(self, b_net) -> None:
        """Attach a SiteEmbeddingB so that B = b_net().

        The net is saved and optimised as its own component, so it is not registered
        here; gradients still flow through B_effective. Requires B_init="none" and
        max_tau_B=0, and scales the net's output layer once by B_init_scale.
        """
        if self.B is not None:
            raise ValueError("set_B_net requires B_init='none' (no raw B parameter/buffer).")
        if self.max_tau_B != 0:
            raise ValueError("set_B_net requires max_tau_B=0 (B thresholding is disabled).")
        self._b_net = [b_net]
        b_net.set_B_init(self.B_init_scale)
        if b_net.features_set and b_net.features.shape[0] != self.n_stim:
            raise ValueError(
                f"SiteEmbeddingB produces {b_net.features.shape[0]} columns but dynamics "
                f"expects n_stim={self.n_stim}.")

    @property
    def B_effective(self) -> torch.Tensor:
        """Effective input matrix of shape (D, n_stim)."""
        if self.B_net is not None:
            return self.B_net()
        if self.B is None:
            raise ValueError(
                "B is None (B_init='none'); attach a B_net via set_B_net() before use.")
        return self.B

    @property
    def scale_tril(self):
        diag_std = torch.exp(torch.clamp(self.log_diag, min=self.min_logstd, max=self.max_logstd))
        L = torch.diag(diag_std)
        if not self.diagonal_cov:
            L = L + torch.tril(self.off_diag, diagonal=-1)
        # Euler-Maruyama diffusion scaling.
        if not self.discrete:
            L = L * math.sqrt(self.dt)
        return L

    @abstractmethod
    def _drift(self, x_prev: torch.Tensor, u_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Drift of the dynamics; u_ctx is optional task context (..., n_task_channels)."""
        pass

    def _mu(self, x_prev: torch.Tensor, u_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean of x(t+1) from the autonomous dynamics (no stim term)."""
        drift = self._drift(x_prev, u_ctx)
        if self.theta is None:
            return (1 - self.dt) * x_prev + self.dt * drift
        gamma = F.softplus(self.theta)  # (D,) per-latent leak rate
        return x_prev + self.dt * (drift - gamma * x_prev)

    def step(self, x_prev: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean of x(t+1) given x(t) and u(t+1), with interventional masking on stim channels."""
        u_ctx = (u[..., self.n_stim:] if self.n_task_channels > 0 else None) if u is not None else None
        mu = self._mu(x_prev, u_ctx)
        if u is None:
            return mu
        u_stim = u[..., :self.n_stim]    # (..., n_stim)
        inputs = u_stim @ self.B_effective.T
        mask = (self.get_inputs(u) == 0).float()  # 1 where not driven, 0 where driven
        if self.interventional:
            mu = mask * mu + (1 - mask) * inputs
        else:
            mu = mu + (1 - mask) * inputs
        return mu

    def flow(self, x_prev: torch.Tensor, u_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Vector field dx/dt at x, optionally conditioned on context."""
        mu = self._mu(x_prev, u_ctx) # (B, D)
        return (mu - x_prev) / self.dt

    @torch.no_grad()
    def get_inputs(self, u: torch.Tensor) -> torch.Tensor:
        """u_stim @ B_clean.T, shape (..., D); non-zero entries mark driven latents."""
        u_stim = u[..., :self.n_stim]
        B_eff = self.B_effective
        B_clean = torch.where(B_eff.abs() < self.B_threshold,
                              torch.zeros_like(B_eff), B_eff)
        return u_stim @ B_clean.T

    def sample(self, u: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample a latent trajectory (B, T, D) from the transition dynamics."""
        if u.dim() == 2:
            batch_size = x0.shape[0] if x0 is not None else 1
            u = u.unsqueeze(0).expand(batch_size, -1, -1)
        batch_size, T, _ = u.shape

        if x0 is None:
            x0 = self.initial.sample(batch_size)

        x = torch.zeros(batch_size, T, self.D, device=u.device, dtype=x0.dtype)
        x[:, 0, :] = x0
        for t in range(1, T):
            mu = self.step(x[:, t - 1, :], u[:, t, :]) # (B, D)
            dist = MultivariateNormal(mu, scale_tril=self.scale_tril)
            x[:, t, :] = dist.rsample() # (B, D)
        return x

    def mean(self, u: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Noise-free rollout (B, T, D) of the transition mean."""
        if u.dim() == 2:
            batch_size = x0.shape[0] if x0 is not None else 1
            u = u.unsqueeze(0).expand(batch_size, -1, -1)
        batch_size, T, _ = u.shape

        if x0 is None:
            x0 = self.initial.sample(batch_size)

        x = torch.zeros(batch_size, T, self.D, device=u.device, dtype=u.dtype)
        x[:, 0, :] = x0
        for t in range(1, T):
            x[:, t, :] = self.step(x[:, t - 1, :], u[:, t, :])
        return x

    def log_prob(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """log p(x | u) = log p(x0) + sum_t log p(x_t | x_{t-1}, u_t), shape (B,)."""
        B, T, D = x.shape
        logp = self.initial.log_prob(x[:, 0, :]) # (B,)
        for t in range(1, T):
            mu = self.step(x[:, t - 1, :], u[:, t, :]) # (B, D)
            dist = MultivariateNormal(mu, scale_tril=self.scale_tril)
            logp += dist.log_prob(x[:, t, :]) # (B,)
        return logp

    def apply_soft_threshold(self, frac: float):
        """Zero entries with |B| < max_tau_B * frac (frac in [0, 1] is the warmup fraction).

        Hard rather than soft thresholding, so large entries are not shrunk.
        Call after optimizer.step(); no-op when B is not trained.
        """
        if not self.train_B or self.B is None:
            return
        tau = self.max_tau_B * frac
        with torch.no_grad():
            self.B.copy_(torch.where(self.B.abs() < tau, torch.zeros_like(self.B), self.B))

    def sparse_B_prior(self) -> torch.Tensor:
        """Log sparsity prior on B: 0 for "none", -sparsity * mean(|B|) for "l1"."""
        if self.B_prior == "none":
            return torch.tensor(0.0, device=next(self.parameters()).device)
        else:
            return -self.sparsity * self.B_effective.abs().mean()

    def sparsity_priors(self) -> torch.Tensor:
        """Total sparsity prior of the transition parameters."""
        return self.sparse_B_prior()

    def describe(self) -> str:
        """Human-readable summary of the dynamics."""
        return super().__repr__()


class LinearDynamics(TransitionModel):
    """Linear drift A [x, ctx], Euler-integrated, with input matrix B."""
    def __init__(self, D: int,
                 M: int,
                 dt: float = 0.1,
                 interventional: bool = True,
                 discrete: bool = False,
                 train_leak: bool = False,
                 train_B: bool = True,
                 B_init: str = "randn",
                 B_init_scale: float = 1.0,
                 sparsity: float = 0.0,
                 B_prior: str = "none",
                 B_threshold: float = 1e-3,
                 max_tau_B: float = 0.05,
                 diagonal_cov: bool = False,
                 x_init_sigma: float = 0.05,
                 min_logstd: float = -5.0,
                 max_logstd: float = 5.0,
                 init_scale: Optional[float] = 0.01,  # scale of the random initial A (None: unscaled)
                 n_task_channels: int = 0):
        super().__init__(D=D, M=M, dt=dt, interventional=interventional,
                         discrete=discrete, train_leak=train_leak,
                         train_B=train_B, B_init=B_init, B_init_scale=B_init_scale, sparsity=sparsity,
                         B_prior=B_prior, B_threshold=B_threshold, max_tau_B=max_tau_B,
                         diagonal_cov=diagonal_cov, x_init_sigma=x_init_sigma,
                         min_logstd=min_logstd, max_logstd=max_logstd,
                         n_task_channels=n_task_channels)
        # A maps [x_prev, u_ctx] -> drift, shape (D, D + n_task_channels).
        A_init = torch.randn(D, D + n_task_channels)
        if init_scale is not None:
            A_init = A_init * init_scale
        self.A = nn.Parameter(A_init)

    @property
    def no_decay_params(self) -> set[str]:
        own = {"A", "B"} if self.train_B else {"A"}
        return super().no_decay_params | own

    def _drift(self, x_prev: torch.Tensor, u_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u_ctx is not None and self.n_task_channels > 0:
            inp = torch.cat([x_prev, u_ctx], dim=-1)  # (..., D + n_task_channels)
        elif self.n_task_channels > 0:
            # Zero context when none is given (e.g. flow()).
            inp = torch.cat([x_prev, x_prev.new_zeros(*x_prev.shape[:-1], self.n_task_channels)], dim=-1)
        else:
            inp = x_prev
        return inp @ self.A.T

    def describe(self, precision: int = 3) -> list[str]:
        """Learned drift equations, one string per latent; terms rounding to zero are omitted."""
        A = self.A.detach().cpu().numpy()  # (D, D + n_task_channels)
        state_names = [f"x{i+1}" for i in range(self.D)]
        ctx_names   = [f"ctx{i+1}" for i in range(self.n_task_channels)]
        feature_names = state_names + ctx_names

        equations = []
        for j in range(self.D):
            terms = []
            for i, coef in enumerate(A[j]):
                rounded = round(coef, precision)
                if rounded == 0:
                    continue
                terms.append((rounded, feature_names[i]))

            if len(terms) == 0:
                eq = f"x{j+1}' = 0"
            else:
                parts = []
                for k, (rounded, name) in enumerate(terms):
                    if k == 0:
                        parts.append(f"{rounded:.{precision}f}*{name}" if rounded >= 0 else f"-{abs(rounded):.{precision}f}*{name}")
                    else:
                        parts.append(f" + {rounded:.{precision}f}*{name}" if rounded >= 0 else f" - {abs(rounded):.{precision}f}*{name}")
                eq = f"x{j+1}' = " + "".join(parts)

            equations.append(eq)

        return equations


class AnalyticMLPDynamics(TransitionModel):
    """MLP drift MLP([x, ctx]), Euler-integrated, with input matrix B."""
    def __init__(self, D: int,
                 M: int,
                 dt: float = 0.1,
                 interventional: bool = True,
                 discrete: bool = False,
                 train_leak: bool = False,
                 train_B: bool = True,
                 B_init: str = "randn",
                 B_init_scale: float = 1.0,
                 sparsity: float = 0.0,
                 B_prior: str = "none",
                 B_threshold: float = 1e-3,
                 max_tau_B: float = 0.05,
                 diagonal_cov: bool = False,
                 x_init_sigma: float = 0.05,
                 min_logstd: float = -5.0,
                 max_logstd: float = 5.0,
                 hidden_dim: int = 100,               # width of the hidden layers
                 num_hidden_layers: int = 1,          # number of hidden layers
                 activation: str = 'gelu',            # 'gelu' or 'softplus'
                 init_scale: Optional[float] = 0.01,  # std of the output-layer weights at init (None: default)
                 n_task_channels: int = 0):
        super().__init__(D=D, M=M, dt=dt, interventional=interventional,
                         discrete=discrete, train_leak=train_leak,
                         train_B=train_B, B_init=B_init, B_init_scale=B_init_scale, sparsity=sparsity,
                         B_prior=B_prior, B_threshold=B_threshold, max_tau_B=max_tau_B,
                         diagonal_cov=diagonal_cov, x_init_sigma=x_init_sigma,
                         min_logstd=min_logstd, max_logstd=max_logstd,
                         n_task_channels=n_task_channels)

        _activations = {'gelu': nn.GELU, 'softplus': nn.Softplus}
        if activation not in _activations:
            raise ValueError(f"activation must be one of {list(_activations)}, got '{activation}'")
        act_cls = _activations[activation]

        layers = []
        in_dim = D + n_task_channels
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), act_cls()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, D))
        self.mlp = nn.Sequential(*layers)

        last_linear = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        if init_scale is not None:
            nn.init.normal_(last_linear.weight, std=init_scale)
        nn.init.zeros_(last_linear.bias)

    @property
    def no_decay_params(self) -> set[str]:
        own = {"B"} if self.train_B else set()
        return super().no_decay_params | own

    def _drift(self, x_prev: torch.Tensor, u_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u_ctx is not None and self.n_task_channels > 0:
            inp = torch.cat([x_prev, u_ctx], dim=-1)
        elif self.n_task_channels > 0:
            inp = torch.cat([x_prev, x_prev.new_zeros(*x_prev.shape[:-1], self.n_task_channels)], dim=-1)
        else:
            inp = x_prev
        return self.mlp(inp)

    def describe(self, precision: int = 3) -> str:
        ctx_str = f", ctx{self.n_task_channels}d" if self.n_task_channels > 0 else ""
        return f"x(t+1) = MLP([x(t){ctx_str}])"


class SiteEmbeddingB(nn.Module):
    """MLP mapping per-site features to columns of the input matrix B.

    Nearby sites share structure, so the map generalises to unseen stimulation sites.
    Features are (x, y, z), plus a white-matter flag unless drop_white_matter.
    """

    _activations = {'gelu': nn.GELU, 'relu': nn.ReLU, 'softplus': nn.Softplus}
    _N_COORD = 3  # leading xyz columns that get spatial normalization

    def __init__(self, drop_white_matter: bool,
                 D: int,
                 M: int,                       # input channels (stim + task)
                 n_task_channels: int = 0,     # trailing channels of u used as context
                 hidden_dim: int = 32,         # width of the hidden layers
                 n_layers: int = 2,            # number of Linear layers
                 activation: str = 'gelu'):    # 'gelu', 'relu' or 'softplus'
        super().__init__()
        if activation not in self._activations:
            raise ValueError(
                f"activation must be one of {list(self._activations)}, got '{activation}'")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.drop_white_matter = drop_white_matter
        self.n_site_features = 3 if drop_white_matter else 4
        self.D = D
        self.n_stim = M - n_task_channels
        if self.n_stim <= 0:
            raise ValueError(
                f"n_stim = M - n_task_channels must be > 0, got M={M}, "
                f"n_task_channels={n_task_channels}.")

        # Placeholders until set_site_features / set_B_init.
        self.register_buffer("features", torch.zeros(self.n_stim, self.n_site_features))
        self.register_buffer("feat_mean", torch.zeros(self._N_COORD))
        self.register_buffer("feat_scale", torch.ones(()))
        self.register_buffer("init_scale_set", torch.zeros((), dtype=torch.bool))
        self.register_buffer("features_set", torch.zeros((), dtype=torch.bool))

        act_cls = self._activations[activation]
        layers = []
        in_dim = self.n_site_features
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), act_cls()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, D))
        self.mlp = nn.Sequential(*layers)

    @property
    def _last_linear(self) -> nn.Linear:
        return [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]

    def set_B_init(self, B_init_scale) -> None:
        """Scale the output layer once so B starts at a controlled magnitude."""
        if self.init_scale_set:
            return
        if B_init_scale is not None:
            with torch.no_grad():
                self._last_linear.weight.mul_(B_init_scale)
                self._last_linear.bias.mul_(B_init_scale)
        self.init_scale_set.fill_(True)

    def set_site_features(self, feats: torch.Tensor) -> None:
        """Store (n_stim, n_site_features) features; row k is stim channel k of u.

        Also caches the coordinate mean and an isotropic scale for normalisation.
        """
        feats = feats.to(torch.float32)
        if feats.shape != self.features.shape:
            raise ValueError(
                f"Expected features of shape {tuple(self.features.shape)} "
                f"(n_stim, n_site_features; drop_white_matter={self.drop_white_matter}), "
                f"got {tuple(feats.shape)}.")
        coords = feats[:, :self._N_COORD]
        centered = coords - coords.mean(0)
        self.features = feats
        self.feat_mean = coords.mean(0)
        self.feat_scale = centered.pow(2).mean().sqrt() + 1e-6
        self.features_set.fill_(True)

    def _normalize(self, feats: torch.Tensor) -> torch.Tensor:
        """Normalise xyz with the cached mean/scale; the white-matter flag is left as-is."""
        coords = (feats[:, :self._N_COORD] - self.feat_mean) / self.feat_scale
        if feats.shape[1] > self._N_COORD:
            return torch.cat([coords, feats[:, self._N_COORD:]], dim=1)
        return coords

    def forward(self) -> torch.Tensor:
        """Return the intervention matrix B of shape (D, n_stim)."""
        if not self.features_set:
            raise RuntimeError("Call set_site_features() before using SiteEmbeddingB.")
        return self.mlp(self._normalize(self.features)).T  # (n_stim, D) -> (D, n_stim)
