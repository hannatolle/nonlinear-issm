"""Base class for simulated rate models with interventions."""

from typing import Callable, Optional
from abc import ABC, abstractmethod
import torch
import math

    
class RateModel(ABC):
    """Continuous-time rate model with Euler integration, interventions and an observation model."""

    def __init__(self, N: int,
                 D: int = None,
                 M: int = None,
                 discrete: bool = False,                  # if False, noise is scaled by sqrt(dt)
                 B: torch.Tensor = None,                  # (D, M) input matrix
                 x_noise: float = 0.0,                    # std of the latent process noise
                 y_noise: float = 0.0,                    # std of the observation noise
                 seed: int = 0,                           # seeds C_obs and all simulator noise
                 obs_nonlinearity: Optional[str] = None,  # pointwise nonlinearity after C_obs
                 C_obs_scale: float = 1.0,                # scale of the observation matrix C_obs
                 alpha: float = 1.0,                      # 1 = hard clamp, 0 = purely additive
                 **kwargs):
        self.N = N
        self.D = D if D is not None else N
        self.M = M if M is not None else N
        self.discrete = discrete
        if B is not None:
            self.B = B
        elif self.M <= self.D:
            self.B = torch.eye(self.D, self.M)
        else:
            # M > D: channel m drives latent m % D.
            _B = torch.zeros(self.D, self.M)
            _B[torch.arange(self.M) % self.D, torch.arange(self.M)] = 1.0
            self.B = _B
        self.x_noise = x_noise
        self.y_noise = y_noise
        self.seed = seed
        self.obs_nonlinearity = obs_nonlinearity
        assert self.obs_nonlinearity in (None, 'tanh', 'sin', 'elu', 'arcsinh', 'cubic'), \
            f"obs_nonlinearity must be None, 'tanh', 'sin', 'elu', 'arcsinh', or 'cubic', got '{obs_nonlinearity}'"
        assert self.B.shape == (self.D, self.M), "Input matrix B must be of shape (D, M)"
        self.alpha = alpha
        assert 0.0 <= self.alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
        self.params = {'D': D, 'M': M, 'N': N, 'discrete': discrete,
                       'x_noise': x_noise, 'y_noise': y_noise,
                       'seed': seed, 'obs_nonlinearity': obs_nonlinearity,
                       'C_obs_scale': C_obs_scale, 'alpha': alpha,
                       **kwargs}

        # Local generator so the global random state is unaffected.
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
        C_raw = torch.randn(self.N, self.D, generator=self.gen)  # (N, D)
        if self.N >= self.D:
            Q, _ = torch.linalg.qr(C_raw)
            self.C_obs = Q * C_obs_scale  # (N, D)
        else:
            # N < D: no injective map exists; use row-normalised rows.
            self.C_obs = C_raw / C_raw.norm(dim=1, keepdim=True) * C_obs_scale

    def rand_state(self, n_samples: int = 1) -> torch.Tensor:
        """Random initial state (n_samples, D)."""
        return 0.01 * torch.randn(n_samples, self.D, generator=self.gen)

    def generate_x_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian latent noise; override for other noise models."""
        noise = torch.randn(x.shape, dtype=x.dtype, generator=self.gen)
        return self.x_noise * noise.to(x.device)

    def generate_y_noise(self, y: torch.Tensor) -> torch.Tensor:
        """Gaussian observation noise; override for other noise models."""
        noise = torch.randn(y.shape, dtype=y.dtype, generator=self.gen)
        return self.y_noise * noise.to(y.device)

    def run(self, T: float, 
            dt: float, 
            x0: torch.Tensor, 
            u: Optional[torch.Tensor] = None,
            b: Optional[Callable[[float], torch.Tensor]] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Euler-integrate the system.

        Args:
            x0: (batch_size, D)
            u: hard-intervention inputs, (batch_size, T_steps, M)
            b: optional additive (soft) perturbation b(t) -> (D,)

        Returns:
            t: (T_steps,), x: (T_steps, batch_size, D) -- note time-major.
        """
        t_eval = torch.arange(0, T, dt)
        n_steps = len(t_eval)
        batch_size = x0.shape[0]

        x = torch.zeros(n_steps, batch_size, self.D)
        x[0] = x0

        for t in range(1, n_steps):
            current_t = t_eval[t]
            dx = dt * self.step(current_t, x[t - 1])  # (batch_size, D)

            if self.x_noise > 0:
                scale = 1 if self.discrete else math.sqrt(dt)
                dx += scale * self.generate_x_noise(x[t - 1])

            # Soft (additive) perturbation.
            if b is not None:
                dx += dt * b(current_t)

            # Hard interventions: driven latents are set to Bu.
            if u is not None:
                du = torch.einsum('dm,bm->bd', self.B, u[:, t]) # (batch_size, D)

                if self.alpha != 1.0:
                    # Blend clamp and free dynamics at driven entries; the branch keeps
                    # alpha=1 bitwise identical to the pure clamp.
                    x[t] = torch.where(du == 0, x[t - 1] + dx,
                                       du + (1.0 - self.alpha) * (x[t - 1] + dx))
                else:
                    x[t] = torch.where(du == 0, x[t - 1] + dx, du)

            else:
                x[t] = x[t - 1] + dx

        return t_eval, x

    @abstractmethod
    def step(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """dx/dt = f(x, t)."""
        pass

    def _obs(self, x: torch.Tensor,
             t: Optional[float] = None,
             u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Noise-free observation y = act(C_obs x); C_obs (N, D) has orthonormal columns when N >= D."""
        y = x @ self.C_obs.T  # (..., D) @ (D, N) -> (..., N)
        if self.obs_nonlinearity == 'tanh':
            y = torch.tanh(y)
        elif self.obs_nonlinearity == 'sin':
            y = torch.sin(y)
        elif self.obs_nonlinearity == 'elu':
            y = torch.nn.functional.elu(y, alpha=1.0)
        elif self.obs_nonlinearity == 'arcsinh':
            y = torch.asinh(y)
        elif self.obs_nonlinearity == 'cubic':
            y = y + 0.01 * y ** 3
        return y

    def obs(self, x: torch.Tensor, 
            t: Optional[float] = None,
            u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Noisy observation of x."""
        y = self._obs(x, t, u)
        if self.y_noise > 0:
            y += self.generate_y_noise(y)
        return y

    def flow(self, x: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        """Vector field dx/dt at x (autonomous systems)."""
        return self.step(t, x) # (B, D)
