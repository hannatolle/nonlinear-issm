"""Two-dimensional toy dynamical systems."""

import math
import torch
from .base import RateModel


class VanDerPol(RateModel):
    """
    Van der Pol oscillator:
        dx/dt = y
        dy/dt = mu * (1 - x^2) * y - x
    """
    def __init__(self, N: int = 2,
        D: int = 2,
        M: int = 2,
        discrete: bool = False,
        B: torch.Tensor = None,
        x_noise: float = 0.0,
        y_noise: float = 0.0,
        seed: int = 0,
        obs_nonlinearity: str = None,
        C_obs_scale: float = 1.0,
        alpha: float = 1.0,
        mu: float = 2.0):
        super().__init__(N=N, D=D, M=M, discrete=discrete, B=B,
                         x_noise=x_noise, y_noise=y_noise,
                         seed=seed, obs_nonlinearity=obs_nonlinearity,
                         C_obs_scale=C_obs_scale, alpha=alpha, mu=mu)
        self.mu = mu

    def step(self, t: float, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1
        return torch.stack([dx1, dx2], dim=-1)

    def describe(self) -> list:
        """Ground-truth latent state equations."""
        return [f"x1' = x2",
                f"x2' = {self.mu:.2f}*x2 - {self.mu:.2f}*x1^2*x2 - x1"]


class FitzHughNagumo(RateModel):
    """
    FitzHugh-Nagumo neuron model (v fast, w slow recovery):
        dv/dt = v - v^3/3 - w + I_ext
        dw/dt = (v + a - b*w) / tau

    The defaults give a limit cycle; I_ext below the Hopf threshold (~0.34) gives a stable fixed point.
    """

    def __init__(self, N: int = 2,
                 D: int = 2,
                 M: int = 2,
                 discrete: bool = False,
                 B: torch.Tensor = None,
                 x_noise: float = 0.0,
                 y_noise: float = 0.0,
                 seed: int = 0,
                 obs_nonlinearity: str = None,
                 C_obs_scale: float = 1.0,
                 alpha: float = 1.0,
                 a: float = 0.7,         # recovery offset
                 b: float = 0.8,         # recovery coupling
                 tau: float = 12.5,      # recovery time constant
                 I_ext: float = 0.5):    # constant external current
        super().__init__(N=N, D=D, M=M, discrete=discrete, B=B,
                         x_noise=x_noise, y_noise=y_noise,
                         seed=seed, obs_nonlinearity=obs_nonlinearity,
                         C_obs_scale=C_obs_scale, alpha=alpha,
                         a=a, b=b, tau=tau, I_ext=I_ext)
        assert self.D == 2, "FitzHughNagumo must have D=2"
        self.a = a
        self.b = b
        self.tau = tau
        self.I_ext = I_ext

    def step(self, t: float, x: torch.Tensor) -> torch.Tensor:
        v, w = x[:, 0], x[:, 1]
        dv = v - v**3 / 3.0 - w + self.I_ext
        dw = (v + self.a - self.b * w) / self.tau
        return torch.stack([dv, dw], dim=-1)

    def describe(self) -> list:
        """Ground-truth latent state equations."""
        return [f"v' = v - v^3/3 - w + {self.I_ext:.2f}",
                f"w' = (v + {self.a:.2f} - {self.b:.2f}*w) / {self.tau:.2f}"]


class Saddle(RateModel):
    """
    Saddle model of nonlinear evidence integration:
        dx1/dt = a*x1^3 + b*x1 + coh
        dx2/dt = c*x2 + coh

    With a < 0, b > 0, x1 has two stable choice states separated by a saddle.
    Decision-making model from Nejatbakhsh & Wang (2025).
    """

    def __init__(self, N: int = 10,
                 D: int = 2,
                 M: int = 2,
                 discrete: bool = False,
                 B: torch.Tensor = None,
                 x_noise: float = 0.1,
                 y_noise: float = 0.1,
                 seed: int = 0,
                 obs_nonlinearity: str = 'tanh',
                 C_obs_scale: float = 1.0,
                 alpha: float = 1.0,
                 a: float = -0.6,    # cubic coefficient
                 b: float = 2.0,     # linear coefficient
                 c: float = -1.0,    # decay of x2
                 coh: float = 0.1):  # stimulus coherence drive
        super().__init__(N=N, D=D, M=M, discrete=discrete, B=B,
                         x_noise=x_noise, y_noise=y_noise,
                         seed=seed, obs_nonlinearity=obs_nonlinearity,
                         C_obs_scale=C_obs_scale, alpha=alpha,
                         a=a, b=b, c=c, coh=coh)
        assert self.D == 2, "Saddle must have D=2"
        self.a = a
        self.b = b
        self.c = c
        self.coh = coh

    def step(self, t: float, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        dx1 = self.a * x1 ** 3 + self.b * x1 + self.coh
        dx2 = self.c * x2 + self.coh
        return torch.stack([dx1, dx2], dim=-1)

    def describe(self) -> list:
        """Ground-truth latent state equations."""
        return [f"x1' = {self.a:.2f}*x1^3 + {self.b:.2f}*x1 + {self.coh:.2f}",
                f"x2' = {self.c:.2f}*x2 + {self.coh:.2f}"]


class LineAttractor(RateModel):
    """
    Line attractor (perfect integration):
        dx/dt = A x,   A = M(theta) R diag(0, eval1) R^{-1}
    with R = [r0; l0] (rows) and M(theta) a 2D rotation.
    Decision-making model from Nejatbakhsh & Wang (2025).
    """

    def __init__(self, N: int = 10,
                 D: int = 2,
                 M: int = 2,
                 discrete: bool = False,
                 B: torch.Tensor = None,
                 x_noise: float = 0.1,
                 y_noise: float = 0.1,
                 seed: int = 0,
                 obs_nonlinearity: str = 'tanh',
                 C_obs_scale: float = 1.0,
                 alpha: float = 1.0,
                 l0_x1: float = 1.0,   # left eigenvector  l0 = [l0_x1, l0_x2]
                 l0_x2: float = 1.0,
                 r0_x1: float = 1.0,   # right eigenvector r0 = [r0_x1, r0_x2]
                 r0_x2: float = 0.0,
                 eval1: float = -1.0,  # stable eigenvalue
                 theta_deg: float = 45.0):  # rotation angle (degrees)
        super().__init__(N=N, D=D, M=M, discrete=discrete, B=B,
                         x_noise=x_noise, y_noise=y_noise,
                         seed=seed, obs_nonlinearity=obs_nonlinearity,
                         C_obs_scale=C_obs_scale, alpha=alpha,
                         l0_x1=l0_x1, l0_x2=l0_x2, r0_x1=r0_x1, r0_x2=r0_x2,
                         eval1=eval1, theta_deg=theta_deg)
        assert self.D == 2, "LineAttractor must have D=2"
        self.eval1 = eval1
        self.theta_deg = theta_deg

        l0_t = torch.tensor([l0_x1, l0_x2], dtype=torch.float32)
        l0_t = l0_t / torch.linalg.norm(l0_t)
        r0_t = torch.tensor([r0_x1, r0_x2], dtype=torch.float32)
        Lambda = torch.diag(torch.tensor([0.0, eval1], dtype=torch.float32))
        R = torch.stack([r0_t, l0_t])          # rows are the eigenvectors
        L = torch.linalg.inv(R)
        A = R @ Lambda @ L
        theta = math.radians(theta_deg)
        cth, sth = math.cos(theta), math.sin(theta)
        Mrot = torch.tensor([[cth, -sth], [sth, cth]], dtype=torch.float32)
        self.A = Mrot @ A                       # (D, D)

    def step(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T

    def describe(self) -> list:
        """Ground-truth latent state equations."""
        A = self.A
        return [f"x1' = {A[0, 0]:.2f}*x1 + {A[0, 1]:.2f}*x2",
                f"x2' = {A[1, 0]:.2f}*x1 + {A[1, 1]:.2f}*x2"]


class PointAttractor(RateModel):
    """
    Point attractor (input-driven decisions): dx/dt = diag(-|a1|, -|a2|) x.
    Decision-making model from Nejatbakhsh & Wang (2025).
    """

    def __init__(self, N: int = 10,
                 D: int = 2,
                 M: int = 2,
                 discrete: bool = False,
                 B: torch.Tensor = None,
                 x_noise: float = 0.1,
                 y_noise: float = 0.1,
                 seed: int = 0,
                 obs_nonlinearity: str = 'tanh',
                 C_obs_scale: float = 1.0,
                 alpha: float = 1.0,
                 a1: float = -0.5,   # decay rate of x1
                 a2: float = -1.0):  # decay rate of x2
        assert a1 < 0 and a2 < 0, "PointAttractor decay rates a1, a2 must be negative"
        super().__init__(N=N, D=D, M=M, discrete=discrete, B=B,
                         x_noise=x_noise, y_noise=y_noise,
                         seed=seed, obs_nonlinearity=obs_nonlinearity,
                         C_obs_scale=C_obs_scale, alpha=alpha, a1=a1, a2=a2)
        assert self.D == 2, "PointAttractor must have D=2"
        self.a1 = a1
        self.a2 = a2
        self.A = torch.diag(torch.tensor([-abs(a1), -abs(a2)], dtype=torch.float32))

    def step(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T

    def describe(self) -> list:
        """Ground-truth latent state equations."""
        return [f"x1' = {-abs(self.a1):.2f}*x1",
                f"x2' = {-abs(self.a2):.2f}*x2"]
