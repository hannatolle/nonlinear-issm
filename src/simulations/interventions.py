"""Stimulation protocols u(t) for simulated interventions."""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Union, Optional


class StimulatorBase(ABC):
    """Base class for stimulation protocols.

    sigma is the std of Gaussian noise added to active pulses; stim_control_ratio is the
    fraction of batches that receive stimulation (the rest are zeroed controls).
    """

    def __init__(self, M: int, sigma: float = 0.0, seed: Optional[int] = None,
                 stim_control_ratio: float = 1.0):
        assert 0.0 <= stim_control_ratio <= 1.0, (
            f"stim_control_ratio must be in [0, 1]; got {stim_control_ratio}")
        self.M = M
        self.sigma = sigma
        self.seed = seed
        self.stim_control_ratio = stim_control_ratio

    def _apply_control_ratio(self, u: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if self.stim_control_ratio >= 1.0:
            return u
        n_batches = u.shape[0]
        n_control = round(n_batches * (1 - self.stim_control_ratio))
        if n_control > 0:
            control_idx = rng.choice(n_batches, size=n_control, replace=False)
            u[control_idx] = 0.0
        return u

    @abstractmethod
    def generate(self,
                 n_batches: int,
                 T: float,
                 dt: float,
                 stimulated: Optional[Union[list, np.ndarray]] = None) -> torch.Tensor:
        """Return (n_batches, T_steps, M) inputs, T_steps = round(T / dt).

        T and dt are in seconds; stimulated lists the channel indices (default: all).
        """


class RegularStimulator(StimulatorBase):
    """Regular pulses, round-robin over the stimulated channels.

    Each cycle pulses one channel for stim_d seconds, then rests for rest_d seconds.
    amplitude is either fixed (scalar or per-channel array of shape (M,)) or a list
    of values drawn per pulse, at random or cyclically (amplitude_mode).
    """
    def __init__(self,
                 M: int,
                 amplitude: Union[float, list, np.ndarray] = 1.0,
                 stim_d: float = 1.0,
                 rest_d: float = 1.0,
                 sigma: float = 0.0,
                 seed: Optional[int] = None,
                 amplitude_mode: str = 'random',
                 stim_control_ratio: float = 1.0):
        super().__init__(M=M, sigma=sigma, seed=seed, stim_control_ratio=stim_control_ratio)
        assert amplitude_mode in ('random', 'cyclic'), (
            f"amplitude_mode must be 'random' or 'cyclic'; got {amplitude_mode!r}")
        self.stim_d = stim_d
        self.rest_d = rest_d
        self.amplitude_mode = amplitude_mode

        # A list means per-pulse amplitudes; a scalar/array means fixed amplitudes.
        if isinstance(amplitude, list):
            self._amplitude_values = np.array(amplitude, dtype=float)
            self._amplitude_fixed = None
        else:
            self._amplitude_values = None
            self._amplitude_fixed = (
                np.full(M, amplitude) if np.isscalar(amplitude)
                else np.array(amplitude, dtype=float).flatten())

    @property
    def min_amplitude(self) -> float:
        vals = self._amplitude_values if self._amplitude_values is not None else self._amplitude_fixed
        return float(np.min(vals))

    @property
    def max_amplitude(self) -> float:
        vals = self._amplitude_values if self._amplitude_values is not None else self._amplitude_fixed
        return float(np.max(vals))

    def generate(self,
                 n_batches: int,
                 T: float,
                 dt: float,
                 stimulated: Optional[Union[list, np.ndarray]] = None) -> torch.Tensor:
        """Generate (n_batches, T_steps, M) inputs, T_steps = round(T / dt)."""
        if stimulated is None:
            stimulated = np.arange(self.M)
        stimulated = np.array(stimulated).flatten()

        T_steps = round(T / dt)

        if self.stim_d == 0:
            return torch.zeros(n_batches, T_steps, self.M)

        stim_steps = max(1, round(self.stim_d / dt))
        rest_steps = round(self.rest_d / dt)
        cycle_steps = stim_steps + rest_steps

        # Local RNG: does not affect global numpy or torch random state
        rng = np.random.default_rng(self.seed)

        u_list = []
        for _ in range(n_batches):
            I = np.zeros((T_steps, self.M))
            t = 0
            pulse_idx = 0
            n_nodes = len(stimulated)
            while t + stim_steps <= T_steps:
                node = stimulated[pulse_idx % n_nodes]
                amp = self._get_amplitude(node, pulse_idx, rng)
                I[t:t + stim_steps, node] = amp
                if self.sigma > 0:
                    I[t:t + stim_steps, node] += self.sigma * rng.standard_normal(stim_steps)
                pulse_idx += 1
                t += cycle_steps
            u_list.append(torch.tensor(I, dtype=torch.float32))

        return self._apply_control_ratio(torch.stack(u_list), rng)

    def _get_amplitude(self, node: int, pulse_idx: int, rng: np.random.Generator) -> float:
        if self._amplitude_values is not None:
            if self.amplitude_mode == 'cyclic':
                return float(self._amplitude_values[pulse_idx % len(self._amplitude_values)])
            else:  # 'random'
                return float(rng.choice(self._amplitude_values))
        else:
            return float(self._amplitude_fixed[node])


class RandomStimulator(StimulatorBase):
    """Poisson pulse timing with uniformly sampled amplitudes.

    One global Poisson process (stim_rate, Hz) fires pulses of stim_d seconds on a
    random stimulated channel, amplitude ~ Uniform(min_amplitude, max_amplitude).
    Overlapping pulses on the same channel are overwritten by the later one.
    """
    def __init__(self,
                 M: int,
                 stim_rate: float = 1.0,
                 stim_d: float = 1.0,
                 min_amplitude: float = 0.5,
                 max_amplitude: float = 1.5,
                 sigma: float = 0.0,
                 seed: Optional[int] = None,
                 stim_control_ratio: float = 1.0):
        super().__init__(M=M, sigma=sigma, seed=seed, stim_control_ratio=stim_control_ratio)
        assert min_amplitude <= max_amplitude, (
            f"min_amplitude ({min_amplitude}) must be <= max_amplitude ({max_amplitude})")
        self.stim_rate = stim_rate
        self.stim_d = stim_d
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def generate(self,
                 n_batches: int,
                 T: float,
                 dt: float,
                 stimulated: Optional[Union[list, np.ndarray]] = None) -> torch.Tensor:
        """Generate (n_batches, T_steps, M) inputs, T_steps = round(T / dt)."""
        if stimulated is None:
            stimulated = np.arange(self.M)
        stimulated = np.array(stimulated).flatten()
        n_nodes = len(stimulated)

        T_steps = round(T / dt)

        if self.stim_rate == 0:
            return torch.zeros(n_batches, T_steps, self.M)

        stim_steps = max(1, round(self.stim_d / dt))

        rng = np.random.default_rng(self.seed)

        u_list = []
        for _ in range(n_batches):
            I = np.zeros((T_steps, self.M))

            t = 0.0
            while True:
                iet = rng.exponential(1.0 / self.stim_rate)
                t += iet
                t_step = round(t / dt)
                if t_step >= T_steps:
                    break

                node = stimulated[rng.integers(n_nodes)]
                amp = rng.uniform(self.min_amplitude, self.max_amplitude)
                end = min(t_step + stim_steps, T_steps)
                I[t_step:end, node] = amp
                if self.sigma > 0:
                    I[t_step:end, node] += self.sigma * rng.standard_normal(end - t_step)

            u_list.append(torch.tensor(I, dtype=torch.float32))

        return self._apply_control_ratio(torch.stack(u_list), rng)


class HybridStimulator(StimulatorBase):
    """Stimulation protocol with independent timing and amplitude switches.

    timing="poisson" as in RandomStimulator (uses stim_rate); "regular" as in
    RegularStimulator (uses rest_d). amplitude="diverse" draws each pulse from
    Uniform(min_amplitude, max_amplitude); "constant" draws one amplitude for the
    whole run, |A| ~ Uniform(constant_amp_frac * max_amplitude, max_amplitude) with
    random sign (assumes symmetric support). It is drawn first, so constant runs
    sharing a seed use the same amplitude regardless of timing.
    """
    def __init__(self,
                 M: int,
                 timing: str = "poisson",
                 amplitude: str = "diverse",
                 stim_rate: float = 1.0,
                 rest_d: float = 1.0,
                 stim_d: float = 0.25,
                 min_amplitude: float = -1.0,
                 max_amplitude: float = 1.0,
                 constant_amp_frac: float = 0.5,
                 sigma: float = 0.0,
                 seed: Optional[int] = None,
                 stim_control_ratio: float = 1.0):
        super().__init__(M=M, sigma=sigma, seed=seed, stim_control_ratio=stim_control_ratio)
        assert timing in ("poisson", "regular"), (
            f"timing must be 'poisson' or 'regular'; got {timing!r}")
        assert amplitude in ("constant", "diverse"), (
            f"amplitude must be 'constant' or 'diverse'; got {amplitude!r}")
        assert min_amplitude <= max_amplitude, (
            f"min_amplitude ({min_amplitude}) must be <= max_amplitude ({max_amplitude})")
        assert 0.0 <= constant_amp_frac <= 1.0, (
            f"constant_amp_frac must be in [0, 1]; got {constant_amp_frac}")
        self.timing = timing
        self.amplitude = amplitude
        self.stim_rate = stim_rate
        self.rest_d = rest_d
        self.stim_d = stim_d
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.constant_amp_frac = constant_amp_frac

    def _draw_constant(self, rng: np.random.Generator) -> float:
        """One fixed amplitude for the whole run."""
        mag = rng.uniform(self.constant_amp_frac * self.max_amplitude, self.max_amplitude)
        sign = rng.choice([-1.0, 1.0])
        return float(sign * mag)

    def _pulse_amp(self, rng: np.random.Generator, const_amp: Optional[float]) -> float:
        return (const_amp if const_amp is not None
                else float(rng.uniform(self.min_amplitude, self.max_amplitude)))

    def generate(self,
                 n_batches: int,
                 T: float,
                 dt: float,
                 stimulated: Optional[Union[list, np.ndarray]] = None) -> torch.Tensor:
        """Generate (n_batches, T_steps, M) inputs, T_steps = round(T / dt)."""
        if stimulated is None:
            stimulated = np.arange(self.M)
        stimulated = np.array(stimulated).flatten()

        T_steps = round(T / dt)

        if (self.timing == "poisson" and self.stim_rate == 0) or \
           (self.timing == "regular" and self.stim_d == 0):
            return torch.zeros(n_batches, T_steps, self.M)

        rng = np.random.default_rng(self.seed)
        const_amp = self._draw_constant(rng) if self.amplitude == "constant" else None

        if self.timing == "poisson":
            u = self._generate_poisson(n_batches, T_steps, dt, stimulated, rng, const_amp)
        else:
            u = self._generate_regular(n_batches, T_steps, dt, stimulated, rng, const_amp)
        return self._apply_control_ratio(u, rng)

    def _generate_poisson(self, n_batches, T_steps, dt, stimulated, rng, const_amp):
        n_nodes = len(stimulated)
        stim_steps = max(1, round(self.stim_d / dt))
        u_list = []
        for _ in range(n_batches):
            I = np.zeros((T_steps, self.M))
            t = 0.0
            while True:
                t += rng.exponential(1.0 / self.stim_rate)
                t_step = round(t / dt)
                if t_step >= T_steps:
                    break
                node = stimulated[rng.integers(n_nodes)]
                amp = self._pulse_amp(rng, const_amp)
                end = min(t_step + stim_steps, T_steps)
                I[t_step:end, node] = amp
                if self.sigma > 0:
                    I[t_step:end, node] += self.sigma * rng.standard_normal(end - t_step)
            u_list.append(torch.tensor(I, dtype=torch.float32))
        return torch.stack(u_list)

    def _generate_regular(self, n_batches, T_steps, dt, stimulated, rng, const_amp):
        n_nodes = len(stimulated)
        stim_steps = max(1, round(self.stim_d / dt))
        cycle_steps = stim_steps + round(self.rest_d / dt)
        u_list = []
        for _ in range(n_batches):
            I = np.zeros((T_steps, self.M))
            t = 0
            pulse_idx = 0
            while t + stim_steps <= T_steps:
                node = stimulated[pulse_idx % n_nodes]
                amp = self._pulse_amp(rng, const_amp)
                I[t:t + stim_steps, node] = amp
                if self.sigma > 0:
                    I[t:t + stim_steps, node] += self.sigma * rng.standard_normal(stim_steps)
                pulse_idx += 1
                t += cycle_steps
            u_list.append(torch.tensor(I, dtype=torch.float32))
        return torch.stack(u_list)
