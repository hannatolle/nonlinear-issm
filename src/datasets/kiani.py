"""Nejatbakhsh et al. (2023) macaque PFC micro-stimulation dataset (Utah-array spike counts)."""

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import loadmat

from src.datasets.common import NeuralActivity


class KianiLabDataLoader:
    """Loader for resting-state micro-stimulation sessions.

    Each *.PAG.mat file is one continuous session: '*.resting.ustim.PAG.mat' are
    stim sessions, '*.resting.PAG.mat' controls; '*.msacc.PAG.mat' (task) is ignored.
    Spikes are binned, sessions cropped to a common length, and fixed-length trial
    windows are centred on stim onsets (stim) or random times (control).

    Layout of each MATLAB FIRA struct:
        content[0]: metadata struct; content[0]['events'][0] is a string array
                    of event names, one per column of content[1].
        content[1]: (n_trials, n_events) float array — one row per behavioral
                    trial, one column per event. Values are event times in ms
                    (trial-relative).
        content[2]: (n_trials, 1) object array — content[2][t, 0] is a
                    (n_channels,) object array of spike-time vectors per trial.

    Args:
        fs: bin width in ms.
        trial_len: bins per trial window (trial_len // 2 either side of the event).
        seed: seeds the control-window placement.
        population_mean_u: u on the stimulated channel during stimulation is the
            cross-channel mean spike count instead of that electrode's own count.
        binary_u: binarise u to 0/1.
    """

    N_CHANNELS = 96

    def __init__(
        self,
        data_folder: Path,
        fs: int = 30,
        trial_len: int = 60,
        seed: int = 0,
        population_mean_u: bool = False,
        binary_u: bool = False,
    ):
        self.data_folder = Path(data_folder)
        self.fs = fs
        self.trial_len = trial_len
        self.margin = trial_len // 2
        self.seed = seed
        self.population_mean_u = population_mean_u
        self.binary_u = binary_u
        self._files = self._find_session_files()

    def get_active_channels(self) -> np.ndarray:
        """Sorted 0-indexed channels stimulated in any stim session; M = len(result)."""
        active = set()
        for path in self._files["stim"]:
            content = loadmat(
                str(path),
                squeeze_me=False,
                chars_as_strings=True,
                mat_dtype=True,
                struct_as_record=True,
                simplify_cells=True,
            )["FIRA"]
            _, stim_events = self._extract_events(content)
            for s in stim_events:
                active.add(int(s["chan"]))
        return np.array(sorted(active), dtype=np.int64)

    def get_trials(
        self,
        condition: Literal["control", "stim", "all"] = "all",
        compress_u: bool = True,
    ) -> "NeuralActivity":
        """Fixed-length trial windows from all sessions of a condition.

        One window per stim onset in stim sessions; in control sessions, as many
        random windows as the largest stim-event count of any stim session.

        Returns:
            NeuralActivity with d (trials, trial_len, N) spike counts, t (trial_len,)
            in seconds from window start, and u (trials, trial_len, M), zero on
            control trials. M = len(get_active_channels()) if compress_u, else 96.
        """
        key = condition.lower()
        if key == "all":
            session_paths = self._files["control"] + self._files["stim"]
        elif key in ("control", "stim"):
            session_paths = self._files[key]
        else:
            raise ValueError(
                f'condition must be "control", "stim", or "all", got {condition!r}.'
            )
        if not session_paths:
            raise ValueError(
                f"No sessions found for condition={condition!r} in {self.data_folder}."
            )

        # ys[i], us[i]: (n_bins_i, N); ts_ms[i]: (n_bins_i,) bin-centre times in ms.
        ys, us, ts_ms = [], [], []
        for path in session_paths:
            y, u, t_ms = self._load_session_continuous(path)
            ys.append(y); us.append(u); ts_ms.append(t_ms)

        dts_ms = [t[1] - t[0] for t in ts_ms]
        dt_ref_ms = dts_ms[0]
        bad = [
            (session_paths[i].name, round(dts_ms[i], 6))
            for i in range(1, len(dts_ms))
            if abs(dts_ms[i] - dt_ref_ms) > 1e-3
        ]
        if bad:
            warnings.warn(
                f"Inconsistent bin spacing across sessions (expected ≈{dt_ref_ms:.4f} ms). "
                f"Deviating sessions: {bad}. Using {dt_ref_ms:.6f} ms from the first session."
            )
        dt_s = dt_ref_ms * 1e-3

        # Crop to a common length, dropping 10 bins at each recording edge.
        min_n_bins = min(y.shape[0] for y in ys)
        last = min_n_bins - 10
        ys = [y[10:last] for y in ys]   # each (last-10, N)
        us = [u[10:last] for u in us]   # each (last-10, N)

        # Bins where the number of active stim channels increases.
        stim_onsets = [
            np.where(np.diff((u_ > 0).sum(axis=1)) > 0)[0]
            for u_ in us
        ]
        n_stims_max = max((len(s) for s in stim_onsets), default=0)

        rng = np.random.default_rng(self.seed)
        trial_y, trial_u = [], []
        n_bins_common = last - 10

        for i, path in enumerate(session_paths):
            y_, u_ = ys[i], us[i]
            is_stim = "ustim" in path.name

            if is_stim:
                for j in stim_onsets[i]:
                    start, end = int(j) - self.margin, int(j) + self.margin
                    if start < 0 or end > n_bins_common:
                        continue
                    trial_y.append(y_[start:end])  # (trial_len, N)
                    trial_u.append(u_[start:end])  # (trial_len, N)
            else:
                centres = rng.integers(self.margin, n_bins_common - self.margin,
                                       size=n_stims_max)
                for j in centres:
                    start, end = int(j) - self.margin, int(j) + self.margin
                    trial_y.append(y_[start:end])             # (trial_len, N)
                    trial_u.append(np.zeros_like(u_[start:end]))  # (trial_len, N)

        if not trial_y:
            raise ValueError(
                "No valid trial windows could be extracted. "
                "Check that trial_len is smaller than the session length."
            )

        d = np.stack(trial_y)    # (total_trials, trial_len, N)
        u_out = np.stack(trial_u)  # (total_trials, trial_len, N)
        t_out = np.arange(self.trial_len) * dt_s

        if compress_u:
            active = self.get_active_channels()
            u_out = u_out[:, :, active]          # (total_trials, trial_len, n_active)

        if self.binary_u:
            u_out = (u_out > 0).astype(u_out.dtype)

        return NeuralActivity(d=d, t=t_out, u=u_out)

    def _find_session_files(self) -> dict[str, list[Path]]:
        """Sorted control and stim *.PAG.mat paths in data_folder, keyed "control"/"stim"."""
        all_files = sorted(self.data_folder.glob("*.PAG.mat"))
        control = [f for f in all_files if "resting" in f.name and "ustim" not in f.name]
        stim    = [f for f in all_files if "resting.ustim" in f.name]
        return {"control": control, "stim": stim}

    def _load_session_continuous(
        self, path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load one session as continuous arrays.

        Returns y (n_bins, N) spike counts, u (n_bins, N) non-zero only on the
        stimulated channel during stimulation, and t_ms (n_bins,) bin-centre times.
        """
        content = loadmat(
            str(path),
            squeeze_me=False,
            chars_as_strings=True,
            mat_dtype=True,
            struct_as_record=True,
            simplify_cells=True,
        )["FIRA"]

        spikes, stim_events = self._extract_events(content)

        all_spk = np.concatenate([s for s in spikes if len(s) > 0])
        min_t, max_t = float(all_spk.min()), float(all_spk.max())
        n_bins = int(np.round((max_t - min_t) / self.fs))

        y, t_ms = self._spktimes_to_counts(spikes, n_bins, (min_t, max_t))

        u = np.zeros_like(y)
        pop_mean = y.mean(axis=1) if self.population_mean_u else None
        for s in stim_events:
            stim_mask = (t_ms > s["time"]) & (t_ms < s["time"] + s["dur"])
            u[stim_mask, s["chan"]] = (
                pop_mean[stim_mask] if self.population_mean_u else y[stim_mask, s["chan"]]
            )

        return y, u, t_ms

    @staticmethod
    def _extract_events(
        content,
    ) -> tuple[list[np.ndarray], list[dict]]:
        """Recording-relative spike times and stim events from a FIRA struct.

        Trial-relative times are converted via t_rec = t_trial + abs_ref[t] - abs_ref[0] - start_tri[0];
        the last 4 trials are excluded from the spikes.

        Returns:
            spikes: 96 arrays of spike times (ms), one per channel.
            stim_events: dicts with "chan" (0-indexed), "time" and "dur" (ms); empty for controls.
        """
        n_trials, _ = content[1].shape
        events = content[0]["events"][0]

        # "ustim_chan" is absent from control sessions.
        required_events = ["abs_ref", "start_tri", "elestim", "elestim_on", "elestim_off"]
        indices = {a: np.where(events == a)[0][0] for a in required_events}
        if "ustim_chan" in list(events):
            indices["ustim_chan"] = np.where(events == "ustim_chan")[0][0]

        # Control sessions have no elestim == 1 trials, so ustim_chan is never read.
        stim_events = [
            {
                "chan": int(content[1][t][indices["ustim_chan"]] - 1),  # MATLAB 1-indexed
                "time": (
                    content[1][t][indices["elestim_on"]]
                    + content[1][t][indices["abs_ref"]]
                    - content[1][0][indices["abs_ref"]]
                    - content[1][0][indices["start_tri"]]
                ),
                "dur": (
                    content[1][t][indices["elestim_off"]]
                    - content[1][t][indices["elestim_on"]]
                ),
            }
            for t in range(n_trials)
            if content[1][t][indices["elestim"]] == 1
        ]

        # spiking_times[t][c]: trial-relative spike times (ms) of channel c in trial t.
        spiking_times = [
            [
                list(content[2][t, 0][c])
                for c in range(len(content[2][t, 0]))
            ]
            for t in range(n_trials - 4)
        ]

        spikes = [
            np.concatenate([
                np.array(spiking_times[t][c])
                + content[1][t][indices["abs_ref"]]
                - content[1][0][indices["abs_ref"]]
                - content[1][0][indices["start_tri"]]
                for t in range(n_trials - 4)
            ])
            for c in range(KianiLabDataLoader.N_CHANNELS)
        ]

        return spikes, stim_events

    @staticmethod
    def _spktimes_to_counts(
        spikes: list[np.ndarray],
        n_bins: int,
        rng: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bin spike times over rng = (min_t, max_t) ms into y (n_bins, N) and times t (n_bins,)."""
        # t spans the endpoints, so it is only approximately the bin centres.
        t = np.linspace(rng[0], rng[1], n_bins)
        bin_edges = np.linspace(rng[0], rng[1], n_bins + 1)
        y = np.zeros((n_bins, len(spikes)))
        for c, spk in enumerate(spikes):
            y[:, c] = np.histogram(spk, bin_edges)[0]
        return y, t
