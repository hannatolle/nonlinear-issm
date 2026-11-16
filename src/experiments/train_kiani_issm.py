"""
Train an iSSM on the macaque PFC micro-stimulation dataset (Nejatbakhsh et al., 2023).
"""

import numpy as np
import torch

from src.datasets import KianiLabDataLoader
from .train_real_issm import TrainRealDataISSM


class TrainKianiISSM(TrainRealDataISSM):
    """Train an iSSM on macaque PFC spike counts under micro-stimulation.

    Sessions are discovered from all *.resting.ustim.PAG.mat (stim) and
    *.resting.PAG.mat (control) files in dataset_path.

    Trials are extracted by windowing ±(trial_len//2) bins around each
    stimulation event (stim sessions) or around randomly sampled timepoints
    (control sessions, matched in count to the maximum stim-event count).

    u carries the stimulated electrode's own spike count during the stim
    window on its channel, zero elsewhere. With compress_u=True, u keeps only
    channels ever stimulated (M = n_active_channels).

    Strata for splitting: 0 = control (u all-zero), 1 = stim.
    """

    def check_config(self):
        required_components = ["dynamics", "emission", "likelihood", "recognition"]
        for comp in required_components:
            if comp not in self.components:
                raise ValueError(f"TrainKianiISSM requires component '{comp}'.")
        self.match_settings(strict=True)
        s = self.fetch_settings()
        if s.condition not in ("stim", "all"):
            raise ValueError(
                f"condition must be 'stim' or 'all', got '{s.condition}'. "
                f"('control' alone is not supported: control-window count is "
                f"determined from stim events and cannot be inferred without stim sessions.)"
            )
        if s.n_val_samples == 0 and s.early_stopping:
            raise ValueError(
                "early_stopping=True requires n_val_samples > 0 "
                "(validation set is needed to monitor training)."
            )
        expected_dt = s.fs / 1000.0
        if abs(expected_dt - s.dt) > 1e-6:
            raise ValueError(
                f"dt must equal fs / 1000 = {expected_dt:.6f} s, "
                f"but config has dt={s.dt}. "
                f"Either set dt={expected_dt} or adjust fs."
            )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Load spike counts and micro-stimulation inputs.

        Returns:
            y:      (total_trials, trial_len, N) float32 tensor.
            u:      (total_trials, trial_len, M) float32 tensor.
            strata: (total_trials,) int32 array; 0 = control, 1 = stim.
        """
        s = self.fetch_settings()

        loader = KianiLabDataLoader(
            data_folder=s.dataset_path,
            fs=s.fs,
            trial_len=s.trial_len,
            seed=self.seed,
            population_mean_u=s.population_mean_u,
            binary_u=s.binary_u,
        )

        act = loader.get_trials(s.condition, compress_u=s.compress_u)
        y = torch.tensor(act.d, dtype=torch.float32)   # (total, T, N)
        u = torch.tensor(act.u, dtype=torch.float32)   # (total, T, M)

        strata = (u.abs().sum(dim=(1, 2)) > 0).long().numpy().astype(np.int32)

        if y.shape[2] != s.N:
            raise ValueError(
                f"Expected N={s.N} channels, but data has {y.shape[2]}. "
                f"Update N in the config."
            )
        if u.shape[2] != s.M:
            raise ValueError(
                f"Expected M={s.M} input channels, but data has {u.shape[2]}. "
                f"Update M in the config (tip: run loader.get_active_channels() "
                f"to find the correct M when compress_u=True)."
            )

        return y, u, np.array(strata, dtype=np.int32)
