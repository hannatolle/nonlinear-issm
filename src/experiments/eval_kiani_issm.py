"""
Evaluate an iSSM trained on the macaque PFC micro-stimulation dataset.
"""

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


from src.datasets import RealDataDataset, get_dataloaders_real
from src.datasets import KianiLabDataLoader
from .eval_real_issm import EvalRealISSM, _SPLIT_SEED


class EvalKianiISSM(EvalRealISSM):
    """Evaluate an iSSM trained on the macaque micro-stimulation dataset.

    The test set is the held-out trials of the training pool (requires
    n_test_samples > 0 in the training config).
    """

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def _load_test_loader(self, train_config: dict) -> Optional[DataLoader]:
        """Rebuild the training pool and return the held-out test loader, or None."""
        ts = train_config["settings"]

        loader = KianiLabDataLoader(
            data_folder=ts["dataset_path"],
            fs=ts["fs"],
            trial_len=ts["trial_len"],
            seed=_SPLIT_SEED,
            population_mean_u=ts["population_mean_u"],
            binary_u=ts.get("binary_u", False))

        act = loader.get_trials(ts["condition"], compress_u=ts["compress_u"])
        y = torch.tensor(act.d, dtype=torch.float32)
        u = torch.tensor(act.u, dtype=torch.float32)

        n_test = ts.get("n_test_samples", 0)
        if n_test == 0:
            return None
        strata = (u.abs().sum(dim=(1, 2)) > 0).long().numpy().astype(np.int32)

        # Prefer the saved split; otherwise reproduce it from the split seed.
        splits_path = os.path.join(getattr(self, "_first_run_dir", ""), "splits.json")
        if os.path.isfile(splits_path):
            with open(splits_path) as _f:
                saved_splits = json.load(_f)
            test_idx = np.array(saved_splits["test_indices"], dtype=np.int64)
            self._test_trial_indices = test_idx
            ds = RealDataDataset(y=y[torch.tensor(test_idx)], u=u[torch.tensor(test_idx)])
            return DataLoader(ds, batch_size=ts.get("batch_size", 16), shuffle=False)

        _, _, test_loader, _ = get_dataloaders_real(
            y, u, strata,
            batch_size=ts.get("batch_size", 16),
            n_val_samples=ts.get("n_val_samples", 0),
            n_test_samples=n_test,
            seed=_SPLIT_SEED)
        return test_loader

    # -----------------------------------------------------------------------
    # Trial tier assignment
    # -----------------------------------------------------------------------

    def _get_trial_tiers(
        self,
        train_config: dict,
        u_all: torch.Tensor,
    ) -> np.ndarray:
        """Assign each test trial to "Control" (u=0) or "Stim" (u≠0)."""
        is_stim = u_all.abs().sum(dim=(1, 2)).cpu().numpy() > 0
        return np.where(is_stim, "Stim", "Control")
