"""
Evaluate an iSSM trained on the Huang et al. (2023) SEEG single-pulse-stimulation
dataset (ds004457).
"""

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import RealDataDataset
from .eval_real_issm import EvalRealISSM
from .train_huang_issm import TrainHuangISSM


class EvalHuangISSM(EvalRealISSM):
    """Evaluate an iSSM trained on the Huang 2023 human SEEG/CCEP dataset.

    Every Huang trial is a single stimulation pulse (no control trials). The
    test set is exactly the trials saved to splits.json["test_indices"] at
    training time: the in-distribution held-out trials plus, when an OOD
    stim-site holdout was used, all trials of the held-out sites
    (splits.json["ood_test_indices"]). Multi-tier eval splits these into
    "Stim" vs "Stim_OOD".

    Only the test loader (with its observation mask) and the trial tiers are
    dataset-specific; everything else is in EvalRealISSM.
    """

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def _load_test_loader(self, train_config: dict) -> Optional[DataLoader]:
        """Rebuild the training pool and observation mask, then slice y / u / mask
        to the test indices in splits.json (required).
        """
        ts = train_config["settings"]

        loader = TrainHuangISSM._build_loader(ts)
        act = loader.get_trials()
        y = torch.tensor(act.d, dtype=torch.float32)   # (n_trials, T, N)
        u = torch.tensor(act.u, dtype=torch.float32)   # (n_trials, T, M)

        full_mask, _ = loader.build_loss_mask(act.t)

        splits_path = os.path.join(getattr(self, "_first_run_dir", ""), "splits.json")
        if not os.path.isfile(splits_path):
            raise FileNotFoundError(
                f"EvalHuangISSM requires splits.json in the seed dir; not found "
                f"at {splits_path}.")
        with open(splits_path) as f:
            saved = json.load(f)
        # Saved indices are positions in the training pool: verify it is the same pool.
        pool_ids = saved.get("pool_trial_ids")
        if pool_ids is not None and not np.array_equal(np.asarray(pool_ids), loader.trial_ids):
            raise ValueError(
                f"Rebuilt Huang trial pool ({loader.n_trials} trials) differs from the "
                f"training pool in splits.json ({len(pool_ids)} trials); test_indices "
                f"would point at the wrong trials.")

        test_idx = np.array(saved["test_indices"], dtype=np.int64)
        self._test_trial_indices = test_idx
        self._ood_test_indices = {int(i) for i in (saved.get("ood_test_indices") or [])}
        if test_idx.size == 0:
            return None

        idx_t = torch.tensor(test_idx)
        ds = RealDataDataset(y=y[idx_t], u=u[idx_t], mask=full_mask[idx_t])
        return DataLoader(ds, batch_size=ts.get("batch_size", 16), shuffle=False)

    # -----------------------------------------------------------------------
    # Trial tier assignment
    # -----------------------------------------------------------------------

    def _get_trial_tiers(
        self,
        train_config: dict,
        u_all: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """Label each test trial "Stim_OOD" (held-out site) or "Stim".

        Returns None when there was no OOD holdout, so the base skips the
        multi-tier CSV (a single all-"Stim" tier is uninformative).
        """
        ood = getattr(self, "_ood_test_indices", None)
        if not ood:
            return None
        is_ood = np.array([int(i) in ood for i in self._test_trial_indices])
        return np.where(is_ood, "Stim_OOD", "Stim").astype(object)
