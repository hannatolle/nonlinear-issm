"""Dataset-agnostic containers, torch Dataset wrappers and train/val/test split helpers."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NeuralActivity:
    """Trial-structured activity: d (trials, T, N), t (T,) in seconds, u (trials, T, M) or None."""

    def __init__(self, d: np.ndarray, t: np.ndarray, u: np.ndarray | None = None):
        self.d = d
        self.t = t
        self.u = u

    @property
    def shape(self):
        return self.d.shape


class SimDynamicsDataset(Dataset):
    def __init__(self, y: torch.Tensor, 
                 u: torch.Tensor,
                 x: torch.Tensor):
        B, T, N = y.shape
        _, _, D = x.shape
        _, _, M = u.shape
        assert x.shape == (B, T, D), "x and y must be of consistent shape (B, T, ...)"
        assert u.shape == (B, T, M), "u must be of consistent shape (B, T, ...)"
        self.y = y
        self.u = u
        self.x = x

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.y[idx], self.u[idx], self.x[idx]

class RealDataDataset(Dataset):
    """Real recordings without ground-truth latents.

    Items are (y, u, mask) with mask the per-trial (T, N) bool mask, or torch.zeros(1)
    as a placeholder when there is no mask (keeps the triple layout of SimDynamicsDataset).
    """

    def __init__(self, y: torch.Tensor, u: torch.Tensor,
                 mask: torch.Tensor | None = None):
        B, T, N = y.shape
        _, _, M = u.shape
        assert u.shape == (B, T, M), "y and u must have the same batch and time dimensions"
        self.y = y
        self.u = u
        self.mask = mask

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        m = self.mask[idx] if self.mask is not None else torch.zeros(1)
        return self.y[idx], self.u[idx], m


def _stratified_split(
    strata: np.ndarray,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
    force_test_idx: np.ndarray | list[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices into train/val/test preserving stratum proportions.

    Each requested split gets at least one sample per stratum. force_test_idx (e.g.
    an OOD stim-site holdout) is put in the test set and excluded from the pool;
    n_test then only tops the test set up to its target.
    """
    n_total = len(strata)
    force = sorted(set(int(i) for i in (force_test_idx if force_test_idx is not None else [])))
    pool = np.setdiff1d(np.arange(n_total), np.array(force, dtype=int))
    n_pool = len(pool)
    n_test_extra = max(0, n_test - len(force))

    val_idx, test_extra = [], []
    for s in np.unique(strata[pool]):
        s_idx = pool[strata[pool] == s]
        rng.shuffle(s_idx)
        frac = len(s_idx) / n_pool
        n_v = max(1, round(n_val * frac)) if n_val > 0 else 0
        n_t = max(1, round(n_test_extra * frac)) if n_test_extra > 0 else 0
        n_v = min(n_v, len(s_idx))
        n_t = min(n_t, len(s_idx) - n_v)
        val_idx.extend(s_idx[:n_v].tolist())
        test_extra.extend(s_idx[n_v:n_v + n_t].tolist())

    test_idx = force + test_extra
    held = set(val_idx) | set(test_idx)
    train_idx = [i for i in range(n_total) if i not in held]
    return train_idx, val_idx, test_idx


def get_dataloaders_real(
    y: torch.Tensor,
    u: torch.Tensor,
    strata: np.ndarray,
    batch_size: int = 16,
    n_val_samples: int = 20,
    n_test_samples: int = 0,
    seed: int = 0,
    mask: torch.Tensor | None = None,
    force_test_idx: np.ndarray | list[int] | None = None,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, dict]:
    """Stratified train/val/test DataLoaders for real data.

    Args:
        strata: (total_trials,) integer stratum labels.
        mask: optional (total_trials, T, N) bool mask (True = unobserved), sliced into every split.
        force_test_idx: indices forced into the test set (see _stratified_split).

    Returns:
        (train_loader, val_loader, test_loader, indices); val_loader is None when
        n_val_samples=0, test_loader is None when the test set is empty. indices maps
        "train_indices"/"val_indices"/"test_indices" to positions in y/u.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = _stratified_split(
        strata, n_val_samples, n_test_samples, rng, force_test_idx=force_test_idx)

    def _make_loader(idx, shuffle, mask=None):
        idx_t = torch.tensor(idx, dtype=torch.long)
        m = mask[idx_t] if mask is not None else None
        ds = RealDataDataset(y=y[idx_t], u=u[idx_t], mask=m)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    val_loader = _make_loader(val_idx, shuffle=False, mask=mask) if n_val_samples > 0 else None
    # test_idx can be non-empty with n_test_samples == 0 (forced OOD holdout).
    test_loader = _make_loader(test_idx, shuffle=False, mask=mask) if len(test_idx) > 0 else None

    indices = {
        "train_indices": train_idx,
        "val_indices":   val_idx,
        "test_indices":  test_idx,
    }
    return (_make_loader(train_idx, shuffle=True, mask=mask), val_loader, test_loader, indices)


def masked_channel_stats(
    y: torch.Tensor, mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel (N,) mean and std over (trial, time), ignoring masked cells.

    mask is (trials, T, N) bool, True = unobserved. std uses n-1 and is clamped to >= 1e-8.
    """
    if mask is None:
        return y.mean(dim=(0, 1)), y.std(dim=(0, 1)).clamp(min=1e-8)
    keep = ~mask
    n = keep.sum(dim=(0, 1))
    if (n < 2).any():
        raise ValueError("masked_channel_stats: a channel has fewer than 2 observed cells.")
    y0 = y.masked_fill(mask, 0.0)
    mean = y0.sum(dim=(0, 1)) / n
    sq = ((y0 - mean) ** 2).masked_fill(mask, 0.0)
    std = torch.sqrt(sq.sum(dim=(0, 1)) / (n - 1)).clamp(min=1e-8)
    return mean, std


def get_dataloaders(x_total, y_total, u_total, batch_size=5, n_val_samples=5, n_test_samples=0):
    """Random train/val/test DataLoaders for simulated data; test_loader is None when n_test_samples=0."""
    n_samples = x_total.shape[0]
    n_train = n_samples - n_val_samples - n_test_samples

    _g = torch.Generator()
    _g.manual_seed(0)
    indices = torch.randperm(n_samples, generator=_g)
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:n_train + n_val_samples]
    test_indices  = indices[n_train + n_val_samples:]

    def _make_loader(idx, shuffle):
        ds = SimDynamicsDataset(y=y_total[idx], u=u_total[idx], x=x_total[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    test_loader = _make_loader(test_indices, shuffle=False) if n_test_samples > 0 else None
    return (
        _make_loader(train_indices, shuffle=True),
        _make_loader(val_indices,   shuffle=False),
        test_loader,
    )
