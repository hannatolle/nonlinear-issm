"""Huang et al. (2023) SEEG single-pulse stimulation dataset (OpenNeuro ds004457).

Loads the arrays written by scripts.huang.preprocess (run in the camel_preproc env),
one subject at a time. Each preproc/<subject>/ directory holds:
    epochs.npz     X[trial, ch, t] (microvolts), u[trial, t, M], tt[t], fs
    channels.csv   kept SEEG channels (name, orig_idx, seeg_idx, status, good)
    stim_sites.csv M stim sites (one u channel each)
    events.csv     per-trial stim_site_idx (used as the stratification label)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.datasets.common import NeuralActivity


def _is_white_matter(labels: pd.Series) -> np.ndarray:
    """Boolean mask of white-matter Destrieux labels (White_Matter, WM_*); empty labels are non-WM."""
    s = labels.fillna("").astype(str).str.lower()
    return (s.str.contains("white_matter") | s.str.startswith("wm")).to_numpy()


# Clinical epileptogenicity hierarchy (higher = closer to the seizure onset).
_ZONE_RANK = {"soz": 3, "earlypropagationzone": 2, "irritativezone": 1}
_ZONE_NAME = {0: "Healthy", 1: "IrritativeZone", 2: "EarlyPropagationZone", 3: "SOZ"}


def _zone_severity(cell) -> int:
    """Max epileptogenicity rank (0 = healthy, 3 = SOZ) of a comma-joined seizure_zone cell."""
    if cell is None:
        return 0
    s = str(cell).strip().lower()
    if s in ("", "nan"):
        return 0
    return max((_ZONE_RANK.get(tok.strip(), 0) for tok in s.split(",")), default=0)


class HuangDataloader:
    """Loader for one preprocessed subject.

    Args:
        drop_bad: exclude channels flagged bad in channels.csv.
        require_location: keep only channels with an anatomical location (has_location).
        drop_white_matter: drop white-matter recording channels (N), white-matter stim
            sites (M) and the trials that stimulated them.
        resample_hz: anti-aliased decimation rate for y (must divide fs); u is rebuilt
            on the reduced grid rather than decimated. None keeps the native rate.
        t_start_ms, t_end_ms: crop window relative to the pulse at t=0; must bracket 0.
        normalize_u: scale u so its max is 1 (keeps relative current amplitudes).
        drop_artefact_trials: exclude trials flagged drop==1 in stim_artefact_trials.csv.
        drop_residual_artefact_trials: exclude trials where stim_artefact_mask.csv flags
            a kept channel other than the trial's own stim contacts.

    Filters are applied at load time; N, M and n_trials reflect the kept subset.
    u has one channel per stim site: a single-sample impulse at pulse onset whose
    value is the current in mA (or normalised when normalize_u).
    """

    def __init__(self, preproc_dir: str | Path = "data/huang_2023/preproc",
                 subject: str = "sub-2", drop_bad: bool = True,
                 require_location: bool = True, drop_white_matter: bool = False,
                 resample_hz: float | None = None, t_start_ms: float = -50.0,
                 t_end_ms: float = 300.0, normalize_u: bool = True,
                 drop_artefact_trials: bool = True,
                 drop_residual_artefact_trials: bool = True):
        self.dir = Path(preproc_dir) / subject
        self.subject = subject
        self.drop_bad = drop_bad
        self.require_location = require_location
        self.drop_white_matter = drop_white_matter
        self.resample_hz = resample_hz
        self.t_start_ms = t_start_ms
        self.t_end_ms = t_end_ms
        self.normalize_u = normalize_u
        self.drop_artefact_trials = drop_artefact_trials
        self.drop_residual_artefact_trials = drop_residual_artefact_trials
        if not (self.dir / "epochs.npz").exists():
            raise FileNotFoundError(
                f"No preprocessed data at {self.dir}. Run "
                f"`python -m scripts.huang.preprocess --subject {subject}` "
                f"in the camel_preproc env first.")
        # X is loaded lazily in get_trials() so N/M-only callers stay cheap.
        self._npz = np.load(self.dir / "epochs.npz")
        self._tt = self._npz["tt"]    # (t,)
        self.fs = float(self._npz["fs"])
        channels = pd.read_csv(self.dir / "channels.csv")
        self.sites = pd.read_csv(self.dir / "stim_sites.csv")
        self.events = pd.read_csv(self.dir / "events.csv")

        # Recording-channel filter (axis of y).
        keep = np.ones(len(channels), dtype=bool)
        if drop_bad:
            keep &= channels["good"].to_numpy()
        if require_location:
            if "has_location" not in channels.columns:
                raise KeyError(
                    "channels.csv has no 'has_location' column; re-run "
                    "scripts.huang.preprocess to add electrode anatomy, or pass "
                    "require_location=False.")
            keep &= channels["has_location"].to_numpy().astype(bool)
        if drop_white_matter:
            keep &= ~_is_white_matter(channels["Destrieux_label_text"])
        self._keep = np.where(keep)[0]
        self.channels = channels.iloc[self._keep].reset_index(drop=True)
        # Map full recording-channel index (X channel axis) -> kept position.
        self._chan_full_to_kept = {int(c): k for k, c in enumerate(self._keep)}

        # Stim-site filter (columns of u) and trial filter.
        site_keep = np.ones(len(self.sites), dtype=bool)
        if drop_white_matter:
            site_keep &= ~_is_white_matter(self.sites["region"])
        self._site_keep = np.where(site_keep)[0]
        self._site_full_to_kept = {int(s): k for k, s in enumerate(self._site_keep)}
        trial_site = self.events["stim_site_idx"].to_numpy()
        trial_keep = np.isin(trial_site, self._site_keep)
        if drop_artefact_trials:
            atp = self.dir / "stim_artefact_trials.csv"
            if atp.exists():
                at = pd.read_csv(atp)
                drop_ids = set(at.loc[at["drop"].astype(bool), "trial_idx"].astype(int))
                if drop_ids:
                    trial_keep &= ~np.isin(np.arange(len(trial_site)), list(drop_ids))
        # Drop whole trials with residual artefact on a kept non-stim-contact channel.
        self.n_residual_artefact_dropped = 0
        if drop_residual_artefact_trials:
            resid = self._residual_artefact_trials(channels["name"].tolist())
            self.n_residual_artefact_dropped = int((trial_keep & resid).sum())
            trial_keep &= ~resid
        self._trial_keep = np.where(trial_keep)[0]

    def _residual_artefact_trials(self, all_channel_names: list[str]) -> np.ndarray:
        """(n_all_trials,) bool: a kept channel other than the trial's own stim contacts is flagged."""
        ap = self.dir / "stim_artefact_mask.csv"
        if not ap.exists():
            raise FileNotFoundError(
                "drop_residual_artefact_trials=True but stim_artefact_mask.csv is "
                "absent. Run python -m scripts.huang.stim_artefact --subject "
                f"{self.subject} first.")
        df = pd.read_csv(ap)
        n_all = len(self.events)
        if not np.array_equal(df["trial_idx"].to_numpy(), np.arange(n_all)):
            raise ValueError("stim_artefact_mask.csv rows do not match events.csv trials.")
        if list(df.columns[1:]) != all_channel_names:
            raise ValueError("stim_artefact_mask.csv columns do not match channels.csv names.")
        flagged = df.drop(columns=["trial_idx"]).to_numpy().astype(bool)[:, self._keep]
        own = self._stim_contacts(self.events["stim_site_idx"].to_numpy())
        return (flagged & ~own).any(axis=1)

    def _stim_contacts(self, trial_site: np.ndarray) -> np.ndarray:
        """(len(trial_site), N) bool: the kept channels of each trial's stim pair."""
        mask = np.zeros((len(trial_site), self.N), dtype=bool)
        for i, s in enumerate(trial_site):
            row = self.sites.loc[int(s)]
            for m in (row["member1_seeg"], row["member2_seeg"]):
                k = self._chan_full_to_kept.get(int(m))
                if k is not None:
                    mask[i, k] = True
        return mask

    @property
    def N(self) -> int:
        """Number of kept recording (SEEG) channels."""
        return len(self._keep)

    @property
    def M(self) -> int:
        """Number of kept stimulation sites."""
        return len(self._site_keep)

    @property
    def n_trials(self) -> int:
        """Number of kept trials."""
        return len(self._trial_keep)

    @property
    def trial_ids(self) -> np.ndarray:
        """Full (events.csv) trial indices of the kept trials, in pool order."""
        return self._trial_keep.copy()

    def resolve_ood_sites(self, names: list[str]) -> np.ndarray:
        """Map stim-site names to full stim_sites.csv indices (as in get_strata()).

        Raises if a name is unknown or removed by the active filters, so an OOD
        holdout cannot silently change meaning.
        """
        name_to_full = {n: i for i, n in enumerate(self.sites["site_name"])}
        out = []
        for nm in names:
            if nm not in name_to_full:
                raise ValueError(
                    f"OOD channel {nm!r} is not a stim site in {self.subject}'s "
                    f"stim_sites.csv. Known sites: {list(name_to_full)}.")
            full = name_to_full[nm]
            if full not in self._site_keep:
                is_wm = bool(_is_white_matter(self.sites.iloc[[full]]["region"])[0])
                why = ("white matter and drop_white_matter=True"
                       if is_wm and self.drop_white_matter
                       else "excluded by the active channel/site filters")
                raise ValueError(
                    f"OOD test channel {nm!r} (full site idx {full}) is {why}, so "
                    f"it is not in the data pool and cannot be held out for test. "
                    f"Set drop_white_matter=False or remove it from "
                    f"ood_test_channels.")
            out.append(full)
        return np.array(out, dtype=int)

    @property
    def site_features(self) -> torch.Tensor:
        """(M, n_features) site features for SiteEmbeddingB; row k is u-channel k.

        Standard-space x/y/z, plus a 0/1 white-matter flag unless drop_white_matter.
        """
        kept = self.sites.iloc[self._site_keep]
        xyz = kept[["x", "y", "z"]].to_numpy(dtype=np.float64)
        if self.drop_white_matter:
            feats = xyz
        else:
            is_wm = _is_white_matter(kept["region"]).astype(np.float64)[:, None]
            feats = np.concatenate([xyz, is_wm], axis=1)
        return torch.tensor(feats, dtype=torch.float32)

    def get_trials(self) -> NeuralActivity:
        """Kept trials as NeuralActivity: d (n_trials, T, N) in microvolts, t (T,) in s, u (n_trials, T, M).

        y is decimated then cropped. u is rebuilt as a single-sample impulse at the
        sample nearest t=0, since decimating it would smear the impulse.
        """
        from scripts.huang.signal import decimate_signal

        X = self._npz["X"][self._trial_keep][:, self._keep, :]   # (trials, N, T_full)
        tt = self._tt.astype(np.float64)
        if self.resample_hz is not None and self.resample_hz != self.fs:
            # Per-trial decimation keeps peak memory to one (N, T_full) block.
            q = int(round(self.fs / self.resample_hz))
            Xr = np.stack([decimate_signal(X[i], self.fs, self.resample_hz)[0]
                           for i in range(X.shape[0])])
            tt = tt[::q][:Xr.shape[-1]]
        else:
            Xr = X.astype(np.float64)

        sel = (tt >= self.t_start_ms / 1e3) & (tt <= self.t_end_ms / 1e3)
        Xr = Xr[:, :, sel]
        t = tt[sel]
        d = np.transpose(Xr, (0, 2, 1))                      # (trials, T, N)

        # Rebuild u as an onset impulse on the reduced grid.
        onset = int(np.argmin(np.abs(t)))
        T, M = len(t), len(self._site_keep)
        u = np.zeros((len(self._trial_keep), T, M), dtype=np.float64)
        trial_site = self.events["stim_site_idx"].to_numpy()[self._trial_keep]
        current = self.events["current_mA"].to_numpy()[self._trial_keep]
        for i, (s, cur) in enumerate(zip(trial_site, current)):
            u[i, onset, self._site_full_to_kept[int(s)]] = float(cur)
        if self.normalize_u:
            mx = u.max()
            if mx > 0:
                u /= mx
        return NeuralActivity(d=d, t=t, u=u)

    def get_strata(self) -> np.ndarray:
        """Per-trial stratification labels (stim site index) for get_dataloaders_real."""
        return self.events["stim_site_idx"].to_numpy()[self._trial_keep]

    def get_stim_channel_mask(self) -> np.ndarray:
        """(n_trials, N) bool: the two contacts of each trial's stim site.

        Their own recordings saturate around the pulse. Filtered-out contacts are absent.
        """
        trial_site = self.events["stim_site_idx"].to_numpy()[self._trial_keep]
        return self._stim_contacts(trial_site)

    def build_loss_mask(self, t: np.ndarray) -> tuple[torch.Tensor, dict]:
        """(n_trials, T, N) bool mask (True = unobserved) and stats.

        Masks each trial's own stim contacts for the whole trial; t is the get_trials() time axis.
        """
        stim_ch = torch.from_numpy(self.get_stim_channel_mask())      # (trials, N)
        mask = stim_ch.unsqueeze(1).expand(-1, len(t), -1).clone()
        return mask, {"n_stim_ch": int(stim_ch.sum().item())}
