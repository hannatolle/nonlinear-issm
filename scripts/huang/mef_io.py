"""MEF 3.0 reading and BIDS sidecar loading for the Huang 2023 dataset (requires pymef).

Raw-data facts (ds004457):
  - One continuous run per subject (task-ccep_run-01_ieeg.mefd) at 2048 Hz.
  - read_ts_channels_sample returns raw integer counts; multiply by the per-channel
    ufact to get microvolts.
  - MEF channel order is alphabetical and does not match channels.tsv, so channels
    are always addressed by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pymef.mef_session import MefSession


# ----------------------------------------------------------------------------
# BIDS path resolution + sidecar loading
# ----------------------------------------------------------------------------

@dataclass
class SubjectPaths:
    """Resolved BIDS paths for one subject's single CCEP run."""

    subject: str          # e.g. "sub-2"
    ieeg_dir: Path
    mefd: Path
    channels_tsv: Path
    events_tsv: Path
    electrodes_tsv: Path
    ieeg_json: Path


def resolve_subject_paths(raw_dir: Path, subject: str,
                          session: str = "ses-ieeg01",
                          task: str = "task-ccep",
                          run: str = "run-01") -> SubjectPaths:
    """Locate the MEF session and sidecars for subject under raw_dir."""
    raw_dir = Path(raw_dir)
    ieeg_dir = raw_dir / subject / session / "ieeg"
    base = f"{subject}_{session}_{task}_{run}"
    paths = SubjectPaths(
        subject=subject,
        ieeg_dir=ieeg_dir,
        mefd=ieeg_dir / f"{base}_ieeg.mefd",
        channels_tsv=ieeg_dir / f"{base}_channels.tsv",
        events_tsv=ieeg_dir / f"{base}_events.tsv",
        electrodes_tsv=ieeg_dir / f"{subject}_{session}_electrodes.tsv",
        ieeg_json=ieeg_dir / f"{base}_ieeg.json",
    )
    for f in (paths.mefd, paths.channels_tsv, paths.events_tsv):
        if not f.exists():
            raise FileNotFoundError(f"Expected BIDS file not found: {f}")
    return paths


def load_channels(paths: SubjectPaths) -> pd.DataFrame:
    """Load channels.tsv and add a 0-indexed orig_idx column (BIDS order)."""
    df = pd.read_csv(paths.channels_tsv, sep="\t", dtype=str)
    df["orig_idx"] = np.arange(len(df), dtype=int)
    return df


def load_events(paths: SubjectPaths) -> pd.DataFrame:
    """Load events.tsv with numeric sample_start and current_mA."""
    df = pd.read_csv(paths.events_tsv, sep="\t", dtype=str)
    df["sample_start"] = pd.to_numeric(df["sample_start"], errors="coerce")
    # "6.0 mA" -> 6.0
    df["current_mA"] = (
        df["electrical_stimulation_current"].str.extract(r"([\d.]+)").astype(float)
    )
    return df


def load_electrodes(paths: SubjectPaths) -> pd.DataFrame | None:
    """Load electrodes.tsv (positions, seizure_zone) if present, else None."""
    if not paths.electrodes_tsv.exists():
        return None
    return pd.read_csv(paths.electrodes_tsv, sep="\t", dtype=str)


_ANATOMY_COLS = ["x", "y", "z", "hemisphere", "seizure_zone",
                 "Destrieux_label", "Destrieux_label_text"]


def merge_electrode_anatomy(seeg: pd.DataFrame,
                            electrodes: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join anatomy from electrodes.tsv onto the SEEG table.

    Only some electrodes are localised; the rest get NaN columns and
    has_location == False. Coordinates are coerced to float (n/a -> NaN).
    """
    seeg = seeg.copy()
    if electrodes is None:
        for c in _ANATOMY_COLS:
            seeg[c] = np.nan
        seeg["has_location"] = False
        return seeg

    cols = ["name"] + [c for c in _ANATOMY_COLS if c in electrodes.columns]
    el = electrodes[cols].copy()
    for c in ("x", "y", "z"):
        if c in el.columns:
            el[c] = pd.to_numeric(el[c], errors="coerce")
    seeg = seeg.merge(el, on="name", how="left")
    seeg["has_location"] = (
        seeg["x"].notna() if "x" in seeg.columns else False)
    return seeg


def annotate_stim_site_anatomy(sites: pd.DataFrame,
                               seeg: pd.DataFrame) -> pd.DataFrame:
    """Add region, hemisphere and seizure_zone of each site's first member, and mean member xyz."""
    sites = sites.copy()
    by_name = seeg.set_index("name")
    region, hemi, soz, xs, ys, zs = [], [], [], [], [], []
    for _, row in sites.iterrows():
        members = [m for m in (row["member1"], row["member2"]) if m in by_name.index]
        if members:
            r = by_name.loc[members[0]]
            region.append(r.get("Destrieux_label_text", np.nan))
            hemi.append(r.get("hemisphere", np.nan))
            soz.append(r.get("seizure_zone", np.nan))
            xs.append(np.nanmean([by_name.loc[m].get("x", np.nan) for m in members]))
            ys.append(np.nanmean([by_name.loc[m].get("y", np.nan) for m in members]))
            zs.append(np.nanmean([by_name.loc[m].get("z", np.nan) for m in members]))
        else:
            region.append(np.nan); hemi.append(np.nan); soz.append(np.nan)
            xs.append(np.nan); ys.append(np.nan); zs.append(np.nan)
    sites["region"] = region
    sites["hemisphere"] = hemi
    sites["seizure_zone"] = soz
    sites["x"], sites["y"], sites["z"] = xs, ys, zs
    return sites


def select_seeg_channels(channels: pd.DataFrame) -> pd.DataFrame:
    """Return the SEEG rows with seeg_idx (channel axis of the saved X) and a good flag."""
    seeg = channels[channels["type"].str.upper() == "SEEG"].copy()
    seeg = seeg.reset_index(drop=True)
    seeg["seeg_idx"] = np.arange(len(seeg), dtype=int)
    seeg["good"] = (seeg["status"].str.lower() == "good")
    return seeg


# ----------------------------------------------------------------------------
# MEF reading
# ----------------------------------------------------------------------------

class MefReader:
    """pymef wrapper reading SEEG channels by name, in microvolts, one window at a time.

    Use as a context manager.
    """

    def __init__(self, mefd_path: Path, channel_names: list[str],
                 password: str | None = None):
        self.mefd_path = str(mefd_path)
        self.channel_names = list(channel_names)
        self.session = MefSession(self.mefd_path, password)

        info = {d["name"]: d for d in self.session.read_ts_channel_basic_info()}
        missing = [c for c in self.channel_names if c not in info]
        if missing:
            raise KeyError(f"Channels not found in MEF session: {missing}")

        self.ufact = np.array(
            [float(np.ravel(info[c]["ufact"])[0]) for c in self.channel_names]
        )
        self.nsamp = np.array(
            [int(np.ravel(info[c]["nsamp"])[0]) for c in self.channel_names]
        )
        self.fs = float(np.ravel(info[self.channel_names[0]]["fsamp"])[0])
        # All SEEG channels in this dataset share one continuous run length.
        self.n_samples = int(self.nsamp.min())

    def __enter__(self) -> "MefReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def read_window(self, start: int, stop: int) -> np.ndarray:
        """Read samples [start, stop) as a (n_channels, stop - start) float32 array in uV.

        Out-of-range samples are zero-padded so epochs near the recording edges keep a
        fixed length.
        """
        n = stop - start
        out = np.zeros((len(self.channel_names), n), dtype=np.float32)

        rs = max(start, 0)
        re = min(stop, self.n_samples)
        if re <= rs:
            return out  # window entirely outside the recording

        raw = self.session.read_ts_channels_sample(self.channel_names, [rs, re])
        raw = np.asarray([np.asarray(r, dtype=np.float64) for r in raw])  # (ch, re-rs)
        raw *= self.ufact[:, None]  # counts -> microvolts

        out[:, (rs - start):(rs - start) + (re - rs)] = raw.astype(np.float32)
        return out
