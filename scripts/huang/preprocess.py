"""Preprocess one Huang 2023 subject: continuous MEF -> epoched iSSM arrays (needs pymef, mne).

Per stimulation pulse, following the dataset authors' saveBPCs.m signal steps:
  1. Read the [-pre, +post] s epoch (uV).
  2. Interpolate the stim artefact (samples are replaced, not dropped).
  3. CAR over good, non-stimulated SEEG channels (standard or adjusted).
  4. Remove 60 Hz line noise and harmonics.
  5. Subtract the pre-stim baseline median.
  6. Build u: one channel per stim site, a single-sample impulse of the current.

Outputs in data/huang_2023/preproc/<subject>/:
  epochs.npz        X[trial, ch, t] (uV), u[trial, t, M], tt[t], fs
  channels.csv      kept SEEG channels (name, orig_idx, seeg_idx, status, good)
  stim_sites.csv    M sites (site_idx, name, members, member seeg idxs, n_reps, current)
  events.csv        per-trial (trial_idx, stim_site_idx, onset_sample, current_mA, status)
  preproc_meta.json parameters, git SHA, runtime and exclusion counts
  qc_stages.npz     example epochs at each pipeline stage

Usage:
  python -m scripts.huang.preprocess --subject sub-2
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.huang import mef_io, signal as hsig


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _peak_rss_gb() -> float:
    # ru_maxrss is in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


class _Log:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __call__(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


# ----------------------------------------------------------------------------
# Stim-site table
# ----------------------------------------------------------------------------

def build_stim_sites(events: pd.DataFrame, name_to_seeg: dict[str, int]
                     ) -> pd.DataFrame:
    """One row per unique electrical_stimulation_site, in first-occurrence order.

    member*_seeg is -1 when the member is not a kept SEEG channel (it is then not
    excluded from the CAR).
    """
    rows = []
    for site_idx, name in enumerate(dict.fromkeys(events["electrical_stimulation_site"])):
        sub = events[events["electrical_stimulation_site"] == name]
        members = name.split("-")
        m1, m2 = (members + ["", ""])[:2]
        s1 = name_to_seeg.get(m1, -1)
        s2 = name_to_seeg.get(m2, -1)
        if s1 == -1 or s2 == -1:
            print(f"  [warn] stim site {name!r}: member(s) not in SEEG channels "
                  f"(seeg idx {s1}, {s2}); excluded from CAR masking.", flush=True)
        rows.append({
            "site_idx": site_idx, "site_name": name,
            "member1": m1, "member2": m2,
            "member1_seeg": s1, "member2_seeg": s2,
            "n_reps": int(len(sub)),
            "current_mA": float(sub["current_mA"].iloc[0]),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main preprocessing
# ----------------------------------------------------------------------------

def preprocess_subject(args) -> None:
    t0 = time.perf_counter()
    log = _Log(args.verbose)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.outdir) / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = mef_io.resolve_subject_paths(raw_dir, args.subject)
    channels = mef_io.load_channels(paths)
    events_all = mef_io.load_events(paths)
    electrodes = mef_io.load_electrodes(paths)

    # --- Channel selection + anatomy ----------------------------------------
    seeg = mef_io.select_seeg_channels(channels)
    seeg = mef_io.merge_electrode_anatomy(seeg, electrodes)
    name_to_seeg = dict(zip(seeg["name"], seeg["seeg_idx"]))
    good_mask = seeg["good"].to_numpy()
    n_ch = len(seeg)
    n_loc = int(seeg["has_location"].sum())
    log(f"[channels] {len(channels)} total -> {n_ch} SEEG "
        f"({int(good_mask.sum())} good, {int((~good_mask).sum())} bad excluded from CAR; "
        f"{n_loc} with anatomical location, {n_ch - n_loc} without)")

    # --- Event selection -----------------------------------------------------
    n0 = len(events_all)
    ev = events_all.copy()
    ev = ev[ev["sample_start"].notna()]
    n_nosamp = n0 - len(ev)
    ev = ev[ev["status"].str.lower() == "good"]
    n_badstatus = (n0 - n_nosamp) - len(ev)
    currents = set(args.currents)
    ev = ev[ev["current_mA"].isin(currents)]
    n_curr = (n0 - n_nosamp - n_badstatus) - len(ev)
    ev = ev.reset_index(drop=False).rename(columns={"index": "orig_event_row"})
    log(f"[events] {n0} total -> {len(ev)} kept "
        f"(dropped: {n_nosamp} no-sample, {n_badstatus} bad-status, "
        f"{n_curr} off-current; keeping currents {sorted(currents)} mA)")

    # --- Stim-site table -----------------------------------------------------
    sites = build_stim_sites(ev, name_to_seeg)
    sites = mef_io.annotate_stim_site_anatomy(sites, seeg)
    site_to_idx = dict(zip(sites["site_name"], sites["site_idx"]))
    M = len(sites)
    log(f"[sites] {M} unique stim sites (reps: median {int(sites['n_reps'].median())}, "
        f"min {int(sites['n_reps'].min())}, max {int(sites['n_reps'].max())})")

    # --- Epoch geometry ------------------------------------------------------
    fs_expected = 2048.0
    pre_n = int(round(args.pre * fs_expected))
    post_n = int(round(args.post * fs_expected))
    L = pre_n + post_n
    onset_idx = pre_n  # tt == 0 here
    n_trials = len(ev)

    # QC: pick example events = first rep of the sites with the most reps.
    qc_site_order = sites.sort_values("n_reps", ascending=False)["site_name"].tolist()
    qc_event_rows = []
    for sname in qc_site_order[: args.n_qc_examples]:
        qc_event_rows.append(int(ev.index[ev["electrical_stimulation_site"] == sname][0]))
    qc_stages_names = ["raw", "interp", "car", "linenoise", "baseline"]
    qc_stages = {r: {} for r in qc_event_rows}

    # --- Allocate outputs ----------------------------------------------------
    X = np.zeros((n_trials, n_ch, L), dtype=np.float32)
    u = np.zeros((n_trials, L, M), dtype=np.float32)
    tt = (np.arange(L) - onset_idx) / fs_expected

    ev_records = []
    car_excl = {"n_var": [], "n_resp": []}

    with mef_io.MefReader(paths.mefd, seeg["name"].tolist(),
                          password=args.password) as reader:
        fs = reader.fs
        if abs(fs - fs_expected) > 1e-6:
            raise ValueError(f"Unexpected sampling rate {fs} (expected {fs_expected}).")
        log(f"[mef] opened {paths.mefd.name}: {n_ch} ch, "
            f"{reader.n_samples} samples @ {fs:g} Hz")
        log(f"[epochs] window [-{args.pre}, +{args.post}] s -> L={L} samples; "
            f"processing {n_trials} epochs ...")

        t_read = t_car = t_ln = t_base = t_art = 0.0
        for i, (_, row) in enumerate(ev.iterrows()):
            sample_start = int(row["sample_start"])
            site_name = row["electrical_stimulation_site"]
            site_idx = site_to_idx[site_name]
            current = float(row["current_mA"])

            # stim channel mask for this event
            stim_seeg = [s for s in (sites.loc[site_idx, "member1_seeg"],
                                     sites.loc[site_idx, "member2_seeg"]) if s >= 0]
            stim_mask = np.zeros(n_ch, dtype=bool)
            stim_mask[stim_seeg] = True
            ref_mask = good_mask & ~stim_mask

            tic = time.perf_counter()
            epoch = reader.read_window(sample_start - pre_n, sample_start + post_n)
            t_read += time.perf_counter() - tic
            raw = epoch.copy()

            # Interpolate the artefact first: the transient otherwise corrupts the
            # spectrum-fit line-noise estimate and injects a 60 Hz comb.
            tic = time.perf_counter()
            epoch = hsig.interpolate_artifact(
                epoch, onset_idx, fs, pre_ms=args.artifact_pre_ms,
                post_ms=args.artifact_post_ms, method=args.artifact_method)
            t_art += time.perf_counter() - tic
            stage_interp = epoch.copy() if i in qc_stages else None

            tic = time.perf_counter()
            if args.car == "adjusted":
                epoch, info = hsig.adjusted_car(epoch, tt, ref_mask)
                car_excl["n_var"].append(info["n_var_excluded"])
                car_excl["n_resp"].append(info["n_resp_excluded"])
            else:
                epoch = hsig.common_average_reference(epoch, ref_mask)
            t_car += time.perf_counter() - tic
            stage_car = epoch.copy() if i in qc_stages else None

            tic = time.perf_counter()
            epoch = hsig.remove_line_noise(epoch, fs, line_freq=args.line_freq)
            t_ln += time.perf_counter() - tic
            stage_ln = epoch.copy() if i in qc_stages else None

            tic = time.perf_counter()
            epoch = hsig.subtract_baseline(epoch, tt, window=tuple(args.baseline))
            t_base += time.perf_counter() - tic

            X[i] = epoch.astype(np.float32)
            u[i, onset_idx, site_idx] = current

            if i in qc_stages:
                qc_stages[i] = {
                    "raw": raw, "interp": stage_interp, "car": stage_car,
                    "linenoise": stage_ln, "baseline": epoch.copy(),
                    "site_name": site_name, "site_idx": site_idx,
                    "stim_seeg": np.array(stim_seeg, dtype=int),
                }

            ev_records.append({
                "trial_idx": i, "stim_site_idx": site_idx, "site_name": site_name,
                "onset_sample": sample_start, "current_mA": current,
                "status": row["status"], "orig_event_row": int(row["orig_event_row"]),
            })

            if args.verbose and (i + 1) % 50 == 0:
                log(f"    {i + 1}/{n_trials} epochs  "
                    f"(read {t_read:.1f}s, car {t_car:.1f}s, ln {t_ln:.1f}s, "
                    f"base {t_base:.1f}s, art {t_art:.1f}s)")

    log(f"[timing] read {t_read:.1f}s | car {t_car:.1f}s | linenoise {t_ln:.1f}s | "
        f"baseline {t_base:.1f}s | artifact {t_art:.1f}s")

    # --- Save ----------------------------------------------------------------
    np.savez(out_dir / "epochs.npz", X=X, u=u, tt=tt, fs=np.float32(fs_expected))

    chan_cols = (["name", "orig_idx", "seeg_idx", "status", "good", "has_location"]
                 + mef_io._ANATOMY_COLS)
    seeg[chan_cols].to_csv(out_dir / "channels.csv", index=False)
    sites.to_csv(out_dir / "stim_sites.csv", index=False)
    pd.DataFrame(ev_records).to_csv(out_dir / "events.csv", index=False)

    # QC stages: [n_ex, n_stage, ch, t]
    qc_arr = np.stack([
        np.stack([qc_stages[r][s] for s in qc_stages_names], axis=0)
        for r in qc_event_rows
    ], axis=0).astype(np.float32)
    np.savez(
        out_dir / "qc_stages.npz",
        stages=qc_arr, tt=tt,
        stage_names=np.array(qc_stages_names),
        site_names=np.array([qc_stages[r]["site_name"] for r in qc_event_rows]),
        site_idxs=np.array([qc_stages[r]["site_idx"] for r in qc_event_rows]),
        stim_seeg=np.array([
            np.pad(qc_stages[r]["stim_seeg"], (0, 2 - len(qc_stages[r]["stim_seeg"])),
                   constant_values=-1)
            for r in qc_event_rows]),
    )

    runtime = time.perf_counter() - t0
    meta = {
        "subject": args.subject, "created": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "fs": fs_expected, "N_channels": n_ch, "N_good": int(good_mask.sum()),
        "N_located": n_loc,
        "M_sites": M, "n_trials": n_trials, "epoch_samples": L,
        "onset_idx": onset_idx, "X_shape": list(X.shape), "u_shape": list(u.shape),
        "params": {
            "pre_s": args.pre, "post_s": args.post,
            "baseline_s": list(args.baseline),
            "artifact_pre_ms": args.artifact_pre_ms,
            "artifact_post_ms": args.artifact_post_ms,
            "artifact_method": args.artifact_method, "car": args.car,
            "line_freq": args.line_freq, "currents_mA": sorted(set(args.currents)),
        },
        "exclusions": {
            "events_no_sample": int(n_nosamp), "events_bad_status": int(n_badstatus),
            "events_off_current": int(n_curr),
            "channels_bad": int((~good_mask).sum()),
        },
        "adjusted_car_mean_excluded": (
            {"by_variance": float(np.mean(car_excl["n_var"])),
             "by_response": float(np.mean(car_excl["n_resp"]))}
            if args.car == "adjusted" and car_excl["n_var"] else None),
        "runtime_s": round(runtime, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 2),
    }
    with open(out_dir / "preproc_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"[done] {args.subject}: X{list(X.shape)} u{list(u.shape)} -> {out_dir}")
    print(f"[summary] runtime {runtime:.1f}s | peak RSS {_peak_rss_gb():.2f} GB | "
          f"N={n_ch} M={M} trials={n_trials}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, help="e.g. sub-2")
    p.add_argument("--raw_dir", default="data/huang_2023/raw")
    p.add_argument("--outdir", default="data/huang_2023/preproc")
    p.add_argument("--pre", type=float, default=1.0, help="seconds before pulse")
    p.add_argument("--post", type=float, default=2.0, help="seconds after pulse")
    p.add_argument("--baseline", type=float, nargs=2, default=[-0.5, -0.05],
                   help="baseline window (s) for median subtraction")
    p.add_argument("--artifact_pre_ms", type=float, default=2.0,
                   help="ms before the pulse to interpolate (pre-stim is clean)")
    p.add_argument("--artifact_post_ms", type=float, default=15.0,
                   help="ms after the pulse to interpolate (covers the ringing tail)")
    p.add_argument("--artifact_method", choices=["pchip", "crowther", "spline"],
                   default="pchip")
    p.add_argument("--car", choices=["standard", "adjusted"], default="standard",
                   help="standard global CAR or Huang's per-trial adjusted CAR")
    p.add_argument("--line_freq", type=float, default=60.0)
    p.add_argument("--currents", type=float, nargs="+", default=[4.0, 6.0],
                   help="stimulation currents (mA) to keep")
    p.add_argument("--n_qc_examples", type=int, default=3,
                   help="number of example events saved at each stage for QC plots")
    p.add_argument("--password", default=None, help="MEF password (default: none)")
    p.add_argument("--verbose", dest="verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false",
                   help="suppress per-step status output")
    return p


def main() -> None:
    args = build_parser().parse_args()
    preprocess_subject(args)


if __name__ == "__main__":
    main()
