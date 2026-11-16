"""Flag residual stim-artefact contamination in a preprocessed subject.

A residual onset transient can survive the artefact interpolation. A sigma cutoff
alone also flags real CCEP N1 responses, so a cell (trial, channel) is masked only if
its onset peak exceeds entry_sigma pre-stim SDs AND its |X| correlates above
entry_corr with an artefact template built from the worst trials (N1 correlates ~0).

Writes stim_artefact{,_corr,_mask,_trials}.csv (wide: trial_idx + one column per
channel), stim_artefact_template.npz and stim_artefact_config.json next to the
preprocessed arrays; training reads stim_artefact_mask.csv. Needs no pymef/mne.

Usage:
  python -m scripts.huang.stim_artefact --subject sub-2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.huang.signal import artefact_template, template_correlation


def _stim_channels(sites: pd.DataFrame) -> set[int]:
    """SEEG indices of every electrode that is a member of any stim site."""
    s: set[int] = set()
    for col in ("member1_seeg", "member2_seeg"):
        s.update(int(v) for v in sites[col].to_numpy() if v >= 0)
    return s


def score_subject(args) -> None:
    d = Path(args.preproc_dir) / args.subject
    npz = np.load(d / "epochs.npz")
    X, tt, fs = npz["X"], npz["tt"], float(npz["fs"])          # X: (trials, ch, t)
    channels = pd.read_csv(d / "channels.csv")
    sites = pd.read_csv(d / "stim_sites.csv")
    good = channels["good"].to_numpy()
    located = channels["has_location"].to_numpy().astype(bool)
    names = channels["name"].tolist()

    onset = int(np.argmin(np.abs(tt)))
    w = int(round(args.onset_ms * 1e-3 * fs))
    pre_end = onset - int(round(0.05 * fs))                    # pre-stim [start, -50 ms)
    onset_pk = np.abs(X[:, :, onset - w:onset + w + 1]).max(axis=2)   # (trials, ch)
    pre_std = X[:, :, :pre_end].std(axis=2)                    # (trials, ch)
    score = onset_pk / np.where(pre_std > 0, pre_std, np.nan)  # sigmas above baseline

    # trial-level score: fraction of good channels with a real onset spike
    spiky = (score > args.entry_sigma) & (onset_pk > args.entry_uV) & good[None, :]
    frac = spiky.sum(axis=1) / good.sum()
    drop = (frac > args.trial_frac)

    # Template: mean |X| over good, located, non-stim channels of the two worst trials.
    keep = good & located
    keep[list(_stim_channels(sites))] = False
    keep_idx = np.where(keep)[0]
    tmpl_trials = np.argsort(frac)[::-1][:2]                   # top-2 spiking trials
    window = (-args.corr_pre_ms * 1e-3, args.corr_post_ms * 1e-3)
    template, template_sd, sel = artefact_template(X, tt, tmpl_trials, keep_idx, window)
    corr = template_correlation(X, sel, template)             # (trials, ch)

    mask = (score > args.entry_sigma) & (corr > args.entry_corr)   # NaN sigma -> False

    def _wide(arr, dtype):
        out = pd.DataFrame(arr.astype(dtype), columns=names)
        out.insert(0, "trial_idx", np.arange(X.shape[0]))
        return out

    _wide(score, np.float32).to_csv(d / "stim_artefact.csv", index=False)
    _wide(corr, np.float32).to_csv(d / "stim_artefact_corr.csv", index=False)
    _wide(mask, np.int8).to_csv(d / "stim_artefact_mask.csv", index=False)

    pd.DataFrame({
        "trial_idx": np.arange(X.shape[0]),
        "frac_ch_spiking": frac.astype(np.float32),
        "n_ch_spiking": spiky.sum(axis=1).astype(int),
        "drop": drop.astype(int),
    }).to_csv(d / "stim_artefact_trials.csv", index=False)

    np.savez(d / "stim_artefact_template.npz",
             template=template, template_sd=template_sd,
             tmpl_trials=tmpl_trials, t_ms=tt[sel] * 1e3, window_s=np.array(window))

    config = {
        "subject": args.subject,
        "onset_ms": args.onset_ms,
        "entry_sigma": args.entry_sigma,
        "entry_uV": args.entry_uV,
        "entry_corr": args.entry_corr,
        "corr_pre_ms": args.corr_pre_ms,
        "corr_post_ms": args.corr_post_ms,
        "trial_frac": args.trial_frac,
    }
    (d / "stim_artefact_config.json").write_text(json.dumps(config, indent=2))

    # --- summary ---
    flat_sig = score[:, good].ravel()
    flat_sig = flat_sig[np.isfinite(flat_sig)]
    n_sigma_only = int(np.sum((score[:, good] > args.entry_sigma)))
    n_joint = int(mask[:, good].sum())
    corr_flagged = corr[:, good][score[:, good] > args.entry_sigma]
    corr_flagged = corr_flagged[np.isfinite(corr_flagged)]
    print(f"[stim_artefact] {args.subject}: {X.shape[0]} trials x {X.shape[1]} channels "
          f"(onset +/-{args.onset_ms:g} ms peak / pre-stim std)")
    print(f"  per-entry sigma (good ch): median={np.median(flat_sig):.2f}  "
          f"95th={np.percentile(flat_sig, 95):.2f}  max={flat_sig.max():.1f}")
    print(f"  template trials {tmpl_trials.tolist()} over [{window[0]*1e3:g}, "
          f"{window[1]*1e3:g}] ms; sigma>{args.entry_sigma:g} cells have corr "
          f"median={np.median(corr_flagged):.2f}")
    print(f"  joint mask (sigma>{args.entry_sigma:g} & corr>{args.entry_corr:g}): "
          f"{n_joint} cells vs {n_sigma_only} sigma-only "
          f"({100 * n_joint / max(n_sigma_only, 1):.0f}% kept)")
    ndrop = int(drop.sum())
    print(f"  trial-level frac spiking: median={np.median(frac):.3f}  max={frac.max():.3f}")
    print(f"  trials flagged for drop (frac>{args.trial_frac:g}): {ndrop}/{X.shape[0]}")
    if ndrop:
        order = np.argsort(frac)[::-1][:ndrop]
        print("   ", ", ".join(f"trial {int(i)} ({frac[i]:.2f})" for i in order))
    print(f"  saved stim_artefact{{,_corr,_mask,_trials}}.csv, "
          f"stim_artefact_template.npz, stim_artefact_config.json in {d}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, help="e.g. sub-2")
    p.add_argument("--preproc_dir", default="data/huang_2023/preproc")
    p.add_argument("--onset_ms", type=float, default=2.0,
                   help="half-width (ms) of the onset window scanned for the residual peak")
    p.add_argument("--entry_sigma", type=float, default=5.0,
                   help="per-entry flag: onset peak > this many pre-stim sigmas")
    p.add_argument("--entry_uV", type=float, default=100.0,
                   help="per-entry flag: absolute floor (uV) so tiny-baseline channels don't trip")
    p.add_argument("--entry_corr", type=float, default=0.5,
                   help="per-entry mask: correlation to the artefact template above which a "
                        "high-sigma cell is a genuine artefact (not real CCEP N1)")
    p.add_argument("--corr_pre_ms", type=float, default=10.0,
                   help="correlation-window pre-onset extent (ms)")
    p.add_argument("--corr_post_ms", type=float, default=40.0,
                   help="correlation-window post-onset extent (ms)")
    p.add_argument("--trial_frac", type=float, default=0.10,
                   help="trial-level drop: fraction of good channels spiking above which to drop")
    return p


def main() -> None:
    score_subject(build_parser().parse_args())


if __name__ == "__main__":
    main()
