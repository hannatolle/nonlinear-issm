"""
Virtual-stimulation analysis for an iSSM trained on the Huang (2023) SEEG dataset.

The trained b_net maps synthesized bipolar electrode pairs to virtual B columns;
paired impulse-response rollouts give per-pair excitability readouts, which are
compared between epileptogenic and healthy seizure zones.
"""

import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from src.datasets.huang import _ZONE_NAME, _is_white_matter, _zone_severity
from src.models import FlexibleLDS
from src.utils.io import add_root, load_component_from_run, load_json
from .base import Experiment
from .eval_real_issm import _load_standardization, _load_zero_fill_marker
from .train_huang_issm import TrainHuangISSM


# Per-pair readouts compared between epileptogenic and healthy pairs.
_SEIZURE_READOUTS = ["gain", "resp_peak", "resp_auc", "plateau_uv"]


def _parse_contact(name: str):
    """Split an SEEG contact name into (lead, index), e.g. 'RMG5' -> ('RMG', 5)."""
    m = re.match(r"^([A-Za-z]+?)(\d+)$", str(name))
    return (m.group(1), int(m.group(2))) if m else None


@torch.no_grad()
def _virtual_impulse_step0(dynamics, x0, b_col, amp):
    """First post-impulse latent state for a virtual single-site pulse.

    Reproduces TransitionModel.step's stim semantics for a b_col that is NOT one
    of the trained B columns: driven latents (|B|>=B_threshold) are clamped to
    amp*b_col under hard intervention, or added to the autonomous mean otherwise.
    """
    mu = dynamics._mu(x0, None)  # (B, D) autonomous next-state mean
    b_clean = torch.where(b_col.abs() < dynamics.B_threshold,
                          torch.zeros_like(b_col), b_col)
    driven = (b_clean != 0)                     # (D,) which latents the pulse drives
    inputs = (amp * b_col).to(mu.dtype).expand_as(mu)
    if dynamics.interventional:
        return torch.where(driven, inputs, mu)
    return mu + torch.where(driven, inputs, torch.zeros_like(inputs))


def _response_metrics(dY: np.ndarray, dt: float, plateau_start_ms: float) -> dict:
    """Excitability readouts from a paired (perturbed-baseline) rollout.

    dY: (n_ics, T, N) observation deviation. Readouts are in observation space,
    so they are invariant to the latent scale.

    resp_peak captures the transient, plateau_uv the mean response from
    plateau_start_ms on, and resp_auc integrates both. The rollout is not
    constrained beyond the trained trial window, so a long virtual_stim_T
    contaminates the tail.
    """
    dy_norm = np.linalg.norm(dY, axis=2)   # (n_ics, T)
    dev = dy_norm.mean(axis=0)             # (T,) mean response magnitude
    t_ms = np.arange(len(dev)) * dt * 1e3
    late = t_ms >= plateau_start_ms
    return {
        "resp_peak":  float(dev.max()),
        "resp_auc":   float(dev.sum() * dt),
        "plateau_uv": float(dev[late].mean()) if late.any() else float("nan"),
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled-SD) for a vs b; positive = a (epileptogenic) larger."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


class AttractorHuangISSM(Experiment):
    """Virtual stimulation of a single model trained on the Huang (2023) SEEG dataset.

    The trained b_net (SiteEmbeddingB) maps a site's xyz to a column of the
    intervention matrix B, so we can stimulate synthesized bipolar electrode
    pairs with no data. Pairs are grouped by clinical seizure_zone (never seen
    in training) and compared on B-column gain and on the impulse response
    magnitude (resp_peak), persistence (plateau_uv) and total energy (resp_auc),
    all in y-space. Results go to output_dir (default: run_dir/attractor/).
    """

    @classmethod
    def preprocess_config(cls, config: dict) -> dict:
        settings = config.get("settings", {})
        if settings.get("use_default_output_dir", True):
            base_dir = settings.get("run_dir", "")
            if base_dir:
                default_output = os.path.join(base_dir, "attractor")
                provided = config.get("output_dir", "./outputs/default")
                if provided not in ("./outputs", "./outputs/default", default_output):
                    print(
                        f"Warning: use_default_output_dir=True; "
                        f"ignoring output_dir={provided!r}, using {default_output!r}"
                    )
                config = {**config, "output_dir": default_output}
        return config

    def check_config(self):
        s = self.fetch_settings()
        run_dir = add_root(s.run_dir)
        if not os.path.isdir(run_dir):
            raise ValueError(f"run_dir is not a directory: {run_dir}")
        for name in ["dynamics", "emission", "likelihood", "recognition"]:
            pth = os.path.join(run_dir, f"{name}.pth")
            if not os.path.isfile(pth):
                raise ValueError(f"Missing weight file: {pth}")
        if not os.path.isfile(os.path.join(run_dir, "config.json")):
            raise ValueError(f"Missing config.json in {run_dir}")

    # ------------------------------------------------------------------
    # Model and data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dynamics_with_bnet(run_dir: str):
        """Load dynamics and wire in the run's b_net, if any, so B_effective is
        available."""
        dynamics = load_component_from_run(run_dir, "dynamics")
        run_cfg = load_json(os.path.join(run_dir, "config.json"))
        if "b_net" in run_cfg.get("components", {}):
            dynamics.set_B_net(load_component_from_run(run_dir, "b_net"))
        return dynamics

    def _load_model(self, run_dir: str) -> Tuple[FlexibleLDS, object]:
        joint = FlexibleLDS(
            dynamics=self._load_dynamics_with_bnet(run_dir),
            emission=load_component_from_run(run_dir, "emission"),
            likelihood=load_component_from_run(run_dir, "likelihood"),
        ).eval()
        rec = load_component_from_run(run_dir, "recognition")
        rec.set_B_from_dynamics(joint.dynamics)
        return joint, rec.eval()

    @staticmethod
    def _load_data(train_config: dict):
        """Rebuild the training trial pool.

        Returns y (B, T, N), u (B, T, M), ts_vec (T,) in seconds, the
        stim-contact observation mask and the loader.
        """
        loader = TrainHuangISSM._build_loader(train_config["settings"])
        act = loader.get_trials()
        y = torch.tensor(act.d, dtype=torch.float32)
        u = torch.tensor(act.u, dtype=torch.float32)
        ts_vec = act.t.astype(np.float64)
        obs_mask, _ = loader.build_loss_mask(act.t)
        return y, u, ts_vec, obs_mask, loader

    @staticmethod
    @torch.no_grad()
    def _encode_trajectories(
        recognition,
        y: torch.Tensor,
        u: torch.Tensor,
        device: torch.device,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode observations to posterior latent mean trajectories.

        Returns:
            x_enc: (n_trials, T, D) numpy array on CPU.
        """
        recognition = recognition.to(device).eval()
        n_trials = y.shape[0]
        chunks = []
        for start in range(0, n_trials, batch_size):
            y_b = y[start:start + batch_size].to(device)
            u_b = u[start:start + batch_size].to(device)
            chunks.append(recognition.encode_mean(y_b, u_b).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def run(self):
        s = self.fetch_settings()
        os.makedirs(self.output_dir, exist_ok=True)
        device = self.get_device()

        run_dir = add_root(s.run_dir)
        train_config = load_json(os.path.join(run_dir, "config.json"))
        print(f"AttractorHuangISSM: loading run from {run_dir}")
        joint, rec = self._load_model(run_dir)

        print("  Loading data...")
        y, u, ts_vec, obs_mask, loader = self._load_data(train_config)

        y_mean, y_std = _load_standardization(run_dir)
        if y_mean is not None:
            y = (y - y_mean) / y_std
        # Match training: zero-fill masked cells iff training did so.
        if _load_zero_fill_marker(run_dir):
            y = y.masked_fill(obs_mask, 0.0)

        print("  Encoding trajectories...")
        x_enc_np = self._encode_trajectories(rec, y, u, device)

        self._run_virtual_stimulation(
            joint, loader, x_enc_np, ts_vec, train_config, self.output_dir, device)

        meta = {"seed": int(train_config.get("seed", 0)), "settings": dict(s)}
        with open(os.path.join(self.output_dir, "attractor_config.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Virtual-stimulation analysis complete. Results in {self.output_dir}")

    # ------------------------------------------------------------------
    # Seizure-zone virtual-stimulation analysis
    # ------------------------------------------------------------------

    def _run_virtual_stimulation(
        self,
        joint: FlexibleLDS,
        loader,
        x_enc_np: np.ndarray,
        ts_vec: np.ndarray,
        train_config: dict,
        output_dir: str,
        device: torch.device,
    ) -> None:
        s = self.fetch_settings()
        b_net = joint.dynamics.B_net
        if b_net is None:
            raise ValueError("Virtual stimulation requires a model with a b_net.")

        joint = joint.to(device).eval()
        dynamics = joint.dynamics
        b_net = b_net.to(device).eval()

        # 1) Candidate synthesized bipolar pairs (optionally OOD-only).
        pairs = self._build_candidate_pairs(loader)
        if s.ood_only:
            pairs = [p for p in pairs if p["is_ood"]]
        if not pairs:
            print("  Skipping seizure analysis: no candidate bipolar pairs.")
            return
        n_ood = sum(p["is_ood"] for p in pairs)
        print(f"  Seizure analysis: {len(pairs)} bipolar pairs "
              f"({n_ood} OOD / never-stimulated), ood_only={s.ood_only}.")

        # Virtual B columns for every pair via the trained site->column map.
        n_feat = b_net.n_site_features
        feats = torch.tensor(
            np.array([p["feat"][:n_feat] for p in pairs], dtype=np.float32), device=device)
        with torch.no_grad():
            B_cols = b_net.mlp(b_net._normalize(feats)).T.detach()   # (D, P)

        # 2) Resting ICs = encoded latent at pulse onset.
        onset_idx = int(np.argmin(np.abs(ts_vec)))
        x_rest = x_enc_np[:, onset_idx, :]                          # (n_trials, D)
        rng = np.random.default_rng(self.seed)
        n_ics = min(int(s.virtual_stim_n_ics), x_rest.shape[0])
        ic_idx = rng.choice(x_rest.shape[0], size=n_ics, replace=False)
        x0 = torch.tensor(x_rest[ic_idx], dtype=torch.float32, device=device)   # (n_ics, D)

        dt  = float(train_config["settings"]["dt"])
        T   = int(s.virtual_stim_T)
        amp = float(s.virtual_stim_amp)

        # 3) Per-pair excitability readouts and B-column gain.
        rows = []
        for p, pair in enumerate(pairs):
            b_col = B_cols[:, p]
            dY = self._rollout_response(dynamics, joint, x0, b_col, amp, T)
            met = _response_metrics(dY, dt, float(s.plateau_start_ms))
            rows.append({
                "pair": pair["name"], "member_a": pair["a"], "member_b": pair["b"],
                "x": float(pair["xyz"][0]), "y": float(pair["xyz"][1]),
                "z": float(pair["xyz"][2]),
                "zone": _ZONE_NAME[pair["severity"]],
                "zone_severity": pair["severity"],
                "epileptogenic": bool(pair["severity"] > 0),
                "is_ood": bool(pair["is_ood"]),
                "dist_to_nearest_site": pair["dist"],
                "destrieux": pair["destrieux"],
                "gain": float(b_col.norm().item()),
                **met,
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "seizure_virtual_stim.csv"), index=False)

        # 4) Epileptogenic-vs-healthy statistics.
        self._seizure_zone_stats(df, output_dir, ood_only=bool(s.ood_only))
        print(f"  Seizure analysis complete → {output_dir}")

    def _build_candidate_pairs(self, loader) -> List[dict]:
        """Synthesize bipolar pairs from adjacent contacts (same lead, consecutive
        index), represented by their midpoint as in b_net training."""
        ch = loader.channels
        names = ch["name"].tolist()
        xyz = ch[["x", "y", "z"]].to_numpy(dtype=np.float64)
        zone = ch["seizure_zone"].tolist()
        destr = (ch["Destrieux_label_text"].fillna("").tolist()
                 if "Destrieux_label_text" in ch else [""] * len(ch))

        lut = {}
        for i, nm in enumerate(names):
            pc = _parse_contact(nm)
            if pc is not None:
                lut[pc] = i

        # Trained stim sites: member-name sets (for OOD flag) and coords (for dist).
        kept_sites = loader.sites.iloc[loader._site_keep]
        trained = {frozenset({str(r["member1"]), str(r["member2"])})
                   for _, r in kept_sites.iterrows()}
        site_xyz = kept_sites[["x", "y", "z"]].to_numpy(dtype=np.float64)

        pairs = []
        for (lead, idx), i in sorted(lut.items()):
            j = lut.get((lead, idx + 1))
            if j is None:
                continue
            mid = (xyz[i] + xyz[j]) / 2.0
            sev = max(_zone_severity(zone[i]), _zone_severity(zone[j]))
            is_ood = frozenset({names[i], names[j]}) not in trained
            dist = (float(np.min(np.linalg.norm(site_xyz - mid[None], axis=1)))
                    if len(site_xyz) else float("nan"))
            wm = bool(_is_white_matter(pd.Series([destr[i], destr[j]])).any())
            pairs.append({
                "name": f"{names[i]}-{names[j]}", "a": names[i], "b": names[j],
                "xyz": mid, "severity": sev, "is_ood": is_ood, "dist": dist,
                "destrieux": destr[i],
                "feat": [mid[0], mid[1], mid[2], float(wm)],
            })
        return pairs

    @torch.no_grad()
    def _rollout_response(self, dynamics, joint, x0, b_col, amp, T):
        """Paired autonomous rollout: perturbed (impulse at t0) vs baseline.

        Pairing cancels baseline drift (the learned dynamics need not rest at x0).
        Returns dY (n_ics, T, N) as a numpy array.
        """
        x_base = dynamics.step(x0, u=None)
        x_pert = _virtual_impulse_step0(dynamics, x0, b_col, amp)
        xb, xp = [x_base], [x_pert]
        for _ in range(T - 1):
            x_base = dynamics.step(x_base, u=None)
            x_pert = dynamics.step(x_pert, u=None)
            xb.append(x_base)
            xp.append(x_pert)
        Xb = torch.stack(xb, dim=1)   # (n_ics, T, D)
        Xp = torch.stack(xp, dim=1)
        yb = joint.likelihood.mean(joint.emission(Xb))
        yp = joint.likelihood.mean(joint.emission(Xp))
        return (yp - yb).cpu().numpy()

    @staticmethod
    def _seizure_zone_stats(df: pd.DataFrame, output_dir: str, ood_only: bool) -> None:
        """Mann-Whitney test of epileptogenic vs healthy pairs per readout, with BH
        q-values within each subset (all pairs, and OOD pairs when ood_only=False)."""
        from scipy import stats as ss

        subsets = [("all", df)]
        if not ood_only and df["is_ood"].any() and (~df["is_ood"]).any():
            subsets.append(("ood", df[df["is_ood"]]))

        rows = []
        for subset_name, d in subsets:
            a_df = d[d["epileptogenic"]]
            b_df = d[~d["epileptogenic"]]
            sub_rows = []
            for r in _SEIZURE_READOUTS:
                a = a_df[r].dropna().to_numpy()
                b = b_df[r].dropna().to_numpy()
                row = {
                    "readout": r, "subset": subset_name, "test": "mannwhitney",
                    "n_epileptogenic": len(a), "n_healthy": len(b),
                    "median_epileptogenic": float(np.median(a)) if len(a) else float("nan"),
                    "median_healthy": float(np.median(b)) if len(b) else float("nan"),
                }
                if len(a) >= 2 and len(b) >= 2:
                    U, pval = ss.mannwhitneyu(a, b, alternative="two-sided")
                    row["stat"] = float(U)
                    row["p_value"] = float(pval)
                    # Positive = epileptogenic stochastically larger (matches cohen_d).
                    row["rank_biserial"] = float(2 * U / (len(a) * len(b)) - 1)
                    row["cohen_d"] = _cohens_d(a, b)
                else:
                    row["stat"] = row["p_value"] = float("nan")
                    row["rank_biserial"] = row["cohen_d"] = float("nan")
                sub_rows.append(row)

            finite = [i for i, rw in enumerate(sub_rows) if np.isfinite(rw["p_value"])]
            if finite:
                qv = np.atleast_1d(ss.false_discovery_control(
                    [sub_rows[i]["p_value"] for i in finite], method="bh"))
                for i, q in zip(finite, qv):
                    sub_rows[i]["q_value"] = float(q)
            for rw in sub_rows:
                rw.setdefault("q_value", float("nan"))
            rows.extend(sub_rows)

        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "seizure_zone_summary.csv"), index=False)
