"""
Train an iSSM on the Huang et al. (2023) SEEG single-pulse-stimulation dataset.
"""

import numpy as np
import torch

from src.datasets import HuangDataloader
from .train_real_issm import TrainRealDataISSM


class TrainHuangISSM(TrainRealDataISSM):
    """Train an iSSM on one subject of the Huang 2023 SEEG/CCEP dataset (ds004457).

    Every trial is a single biphasic pulse at one bipolar site: strata are the
    stim-site indices and u is a single-sample impulse at pulse onset (graded by
    current, normalized to <=1 by default).

    Loader options:
      - drop_bad / require_location / drop_white_matter select the observed
        channels (N); drop_white_matter also drops white-matter stim sites (M);
      - y is decimated to resample_hz and cropped to [t_start_ms, t_end_ms];
      - drop_artefact_trials removes fully stim-artefact-corrupted trials;
      - drop_residual_artefact_trials removes trials whose artefact flags
        (scripts.huang.stim_artefact) hit any channel other than the trial's
        own stim contacts.

    Observation mask (shared by train, eval and attractor): each trial's own
    stim-contact channels are unobserved for the whole trial. The likelihood
    marginalizes them out, standardization ignores them, and they are
    zero-filled before the encoder.

    N, M and dt (= 1 / resample_hz) are inferred in preprocess_config.
    """

    zero_fill_masked_obs = True

    @staticmethod
    def _parse_ood_channels(ood) -> list[str]:
        """Parse the comma-separated ood_test_channels setting into site names.

        Accepts a string ("RA13-RA14,LTG3-LTG4"), None/"" (no OOD), or a list.
        The setting is a string, not a list, so it never becomes a -j sweep axis.
        """
        if ood is None:
            return []
        if isinstance(ood, (list, tuple)):
            ood = ",".join(str(x) for x in ood)
        return [x.strip() for x in str(ood).split(",") if x.strip()]

    @classmethod
    def _build_loader(cls, s) -> HuangDataloader:
        """Construct a HuangDataloader from a settings namespace/dict."""
        get = (lambda k, d=None: s.get(k, d)) if isinstance(s, dict) else (
            lambda k, d=None: getattr(s, k, d))
        return HuangDataloader(
            preproc_dir=get("preproc_dir", "data/huang_2023/preproc"),
            subject=get("subject", "sub-2"),
            drop_bad=get("drop_bad", True),
            require_location=get("require_location", True),
            drop_white_matter=get("drop_white_matter", False),
            resample_hz=get("resample_hz", 256),
            t_start_ms=get("t_start_ms", -50.0),
            t_end_ms=get("t_end_ms", 300.0),
            normalize_u=get("normalize_u", True),
            drop_artefact_trials=get("drop_artefact_trials", True),
            drop_residual_artefact_trials=get("drop_residual_artefact_trials", False),
        )

    @classmethod
    def preprocess_config(cls, config: dict) -> dict:
        """Infer N, M and dt from the dataset + resample rate before resolution."""
        settings = config.get("settings", {})
        loader = cls._build_loader(settings)
        resample_hz = settings.get("resample_hz", 256)
        settings = {
            **settings,
            "N": loader.N,
            "M": loader.M,
            "dt": 1.0 / float(resample_hz),
        }
        return {**config, "settings": settings}

    def check_config(self):
        required_components = ["dynamics", "emission", "likelihood", "recognition"]
        for comp in required_components:
            if comp not in self.components:
                raise ValueError(f"TrainHuangISSM requires component '{comp}'.")
        self.match_settings(strict=True)
        s = self.fetch_settings()

        fs_raw = 2048
        if fs_raw % s.resample_hz != 0:
            raise ValueError(
                f"resample_hz={s.resample_hz} must evenly divide the recording "
                f"rate {fs_raw} Hz."
            )
        expected_dt = 1.0 / float(s.resample_hz)
        if abs(expected_dt - s.dt) > 1e-9:
            raise ValueError(
                f"dt must equal 1/resample_hz = {expected_dt} s, but config has "
                f"dt={s.dt}. (dt is auto-set in preprocess_config.)"
            )
        if not (s.t_start_ms < 0 < s.t_end_ms):
            raise ValueError(
                f"t_start_ms < 0 < t_end_ms required (the pulse at t=0 must be "
                f"in-window), got t_start_ms={s.t_start_ms}, t_end_ms={s.t_end_ms}."
            )
        if s.n_val_samples == 0 and s.early_stopping:
            raise ValueError(
                "early_stopping=True requires n_val_samples > 0 "
                "(validation set is needed to monitor training).")

        # OOD stim-channel holdout requires a b_net: a held-out site gets no
        # training trials, so its raw-B column would stay ungradiented and its
        # rollout meaningless. b_net produces that column from site features.
        if self._parse_ood_channels(getattr(s, "ood_test_channels", None)):
            if "b_net" not in self.components:
                raise ValueError(
                    "ood_test_channels is set but no 'b_net' component is "
                    "configured; OOD generalization to an unseen stim site "
                    "requires b_net (B is produced from site features, not "
                    "learned per-site).")

        # With a b_net, B is computed by the net, so the raw-B machinery must be
        # inert: B_init='none' and max_tau_B=0 (thresholding a computed B is a no-op).
        if "b_net" in self.components:
            dp = self.components["dynamics"].params
            if dp.get("B_init") != "none":
                raise ValueError("With a 'b_net' component, dynamics B_init must be 'none'.")
            if dp.get("max_tau_B", 0) != 0:
                raise ValueError("With a 'b_net' component, dynamics max_tau_B must be 0.")
            if dp.get("B_prior", "none") not in ("none", "l1"):
                raise ValueError(
                    "With a 'b_net' component, dynamics B_prior must be 'none' or 'l1'.")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Load SEEG voltages and stimulation impulses for one subject.

        Returns:
            y:      (n_trials, T, N) float32 tensor (microvolts).
            u:      (n_trials, T, M) float32 tensor (onset impulses).
            strata: (n_trials,) int32 stim-site indices for stratified splitting.
        """
        s = self.fetch_settings()
        loader = self._build_loader(s)
        self._loader = loader  # used by _get_site_features

        act = loader.get_trials()
        y = torch.tensor(act.d, dtype=torch.float32)   # (trials, T, N)
        u = torch.tensor(act.u, dtype=torch.float32)   # (trials, T, M)
        strata = loader.get_strata().astype(np.int32)

        if y.shape[2] != s.N:
            raise ValueError(f"Expected N={s.N} channels, but data has {y.shape[2]}.")
        if u.shape[2] != s.M:
            raise ValueError(f"Expected M={s.M} stim sites, but data has {u.shape[2]}.")

        # OOD holdout: flag all trials of the named stim channels for test (the
        # sites stay in u / site_features; only their trials are re-routed).
        ood_names = self._parse_ood_channels(getattr(s, "ood_test_channels", None))
        if ood_names:
            full = loader.resolve_ood_sites(ood_names)  # raises on unknown/filtered
            self._force_test_idx = np.where(np.isin(strata, full))[0]
            self._ood_channel_names = ood_names
        else:
            self._force_test_idx = None
            self._ood_channel_names = []

        self._artefact_mask = self._build_artefact_mask(loader, act.t)
        self._pool_trial_ids = loader.trial_ids
        self._loader_T = y.shape[1]
        return y, u, strata

    def _build_artefact_mask(
        self, loader: HuangDataloader, t: np.ndarray
    ) -> torch.Tensor:
        """HuangDataloader.build_loss_mask plus verbose reporting.

        The mask logic lives in the loader so train, eval and the attractor
        analysis share it.
        """
        mask, stats = loader.build_loss_mask(t)

        if self.verbose:
            n_masked = int(mask.sum().item())
            min_obs = int((~mask).sum(dim=-1).min().item())
            print(f"  Observation mask: {stats['n_stim_ch']} stim-contact (trial, ch) "
                  f"whole-trial = {n_masked} masked (trial, t, ch) cells "
                  f"({100 * n_masked / mask.numel():.2f}%); min observed channels = {min_obs}.")
            print(f"  Residual-artefact trials dropped: {loader.n_residual_artefact_dropped}.")
        return mask

    def _get_site_features(self) -> torch.Tensor:
        """Per-stim-site features (M, n_features) for a SiteEmbeddingB, aligned to u:
        coordinates, plus a white-matter flag when white-matter sites are kept.
        """
        return self._loader.site_features

    def _on_data_loaded_verbose(self) -> None:
        s = self.fetch_settings()
        T = getattr(self, "_loader_T", "?")
        print(f"  Effective N={s.N}, M={s.M}, D={s.D}; T={T} @ {s.resample_hz} Hz "
              f"([{s.t_start_ms}, {s.t_end_ms}] ms), normalize_u={s.normalize_u}")
        if getattr(self, "_ood_channel_names", None):
            n_ood = len(self._force_test_idx) if self._force_test_idx is not None else 0
            print(f"  OOD holdout: {n_ood} trials of {self._ood_channel_names} "
                  f"forced to test (excluded from train/val).")
