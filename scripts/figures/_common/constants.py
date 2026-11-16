"""Display names for systems, models and metrics used in the figures.

Visual style (colours, font sizes, widths) lives in src/utils/style.py.
"""

# Systems ----------------------------------------------------------------------

SYSTEM_LABELS = {
    "PointAttractor": "Point attractor",
    "LineAttractor":  "Line attractor",
    "Saddle":         "Bistable well",
    "FitzHughNagumo": "FitzHugh-Nagumo",
    "VanDerPol":      "Van der Pol",
}

# The three non-identifiable decision toy systems, in display order.
DECISION_MODELS = ["PointAttractor", "LineAttractor", "Saddle"]
DECISION_MODEL_LABELS = {k: SYSTEM_LABELS[k] for k in DECISION_MODELS}

# Models -----------------------------------------------------------------------

MODEL_LABELS = {"linear": "Linear", "nonlinear": "Nonlinear"}

# Metrics ----------------------------------------------------------------------

METRIC_LABELS = {
    "y_corr":               "Reconstruction accuracy",
    "y_corr_rollout":       "Rollout accuracy",
    "x_pdc_aligned_corr":   "Latent accuracy",
    "jac_cosine":           "Latent faithfulness",
}

# Intervention-hardness models. Keys are the config/output-dir names.
HARDNESS_MODEL_LABELS = {
    "m1": "iSSM",
    "m2": "SSM",
    "m3": "Input-conditioned SSM",
}

XLABEL_ALPHA = r"Intervention hardness ($\alpha$)"

# Compact spellings for dense multi-panel figures.
METRIC_LABELS_COMPACT = {
    "y_corr":                 "Recon. accuracy",
    "y_corr_rollout":         "Rollout accuracy",
}

# Huang test tiers: in-distribution held-out trials vs trials of held-out stim sites.
HUANG_TIER_LABELS = {"Stim": "Held-in", "Stim_OOD": "Held-out"}

# Virtual-stimulation readouts (Huang seizure-zone analysis).
SEIZURE_READOUT_LABELS = {
    "resp_peak":  "Peak response",
    "plateau_uv": "Sustained response",
    "resp_auc":   "Response AUC",
    "gain":       "Intervention gain",
}

# Generated time series --------------------------------------------------------

SERIES_LABELS = {
    "recon":   "Reconstructions",
    "rollout": "Rollouts",
    "latent":  "Latents",
}

# Axis labels ------------------------------------------------------------------

XLABEL_D_SHORT = "Latent dim. $D$"
XLABEL_N_INTERVENTIONS = "# Interventions"
