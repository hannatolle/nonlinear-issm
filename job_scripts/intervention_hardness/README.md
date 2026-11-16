# Intervention hardness

Three models on the saddle and FitzHugh-Nagumo toy systems, with data generated under interventions
of decreasing hardness, from a hard clamp (alpha = 1) to a purely additive input (alpha = 0):
m1 iSSM, m2 SSM, m3 input-conditioned SSM. Reproduces the supplementary intervention-hardness figure.

## Commands

Run from the repository root.

```bash
bash job_scripts/intervention_hardness/run.sh --mode local           # train, eval and figures, sequentially
bash job_scripts/intervention_hardness/run.sh --mode pbs             # PBS array jobs: train -> eval -> figures
bash job_scripts/intervention_hardness/run.sh --stage figures        # figures from existing outputs
bash job_scripts/intervention_hardness/run.sh --mode pbs --dry_run   # print the submissions only
```

`--stage` is one of `train`, `eval`, `figures`, `all` (default). Environment variables: `N_SEEDS`
(default 10), `N_EPOCHS` (default: config value), `JOBS` (job indices, default all), `OUTPUT_ROOT`
(default `outputs`), `FIGURES_ROOT` (default `figures`). Edit `env_setup` in `job_scripts/common.sh`
to activate the environment on your system.

## What runs

| Configs (`configs/toymodels/intervention_hardness/`) | Jobs | Seeds | Training runs |
|---|---|---|---|
| `saddle_{m1,m2,m3}.json` | job_0..3 = alpha 1, 0.67, 0.33, 0 | 0-9 | 120 |
| `fitzhugh_nagumo_{m1,m2,m3}.json` | job_0..3 | 0-9 | 120 |

240 training runs and 24 evaluations (`EvalISSM -s latent_alignment=corr`, one per job: the models
differ in how inputs enter, so latents are aligned by correlation rather than through B).

## Outputs

```
outputs/intervention_hardness/{Saddle,FitzHughNagumo}/{m1,m2,m3}/job_<k>/seed_<NNN>/
outputs/intervention_hardness/{Saddle,FitzHughNagumo}/{m1,m2,m3}/job_<k>/eval/
figures/hardness/   hardness.png, hardness_stats.csv, hardness_per_seed.csv
```
