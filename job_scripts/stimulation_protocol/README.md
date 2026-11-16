# Stimulation protocol

Nonlinear iSSMs on the three decision-making toy systems, trained on four stimulation protocols:
regular or Poisson stimulation timing, crossed with constant or diverse stimulation amplitudes.
Together with the nonlinear stimulation-rate sweep of `job_scripts/decision_models/`, reproduces
Fig. 6 (saddle) and the corresponding supplementary figure (point and line attractor).

## Commands

Run from the repository root. The figure stage needs the decision-model outputs as well, so run
`job_scripts/decision_models/run.sh` first (its train and eval stages suffice).

```bash
bash job_scripts/stimulation_protocol/run.sh --mode local           # train, eval and figures, sequentially
bash job_scripts/stimulation_protocol/run.sh --mode pbs             # PBS array jobs: train -> eval -> figures
bash job_scripts/stimulation_protocol/run.sh --stage figures        # figures from existing outputs
bash job_scripts/stimulation_protocol/run.sh --mode pbs --dry_run   # print the submissions only
```

`--stage` is one of `train`, `eval`, `figures`, `all` (default). Environment variables: `N_SEEDS`
(default 10), `N_EPOCHS` (default: config value), `JOBS` (job indices, default all), `OUTPUT_ROOT`
(default `outputs`), `FIGURES_ROOT` (default `figures`). Edit `env_setup` in `job_scripts/common.sh`
to activate the environment on your system.

## What runs

| Configs (`configs/toymodels/stimulation_protocol/`) | Jobs | Seeds | Training runs |
|---|---|---|---|
| `{saddle,point_attractor,line_attractor}_{regular,poisson}_{constant,diverse}.json` (12) | job_0 | 0-9 | 120 |

120 training runs and 12 evaluations (`EvalISSM`, one per protocol).

## Outputs

```
outputs/stimulation_protocol/{Saddle,PointAttractor,LineAttractor}/<timing>_<amplitude>/job_0/seed_<NNN>/
outputs/stimulation_protocol/{Saddle,PointAttractor,LineAttractor}/<timing>_<amplitude>/job_0/eval/
figures/interventions/   fig_interventions_saddle.png, supp_fig_interventions_point_line.png, interventions_stats.csv
```
