# Decision models

Linear and nonlinear iSSMs on three decision-making toy systems (saddle, point attractor, line
attractor), over the training stimulation rate 0, 0.25, 0.5, 0.75.
Reproduces Fig. 3 (flow fields at the highest stimulation rate). The nonlinear sweeps are also used by
the stimulation-protocol figure (see `job_scripts/stimulation_protocol/`).

## Commands

Run from the repository root.

```bash
bash job_scripts/decision_models/run.sh --mode local           # train, eval and figures, sequentially
bash job_scripts/decision_models/run.sh --mode pbs             # PBS array jobs: train -> eval -> figures
bash job_scripts/decision_models/run.sh --stage figures        # figures from existing outputs
bash job_scripts/decision_models/run.sh --mode pbs --dry_run   # print the submissions only
```

`--stage` is one of `train`, `eval`, `figures`, `all` (default). Environment variables: `N_SEEDS`
(default 10), `N_EPOCHS` (default: config value), `JOBS` (job indices, default all), `OUTPUT_ROOT`
(default `outputs`), `FIGURES_ROOT` (default `figures`). Edit `env_setup` in `job_scripts/common.sh`
to activate the environment on your system.

## What runs

| Configs (`configs/toymodels/`) | Jobs | Seeds | Training runs |
|---|---|---|---|
| `saddle{_linear,}.json` | job_0..3 = stim_rate 0..0.75 | 0-9 | 80 |
| `point_attractor{_linear,}.json` | job_0..3 | 0-9 | 80 |
| `line_attractor{_linear,}.json` | job_0..3 | 0-9 | 80 |

240 training runs and 24 evaluations (`EvalISSM`, one per job).

## Outputs

```
outputs/decision_models/{Saddle,PointAttractor,LineAttractor}/{linear,nonlinear}/job_<k>/seed_<NNN>/
outputs/decision_models/{Saddle,PointAttractor,LineAttractor}/{linear,nonlinear}/job_<k>/eval/
figures/decision_models/   decision_models_flows.png
```
