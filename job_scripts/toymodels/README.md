# Toy models

Linear and nonlinear iSSMs on two oscillatory toy systems, FitzHugh-Nagumo and Van der Pol, over
the model latent dimension D = 1, 2, 3, 4 (the true system is 2-D).
Reproduces Fig. 2 (FitzHugh-Nagumo) and the corresponding supplementary figure (Van der Pol).

## Commands

Run from the repository root.

```bash
bash job_scripts/toymodels/run.sh --mode local              # train, eval and figures, sequentially
bash job_scripts/toymodels/run.sh --mode pbs                # PBS array jobs: train -> eval -> figures
bash job_scripts/toymodels/run.sh --stage figures           # figures from existing outputs
bash job_scripts/toymodels/run.sh --mode pbs --dry_run      # print the submissions only
```

`--stage` is one of `train`, `eval`, `figures`, `all` (default). Environment variables: `N_SEEDS`
(default 10), `N_EPOCHS` (default: config value), `JOBS` (job indices, default all), `OUTPUT_ROOT`
(default `outputs`), `FIGURES_ROOT` (default `figures`). Edit `env_setup` in `job_scripts/common.sh`
to activate the environment on your system.

## What runs

| Configs (`configs/toymodels/`) | Jobs | Seeds | Training runs |
|---|---|---|---|
| `fitzhugh_nagumo{_linear,}.json` | job_0..3 = D 1..4 | 0-9 | 80 |
| `vanderpol{_linear,}.json` | job_0..3 = D 1..4 | 0-9 | 80 |

160 training runs and 16 evaluations (`EvalISSM`, one per job). The figure panels (a)-(d) use job_1 (D = 2);
panel (e) uses all D.

## Outputs

```
outputs/toymodels/{FitzHughNagumo,VanDerPol}/{linear,nonlinear}/job_<k>/seed_<NNN>/   trained models
outputs/toymodels/{FitzHughNagumo,VanDerPol}/{linear,nonlinear}/job_<k>/eval/         evaluation
figures/toymodels/   fig_toymodels_main_{fhn,vdp}.png, toymodels_{fhn,vdp}_{stats,per_seed}.csv
```
