# Real data

Linear and nonlinear iSSMs on two stimulation datasets: macaque prefrontal spiking under
micro-stimulation, and human intracranial recordings under single-pulse electrical stimulation,
each over four latent dimensions. The human main models are then virtually stimulated at every
electrode pair to compare seizure-onset and other sites.
Reproduces Fig. 4 (macaque), Fig. 5 (human) and the supplementary D-sweep and seizure-zone figures.

## Data

- Macaque: not publicly available. The configs expect the session files in `data/macaque/`.
- Human: OpenNeuro ds004457. See the top-level README for download and preprocessing; the configs
  expect the preprocessed data in `data/huang_2023/preproc/`.

## Commands

Run from the repository root.

```bash
bash job_scripts/realdata/run.sh --mode local                  # train, eval, attractor and figures, sequentially
bash job_scripts/realdata/run.sh --mode pbs                    # PBS array jobs: train -> eval, attractor -> figures
DATASETS=huang bash job_scripts/realdata/run.sh --mode pbs     # human dataset only
bash job_scripts/realdata/run.sh --stage figures               # figures from existing outputs
bash job_scripts/realdata/run.sh --mode pbs --dry_run          # print the submissions only
```

`--stage` is one of `train`, `eval`, `attractor`, `figures`, `all` (default). Environment variables:
`DATASETS` (default `macaque huang`), `N_SEEDS` (default 10), `N_EPOCHS` (default: config value),
`JOBS` (job indices, default all), `OUTPUT_ROOT` (default `outputs`), `FIGURES_ROOT` (default
`figures`). Edit `env_setup` in `job_scripts/common.sh` to activate the environment on your system.

## What runs

| Configs | Experiments | Jobs | Seeds | Runs |
|---|---|---|---|---|
| `configs/macaque/macaque{_linear,}.json` | `TrainKianiISSM`, `EvalKianiISSM` | job_0..3 = D 5, 12, 15, 20 | 0-9 | 80 |
| `configs/huang/huang{_linear,}.json` | `TrainHuangISSM`, `EvalHuangISSM` | job_0..3 = D 15, 20, 25, 30 | 0-9 | 80 |
| human main models | `AttractorHuangISSM` | job_2 (D = 25) | 0-9 | 20 |

160 training runs, 16 evaluations (one per job) and 20 virtual-stimulation runs (one per human main-model
seed). The main figures use job_1 (macaque, D = 12) and job_2 (human, D = 25); the supplements use all D.

## Outputs

```
outputs/realdata/{macaque,huang}/{linear,nonlinear}/job_<k>/seed_<NNN>/   trained models
outputs/realdata/{macaque,huang}/{linear,nonlinear}/job_<k>/eval/         evaluation
outputs/realdata/huang/{linear,nonlinear}/job_2/seed_<NNN>/attractor/     virtual stimulation
figures/realdata/   main_macaque.png, overview_macaque.png, supplement_macaque_d_sweeps.png, macaque_*.csv
figures/huang/      main_huang.png, overview_huang.png, supplement_huang_{d_sweeps,seizure_zones}.png, huang_*.csv
```
