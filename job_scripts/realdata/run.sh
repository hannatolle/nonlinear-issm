#!/usr/bin/env bash
# Macaque micro-stimulation (D = 5, 12, 15, 20) and human intracranial stimulation (D = 15, 20, 25, 30),
# linear vs nonlinear iSSM (job_0..3), plus virtual stimulation of the human main models (job_2).
source "$(dirname "$0")/../common.sh"
parse_args "$@"

DATASETS=${DATASETS:-macaque huang}
STAGES="train eval attractor figures"
RES_TRAIN="select=1:ncpus=4:mem=8gb,walltime=12:00:00"

has() { [[ " $DATASETS " == *" $1 "* ]]; }

if has macaque; then
    add_run TrainKianiISSM EvalKianiISSM configs/macaque/macaque_linear.json "$OUTPUT_ROOT/realdata/macaque/linear" 4
    add_run TrainKianiISSM EvalKianiISSM configs/macaque/macaque.json "$OUTPUT_ROOT/realdata/macaque/nonlinear" 4
fi
if has huang; then
    add_run TrainHuangISSM EvalHuangISSM configs/huang/huang_linear.json "$OUTPUT_ROOT/realdata/huang/linear" 4
    add_run TrainHuangISSM EvalHuangISSM configs/huang/huang.json "$OUTPUT_ROOT/realdata/huang/nonlinear" 4
fi

attractor_cmds() {
    local m s
    has huang && selected 2 || return 0
    for m in linear nonlinear; do
        for ((s = 0; s < N_SEEDS; s++)); do
            printf "%s -x AttractorHuangISSM -s run_dir=%s/realdata/huang/%s/job_2/seed_%03d\n" \
                "$RUN_EX" "$OUTPUT_ROOT" "$m" "$s"
        done
    done
}

figures_cmds() {
    local d=$OUTPUT_ROOT/realdata
    if has macaque; then
        echo "python -m scripts.figures.figure_realdata --linear_train_dir $d/macaque/linear/job_1" \
             "--nonlinear_train_dir $d/macaque/nonlinear/job_1 --output_dir $FIGURES_ROOT/realdata"
    fi
    if has huang; then
        echo "python -m scripts.figures.figure_huang --linear_train_dir $d/huang/linear/job_2" \
             "--nonlinear_train_dir $d/huang/nonlinear/job_2 --output_dir $FIGURES_ROOT/huang"
    fi
}

main
