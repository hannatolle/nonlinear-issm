#!/usr/bin/env bash
# Saddle, point attractor and line attractor, linear vs nonlinear iSSM, stimulation-rate sweep (job_0..3).
source "$(dirname "$0")/../common.sh"
parse_args "$@"

for sys in Saddle:saddle PointAttractor:point_attractor LineAttractor:line_attractor; do
    name=${sys%%:*} cfg=configs/toymodels/${sys#*:}
    add_run TrainISSM EvalISSM "${cfg}_linear.json" "$OUTPUT_ROOT/decision_models/$name/linear" 4
    add_run TrainISSM EvalISSM "$cfg.json" "$OUTPUT_ROOT/decision_models/$name/nonlinear" 4
done

figures_cmds() {
    echo "python -m scripts.figures.figure_decision_models --base_dir $OUTPUT_ROOT/decision_models" \
         "--output_dir $FIGURES_ROOT/decision_models"
}

main
