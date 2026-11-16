#!/usr/bin/env bash
# Saddle, point attractor and line attractor, nonlinear iSSM, stimulation timing (regular/poisson)
# x amplitude (constant/diverse). The figure also reads the decision_models stimulation-rate sweep.
source "$(dirname "$0")/../common.sh"
parse_args "$@"

for sys in Saddle:saddle PointAttractor:point_attractor LineAttractor:line_attractor; do
    for protocol in regular_constant regular_diverse poisson_constant poisson_diverse; do
        add_run TrainISSM EvalISSM "configs/toymodels/stimulation_protocol/${sys#*:}_$protocol.json" \
            "$OUTPUT_ROOT/stimulation_protocol/${sys%%:*}/$protocol" 1
    done
done

figures_cmds() {
    echo "python -m scripts.figures.figure_interventions --decision_models_dir $OUTPUT_ROOT/decision_models" \
         "--protocol_dir $OUTPUT_ROOT/stimulation_protocol --output_dir $FIGURES_ROOT/interventions"
}

main
