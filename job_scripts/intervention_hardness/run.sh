#!/usr/bin/env bash
# Saddle and FitzHugh-Nagumo, three models (m1 iSSM, m2 SSM, m3 input-conditioned SSM),
# intervention strength alpha = 1, 0.67, 0.33, 0 (job_0..3).
source "$(dirname "$0")/../common.sh"
parse_args "$@"

for sys in Saddle:saddle FitzHughNagumo:fitzhugh_nagumo; do
    for m in m1 m2 m3; do
        add_run TrainISSM EvalISSM "configs/toymodels/intervention_hardness/${sys#*:}_$m.json" \
            "$OUTPUT_ROOT/intervention_hardness/${sys%%:*}/$m" 4 "-s latent_alignment=corr"
    done
done

figures_cmds() {
    echo "python -m scripts.figures.figure_hardness --base_dir $OUTPUT_ROOT/intervention_hardness" \
         "--output_dir $FIGURES_ROOT/hardness"
}

main
