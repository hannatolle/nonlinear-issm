#!/usr/bin/env bash
# FitzHugh-Nagumo and Van der Pol, linear vs nonlinear iSSM, latent dimension D = 1..4 (job_0..3).
source "$(dirname "$0")/../common.sh"
parse_args "$@"

for sys in FitzHughNagumo:fitzhugh_nagumo VanDerPol:vanderpol; do
    name=${sys%%:*} cfg=configs/toymodels/${sys#*:}
    add_run TrainISSM EvalISSM "${cfg}_linear.json" "$OUTPUT_ROOT/toymodels/$name/linear" 4
    add_run TrainISSM EvalISSM "$cfg.json" "$OUTPUT_ROOT/toymodels/$name/nonlinear" 4
done

figures_cmds() {
    local d
    for d in fhn:FitzHughNagumo vdp:VanDerPol; do
        echo "python -m scripts.figures.figure_toymodels --dataset ${d%%:*}" \
             "--d_sweep_dir_1 $OUTPUT_ROOT/toymodels/${d#*:}/linear" \
             "--d_sweep_dir_2 $OUTPUT_ROOT/toymodels/${d#*:}/nonlinear" \
             "--output_dir $FIGURES_ROOT/toymodels"
    done
}

main
