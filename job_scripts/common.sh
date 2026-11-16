# Shared driver for job_scripts/<suite>/run.sh. Run the suites from the repo root:
#
#   bash job_scripts/<suite>/run.sh [--mode local|pbs] [--stage <stage>|all] [--dry_run]
#
# Environment: N_SEEDS (10), N_EPOCHS (config value), JOBS (job indices, default all),
# OUTPUT_ROOT (outputs), FIGURES_ROOT (figures).

# ==== Environment setup: edit to activate the camel environment on your system ====
# Runs before any python command, locally and at the start of every PBS job.
env_setup() {
    set +u
    if command -v conda >/dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate camel
    fi
    set -u
}
# ====================================================================================

set -euo pipefail

N_SEEDS=${N_SEEDS:-10}
N_EPOCHS=${N_EPOCHS:-}
JOBS=${JOBS:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
FIGURES_ROOT=${FIGURES_ROOT:-figures}
SUITE=$(basename "$(dirname "$0")")

# PBS resources per stage; suites may override.
RES_TRAIN="select=1:ncpus=4:mem=4gb,walltime=08:00:00"
RES_EVAL="select=1:ncpus=4:mem=8gb,walltime=04:00:00"
RES_ATTRACTOR="select=1:ncpus=4:mem=8gb,walltime=02:00:00"
RES_FIGURES="select=1:ncpus=4:mem=8gb,walltime=02:00:00"

STAGES="train eval figures"
RUN_EX="python -m src.experiments.run_ex"
RUNS=()

parse_args() {
    MODE=local STAGE=all DRY_RUN=false
    while (($#)); do
        case $1 in
            --mode) MODE=$2; shift 2 ;;
            --stage) STAGE=$2; shift 2 ;;
            --dry_run) DRY_RUN=true; shift ;;
            *) echo "unknown argument: $1" >&2; exit 1 ;;
        esac
    done
    [[ $MODE == local || $MODE == pbs ]] || { echo "--mode must be local or pbs" >&2; exit 1; }
}

# add_run <train experiment> <eval experiment> <config> <output dir> <number of jobs> [extra eval args]
add_run() { RUNS+=("$1|$2|$3|$4|$5|${6:-}"); }

selected() { [[ -z $JOBS || " ${JOBS//,/ } " == *" $1 "* ]]; }

job_indices() {
    local j
    for ((j = 0; j < $1; j++)); do
        if selected "$j"; then echo "$j"; fi
    done
}

train_cmds() {
    local r tx ex cfg out n extra j s
    for r in "${RUNS[@]}"; do
        IFS='|' read -r tx ex cfg out n extra <<< "$r"
        for j in $(job_indices "$n"); do
            for ((s = 0; s < N_SEEDS; s++)); do
                echo "$RUN_EX -x $tx -c $cfg -o $out -j $j --group_seeds --seed_idx $s${N_EPOCHS:+ -s n_epochs=$N_EPOCHS}"
            done
        done
    done
}

eval_cmds() {
    local r tx ex cfg out n extra j
    for r in "${RUNS[@]}"; do
        IFS='|' read -r tx ex cfg out n extra <<< "$r"
        for j in $(job_indices "$n"); do
            echo "$RUN_EX -x $ex -s group_dir=$out/job_$j -s use_default_output_dir=false -o $out/job_$j/eval${extra:+ $extra}"
        done
    done
}

abspath() { if [[ $1 == /* ]]; then echo "$1"; else echo "$PWD/$1"; fi; }

# submit <stage> <dependency ids...>: one PBS array job, one task per command in CMDS.
submit() {
    local stage=$1 n=${#CMDS[@]} dir res deps
    shift
    dir=$(abspath "$OUTPUT_ROOT/pbs/$SUITE")
    res="RES_${stage^^}"
    local args=(-N "${SUITE}_$stage" -S /bin/bash -l "${!res}" -j oe -o "$dir/logs/" -v "TASKS=$dir/$stage.tasks")
    ((n > 1)) && args+=(-J "0-$((n - 1))")
    deps=$(IFS=:; echo "$*")
    [[ -n $deps ]] && args+=(-W "depend=afterok:$deps")
    if $DRY_RUN; then
        echo "qsub ${args[*]} job_scripts/worker.sh  # tasks: $n"
        JOB_ID="<${SUITE}_$stage>"
    else
        mkdir -p "$dir/logs"
        printf '%s\n' "${CMDS[@]}" > "$dir/$stage.tasks"
        JOB_ID=$(qsub "${args[@]}" job_scripts/worker.sh)
        echo "$stage: $JOB_ID ($n tasks)"
    fi
}

main() {
    local stages=$STAGE st c
    [[ $STAGE == all ]] && stages=$STAGES
    [[ $STAGE == all || " $STAGES " == *" $STAGE "* ]] || { echo "--stage must be one of: $STAGES all" >&2; exit 1; }
    [[ $MODE == local ]] && ! $DRY_RUN && env_setup
    declare -A ids=()
    for st in $stages; do
        mapfile -t CMDS < <("${st}_cmds")
        ((${#CMDS[@]})) || continue
        if [[ $MODE == local ]]; then
            for c in "${CMDS[@]}"; do
                echo "$c"
                $DRY_RUN || eval "$c"
            done
        else
            case $st in
                eval | attractor) submit "$st" ${ids[train]:-} ;;
                figures) submit "$st" ${ids[eval]:-} ${ids[attractor]:-} ;;
                *) submit "$st" ;;
            esac
            ids[$st]=$JOB_ID
        fi
    done
}
