#!/usr/bin/env bash
# PBS task: runs line PBS_ARRAY_INDEX (0-based) of the task file TASKS.
set -euo pipefail
cd "${PBS_O_WORKDIR:-.}"
source job_scripts/common.sh
env_setup
cmd=$(sed -n "$((${PBS_ARRAY_INDEX:-0} + 1))p" "$TASKS")
echo "$cmd"
eval "$cmd"
