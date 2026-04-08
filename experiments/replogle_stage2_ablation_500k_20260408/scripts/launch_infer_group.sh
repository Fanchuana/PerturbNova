#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <jobid> <task> <group>"
  exit 2
fi
JOBID="$1"
TASK="$2"
GROUP="$3"
bash ./launch_on_allocation.sh "${JOBID}" infer "${TASK}" "${GROUP}"
