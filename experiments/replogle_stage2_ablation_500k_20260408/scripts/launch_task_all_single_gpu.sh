#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <mode:train|infer> <task:fewshot|zeroshot> <frozen_jobid> <unfrozen_jobid> <v1_jobid>"
  exit 2
fi
MODE="$1"
TASK="$2"
FROZEN_JOBID="$3"
UNFROZEN_JOBID="$4"
V1_JOBID="$5"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
(
  cd "${SCRIPT_DIR}"
  bash ./launch_on_allocation.sh "${FROZEN_JOBID}" "${MODE}" "${TASK}" frozen_500k
) &
PID1=$!
(
  cd "${SCRIPT_DIR}"
  bash ./launch_on_allocation.sh "${UNFROZEN_JOBID}" "${MODE}" "${TASK}" unfrozen_500k
) &
PID2=$!
(
  cd "${SCRIPT_DIR}"
  bash ./launch_on_allocation.sh "${V1_JOBID}" "${MODE}" "${TASK}" v1_500k
) &
PID3=$!
STATUS=0
wait ${PID1} || STATUS=$?
wait ${PID2} || STATUS=$?
wait ${PID3} || STATUS=$?
exit ${STATUS}
