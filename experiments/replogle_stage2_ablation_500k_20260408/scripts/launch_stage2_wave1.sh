#!/usr/bin/env bash
set -euo pipefail
ZEROSHOT_JOBID="${ZEROSHOT_JOBID:-146990}"
FEWSHOT_JOBID="${FEWSHOT_JOBID:-146991}"
bash ./launch_on_allocation.sh "${ZEROSHOT_JOBID}" train zeroshot frozen_500k &
bash ./launch_on_allocation.sh "${FEWSHOT_JOBID}" train zeroshot unfrozen_500k &
wait
