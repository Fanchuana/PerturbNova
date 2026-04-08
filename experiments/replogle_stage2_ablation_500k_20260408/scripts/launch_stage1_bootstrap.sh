#!/usr/bin/env bash
set -euo pipefail
ZEROSHOT_JOBID="${ZEROSHOT_JOBID:-146990}"
FEWSHOT_JOBID="${FEWSHOT_JOBID:-146991}"
bash ./launch_on_allocation.sh "${ZEROSHOT_JOBID}" train zeroshot stage1 &
bash ./launch_on_allocation.sh "${FEWSHOT_JOBID}" train fewshot stage1 &
wait
