#!/usr/bin/env bash
set -euo pipefail
FROZEN_JOBID="${FROZEN_JOBID:-146990}"
UNFROZEN_JOBID="${UNFROZEN_JOBID:-146991}"
V1_JOBID="${V1_JOBID:-147435}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
bash ./launch_task_all_single_gpu.sh infer zeroshot "${FROZEN_JOBID}" "${UNFROZEN_JOBID}" "${V1_JOBID}"
