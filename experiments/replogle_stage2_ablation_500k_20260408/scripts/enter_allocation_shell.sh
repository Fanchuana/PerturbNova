#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <jobid>"
  exit 2
fi

JOBID="$1"
NODELIST="$(squeue -h -j "${JOBID}" -o '%N')"
if [[ -z "${NODELIST}" || "${NODELIST}" == "(null)" ]]; then
  echo "[ERROR] cannot resolve nodelist for job ${JOBID}"
  exit 1
fi

echo "[allocation-shell] entering job=${JOBID} node=${NODELIST}"
srun --jobid="${JOBID}" --nodelist="${NODELIST}" --nodes=1 --ntasks=1 --overlap --pty bash -l
