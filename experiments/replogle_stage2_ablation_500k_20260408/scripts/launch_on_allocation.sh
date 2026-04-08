#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <jobid> <mode:train|infer> <task> <group>"
  exit 2
fi

JOBID="$1"
MODE="$2"
TASK="$3"
GROUP="$4"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
GPUS_PER_TASK_GROUP="${GPUS_PER_TASK_GROUP:-4}"
NODELIST="$(squeue -h -j "${JOBID}" -o '%N')"
if [[ -z "${NODELIST}" || "${NODELIST}" == "(null)" ]]; then
  echo "[ERROR] cannot resolve nodelist for job ${JOBID}"
  exit 1
fi

echo "[allocation] job=${JOBID} node=${NODELIST} mode=${MODE} task=${TASK} group=${GROUP} cpus=${CPUS_PER_TASK} gpus=${GPUS_PER_TASK_GROUP}"

srun   --jobid="${JOBID}"   --nodelist="${NODELIST}"   --nodes=1   --ntasks=1   --cpus-per-task="${CPUS_PER_TASK}"   --gres="gpu:${GPUS_PER_TASK_GROUP}" bash -lc "cd '/work/home/cryoem666/xyf/temp/pycharm/PerturbNova' && bash '/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/scripts/run_group.sh' '${MODE}' '${TASK}' '${GROUP}'"
