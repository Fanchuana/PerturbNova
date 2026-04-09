#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <variant:frozen_500k|unfrozen_500k|v1_500k>"
  exit 2
fi
VARIANT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/work/home/cryoem666/xyf/temp/pycharm/PerturbNova"
CFG_ROOT="${PROJECT_ROOT}/experiments/replogle_stage2_ablation_500k_20260408/generated_configs_4gpu/train/fewshot"
NODES=(g01n09 g01n10 g01n12 g01n13)
JOBS=(147433 147434 146990 146991)
SPLITS=(hepg2 jurkat k562 rpe1)
PIDS=()
for i in 0 1 2 3; do
  NODE="${NODES[$i]}"
  JOBID="${JOBS[$i]}"
  SPLIT="${SPLITS[$i]}"
  CFG="${CFG_ROOT}/${SPLIT}/${VARIANT}.toml"
  echo "[fewshot-4gpu] split=${SPLIT} variant=${VARIANT} job=${JOBID} node=${NODE} cfg=${CFG}"
  (
    srun --jobid="${JOBID}" --nodelist="${NODE}" --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:4       bash -lc "cd '${PROJECT_ROOT}' && export NPROC_PER_NODE=4 && bash scripts/train.sh '${CFG}'"
  ) &
  PIDS+=("$!")
done
status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
