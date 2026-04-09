#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:?TASK is required}"
VARIANT="${VARIANT:?VARIANT is required}"
EXP_ROOT="/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408"
PROJECT_ROOT="/work/home/cryoem666/xyf/temp/pycharm/PerturbNova"
TRAIN_CFG_ROOT="${EXP_ROOT}/generated_configs/train/${TASK}"
INFER_CFG_ROOT="${EXP_ROOT}/generated_configs/infer/${TASK}"
SPLITS=(hepg2 jurkat k562 rpe1)

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  set +u
  source "${CONDA_SH}"
  set -u
else
  eval "$(conda shell.bash hook)"
fi
set +u
conda activate my_state
set -u

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

run_train_split() {
  local gpu_id="$1"
  local split="$2"
  local cfg="${TRAIN_CFG_ROOT}/${split}/${VARIANT}.toml"
  echo "[train] task=${TASK} variant=${VARIANT} split=${split} gpu=${gpu_id} cfg=${cfg}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export NPROC_PER_NODE=1
    bash "${PROJECT_ROOT}/scripts/train.sh" "${cfg}"
  ) &
}

run_infer_split() {
  local gpu_id="$1"
  local split="$2"
  local cfg="${INFER_CFG_ROOT}/${split}/${VARIANT}.toml"
  echo "[infer] task=${TASK} variant=${VARIANT} split=${split} gpu=${gpu_id} cfg=${cfg}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export NPROC_PER_NODE=1
    bash "${PROJECT_ROOT}/scripts/infer.sh" "${cfg}"
  ) &
}

wait_all() {
  local status=0
  for pid in "$@"; do
    wait "${pid}" || status=$?
  done
  return ${status}
}

train_pids=()
for i in 0 1 2 3; do
  run_train_split "$i" "${SPLITS[$i]}"
  train_pids+=("$!")
done
wait_all "${train_pids[@]}"

infer_pids=()
for i in 0 1 2 3; do
  run_infer_split "$i" "${SPLITS[$i]}"
  infer_pids+=("$!")
done
wait_all "${infer_pids[@]}"

echo "[done] task=${TASK} variant=${VARIANT}"
