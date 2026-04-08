#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <mode:train|infer> <task:zeroshot|fewshot> <group>"
  exit 2
fi
MODE="$1"
TASK="$2"
GROUP="$3"
SPLITS=(hepg2 jurkat k562 rpe1)
PROJECT_ROOT="/work/home/cryoem666/xyf/temp/pycharm/PerturbNova"
EXP_ROOT="/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408"
CONFIG_ROOT="${EXP_ROOT}/generated_configs/${MODE}/${TASK}"

run_one() {
  local gpu_id="$1"
  local split="$2"
  local cfg="${CONFIG_ROOT}/${split}/${GROUP}.toml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[ablation] missing config: ${cfg}"
    exit 1
  fi
  echo "[ablation] mode=${MODE} task=${TASK} split=${split} group=${GROUP} gpu=${gpu_id}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export NPROC_PER_NODE=1
    if [[ "${MODE}" == "train" ]]; then
      bash "${PROJECT_ROOT}/scripts/train.sh" "${cfg}"
    elif [[ "${MODE}" == "infer" ]]; then
      bash "${PROJECT_ROOT}/scripts/infer.sh" "${cfg}"
    else
      echo "Unsupported MODE=${MODE}"
      exit 1
    fi
  ) &
}

pids=()
for i in 0 1 2 3; do
  run_one "$i" "${SPLITS[$i]}"
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
