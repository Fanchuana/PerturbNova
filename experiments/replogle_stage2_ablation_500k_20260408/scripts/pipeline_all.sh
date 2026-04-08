#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="/work/home/cryoem666/xyf/temp/pycharm/PerturbNova"
OUT_ROOT="${PROJECT_ROOT}/outputs/experiments/replogle_stage2_ablation_500k_20260408"
LOG_DIR="${EXP_DIR}/runtime_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

STAGE1_ZEROSHOT_JOBID="${STAGE1_ZEROSHOT_JOBID:-146990}"
STAGE1_FEWSHOT_JOBID="${STAGE1_FEWSHOT_JOBID:-146991}"
TRAIN_ZEROSHOT_JOBID="${TRAIN_ZEROSHOT_JOBID:-146990}"
TRAIN_FEWSHOT_JOBID="${TRAIN_FEWSHOT_JOBID:-146991}"
INFER_ZEROSHOT_JOBID="${INFER_ZEROSHOT_JOBID:-146990}"
INFER_FEWSHOT_JOBID="${INFER_FEWSHOT_JOBID:-146991}"
SPLITS=(hepg2 jurkat k562 rpe1)

log() {
  echo "[$(date '+%F %T')] $*"
}

job_alive() {
  local jobid="$1"
  squeue -h -j "${jobid}" | grep -q .
}

require_jobs() {
  for jobid in "$@"; do
    if ! job_alive "${jobid}"; then
      echo "[ERROR] job ${jobid} is not active in SLURM queue"
      exit 1
    fi
  done
}

check_stage1_outputs() {
  local missing=0
  for task in zeroshot fewshot; do
    for split in "${SPLITS[@]}"; do
      local ckpt="${OUT_ROOT}/${task}/${split}/stage1/vae_checkpoints/latest.pt"
      if [[ ! -f "${ckpt}" ]]; then
        echo "[MISSING] ${ckpt}"
        missing=1
      fi
    done
  done
  return ${missing}
}

check_group_train_outputs() {
  local task="$1"
  local variant="$2"
  local missing=0
  for split in "${SPLITS[@]}"; do
    local ckpt="${OUT_ROOT}/${task}/${split}/${variant}/checkpoints/latest.pt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[MISSING] ${ckpt}"
      missing=1
    fi
  done
  return ${missing}
}

check_group_infer_outputs() {
  local task="$1"
  local variant="$2"
  local missing=0
  for split in "${SPLITS[@]}"; do
    local agg="${OUT_ROOT}/${task}/${split}/${variant}_infer/cell_eval/${split}_agg_results.csv"
    if [[ ! -f "${agg}" ]]; then
      echo "[MISSING] ${agg}"
      missing=1
    fi
  done
  return ${missing}
}

run_script() {
  local name="$1"
  shift
  log "START ${name}"
  (cd "${SCRIPT_DIR}" && "$@")
  log "DONE  ${name}"
}

run_infer_pair() {
  local task_a="$1"
  local variant_a="$2"
  local task_b="$3"
  local variant_b="$4"
  log "START infer ${task_a}/${variant_a} on job ${INFER_ZEROSHOT_JOBID}"
  (cd "${SCRIPT_DIR}" && bash ./launch_infer_group.sh "${INFER_ZEROSHOT_JOBID}" "${task_a}" "${variant_a}") &
  pid_a=$!
  log "START infer ${task_b}/${variant_b} on job ${INFER_FEWSHOT_JOBID}"
  (cd "${SCRIPT_DIR}" && bash ./launch_infer_group.sh "${INFER_FEWSHOT_JOBID}" "${task_b}" "${variant_b}") &
  pid_b=$!
  status=0
  wait ${pid_a} || status=$?
  wait ${pid_b} || status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "[ERROR] infer pair failed"
    exit ${status}
  fi
  log "DONE  infer pair ${task_a}/${variant_a} + ${task_b}/${variant_b}"
}

main() {
  log "Pipeline log: ${LOG_FILE}"
  require_jobs "${STAGE1_ZEROSHOT_JOBID}" "${STAGE1_FEWSHOT_JOBID}" "${TRAIN_ZEROSHOT_JOBID}" "${TRAIN_FEWSHOT_JOBID}" "${INFER_ZEROSHOT_JOBID}" "${INFER_FEWSHOT_JOBID}"
  log "alloc stage1: zeroshot=${STAGE1_ZEROSHOT_JOBID}, fewshot=${STAGE1_FEWSHOT_JOBID}"
  log "alloc train : zeroshot=${TRAIN_ZEROSHOT_JOBID}, fewshot=${TRAIN_FEWSHOT_JOBID}"
  log "alloc infer : zeroshot=${INFER_ZEROSHOT_JOBID}, fewshot=${INFER_FEWSHOT_JOBID}"

  run_script stage1_bootstrap env ZEROSHOT_JOBID="${STAGE1_ZEROSHOT_JOBID}" FEWSHOT_JOBID="${STAGE1_FEWSHOT_JOBID}" bash ./launch_stage1_bootstrap.sh
  check_stage1_outputs || { echo "[ERROR] stage1 bootstrap outputs incomplete"; exit 1; }

  run_script stage2_wave1 env ZEROSHOT_JOBID="${TRAIN_ZEROSHOT_JOBID}" FEWSHOT_JOBID="${TRAIN_FEWSHOT_JOBID}" bash ./launch_stage2_wave1.sh
  check_group_train_outputs zeroshot frozen_500k || { echo "[ERROR] zeroshot/frozen_500k missing outputs"; exit 1; }
  check_group_train_outputs zeroshot unfrozen_500k || { echo "[ERROR] zeroshot/unfrozen_500k missing outputs"; exit 1; }
  run_infer_pair zeroshot frozen_500k zeroshot unfrozen_500k
  check_group_infer_outputs zeroshot frozen_500k || { echo "[ERROR] zeroshot/frozen_500k infer missing outputs"; exit 1; }
  check_group_infer_outputs zeroshot unfrozen_500k || { echo "[ERROR] zeroshot/unfrozen_500k infer missing outputs"; exit 1; }

  run_script stage2_wave2 env ZEROSHOT_JOBID="${TRAIN_ZEROSHOT_JOBID}" FEWSHOT_JOBID="${TRAIN_FEWSHOT_JOBID}" bash ./launch_stage2_wave2.sh
  check_group_train_outputs zeroshot v1_500k || { echo "[ERROR] zeroshot/v1_500k missing outputs"; exit 1; }
  check_group_train_outputs fewshot frozen_500k || { echo "[ERROR] fewshot/frozen_500k missing outputs"; exit 1; }
  run_infer_pair zeroshot v1_500k fewshot frozen_500k
  check_group_infer_outputs zeroshot v1_500k || { echo "[ERROR] zeroshot/v1_500k infer missing outputs"; exit 1; }
  check_group_infer_outputs fewshot frozen_500k || { echo "[ERROR] fewshot/frozen_500k infer missing outputs"; exit 1; }

  run_script stage2_wave3 env ZEROSHOT_JOBID="${TRAIN_ZEROSHOT_JOBID}" FEWSHOT_JOBID="${TRAIN_FEWSHOT_JOBID}" bash ./launch_stage2_wave3.sh
  check_group_train_outputs fewshot unfrozen_500k || { echo "[ERROR] fewshot/unfrozen_500k missing outputs"; exit 1; }
  check_group_train_outputs fewshot v1_500k || { echo "[ERROR] fewshot/v1_500k missing outputs"; exit 1; }
  run_infer_pair fewshot unfrozen_500k fewshot v1_500k
  check_group_infer_outputs fewshot unfrozen_500k || { echo "[ERROR] fewshot/unfrozen_500k infer missing outputs"; exit 1; }
  check_group_infer_outputs fewshot v1_500k || { echo "[ERROR] fewshot/v1_500k infer missing outputs"; exit 1; }

  log "START summarize_results"
  PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" /work/home/cryoem666/miniconda3/envs/my_state/bin/python "${SCRIPT_DIR}/summarize_results.py"
  log "DONE  summarize_results"
  log "All ablation stages finished successfully."
}

main "$@"
