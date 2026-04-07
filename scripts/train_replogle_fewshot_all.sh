#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_ROOT="${PROJECT_ROOT}/configs/replogle_fewshot/runs"
RUN_MODE="${RUN_MODE:-sequential}"
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-4}"
RUN_NAME="${RUN_NAME:-stage2}"
SPLITS=(hepg2 jurkat k562 rpe1)

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  set -u
elif command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)"
  set -u
else
  echo "[PerturbNova] conda init script not found. Expected: ${CONDA_SH}"
  exit 1
fi
set +u
conda activate my_state
set -u

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

run_single() {
  local split="$1"
  local config_path="${CONFIG_ROOT}/${split}/${RUN_NAME}.toml"
  if [[ ! -f "${config_path}" ]]; then
    echo "[PerturbNova] missing config: ${config_path}"
    exit 1
  fi
  echo "[PerturbNova] launching split=${split} config=${config_path}"
  NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}" bash "${PROJECT_ROOT}/scripts/train.sh" "${config_path}"
}

run_parallel_single_gpu() {
  local -a pids=()
  for gpu_id in 0 1 2 3; do
    local split="${SPLITS[$gpu_id]}"
    local config_path="${CONFIG_ROOT}/${split}/${RUN_NAME}.toml"
    if [[ ! -f "${config_path}" ]]; then
      echo "[PerturbNova] missing config: ${config_path}"
      return 1
    fi
    echo "[PerturbNova] launching split=${split} on GPU ${gpu_id}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export NPROC_PER_NODE=1
      bash "${PROJECT_ROOT}/scripts/train.sh" "${config_path}"
    ) &
    pids+=($!)
  done

  local status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
  done
  return "${status}"
}

case "${RUN_MODE}" in
  sequential)
    for split in "${SPLITS[@]}"; do
      run_single "${split}"
    done
    ;;
  parallel_single_gpu)
    run_parallel_single_gpu
    ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}"
    echo "Supported modes: sequential, parallel_single_gpu"
    exit 1
    ;;
esac
