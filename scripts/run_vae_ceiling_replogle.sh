#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 ]]; then
  echo "Usage: bash ${PROJECT_ROOT}/scripts/run_vae_ceiling_replogle.sh <fewshot|zeroshot> <hepg2|jurkat|k562|rpe1>"
  exit 2
fi

TASK="$1"
CELL="$2"
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
python "${PROJECT_ROOT}/scripts/vae_ceiling_replogle.py" --task "${TASK}" --cell "${CELL}" "${@:3}"

OUT_DIR="${PROJECT_ROOT}/outputs/experiments/replogle_stage2_ablation_500k_20260408/${TASK}/${CELL}/stage1/vae_ceiling_${TASK}_${CELL}"
CONFIG_PATH="${OUT_DIR}/vae_ceiling_infer_config.toml"
echo "[PerturbNova] running cell_eval for ${CONFIG_PATH}"
python -m perturbnova.cli.cell_eval --config "${CONFIG_PATH}"
