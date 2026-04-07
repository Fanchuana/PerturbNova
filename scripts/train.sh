#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash ${PROJECT_ROOT}/scripts/train.sh <config.toml>"
  exit 2
fi
CONFIG_PATH="$1"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

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

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m perturbnova.cli.train --config "${CONFIG_PATH}"
else
  exec python -m perturbnova.cli.train --config "${CONFIG_PATH}"
fi
