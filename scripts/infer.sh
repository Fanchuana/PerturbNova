#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash ${PROJECT_ROOT}/scripts/infer.sh <config.toml>"
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
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m perturbnova.cli.infer --config "${CONFIG_PATH}"
else
  python -m perturbnova.cli.infer --config "${CONFIG_PATH}"
fi

CELL_EVAL_ENABLED="$(
python - "${CONFIG_PATH}" <<'PY'
import sys
from perturbnova.config import load_infer_config

config = load_infer_config(sys.argv[1])
print("1" if config["cell_eval"]["enabled"] else "0")
PY
)"

if [[ "${CELL_EVAL_ENABLED}" == "1" ]]; then
  echo "[PerturbNova] running cell_eval for ${CONFIG_PATH}"
  python -m perturbnova.cli.cell_eval --config "${CONFIG_PATH}"
fi
