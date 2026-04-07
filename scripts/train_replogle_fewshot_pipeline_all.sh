#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_MODE="${PIPELINE_MODE:-stage1_then_stage2}"

run_batch() {
  local run_name="$1"
  RUN_NAME="${run_name}" \
  RUN_MODE="${RUN_MODE:-sequential}" \
  TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-4}" \
  bash "${PROJECT_ROOT}/scripts/train_replogle_fewshot_all.sh"
}

case "${PIPELINE_MODE}" in
  stage1_only)
    run_batch "stage1"
    ;;
  stage2_only)
    run_batch "stage2"
    ;;
  stage2_unfrozen_vae_only)
    run_batch "stage2_unfrozen_vae"
    ;;
  stage1_then_stage2)
    run_batch "stage1"
    run_batch "stage2"
    ;;
  stage1_then_stage2_unfrozen_vae)
    run_batch "stage1"
    run_batch "stage2_unfrozen_vae"
    ;;
  *)
    echo "Unsupported PIPELINE_MODE=${PIPELINE_MODE}"
    echo "Supported modes: stage1_only, stage2_only, stage2_unfrozen_vae_only, stage1_then_stage2, stage1_then_stage2_unfrozen_vae"
    exit 1
    ;;
esac
