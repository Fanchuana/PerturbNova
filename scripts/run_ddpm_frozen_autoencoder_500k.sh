#!/bin/bash
# Train a plain frozen-autoencoder latent DDPM for 500k steps.
# Usage:
#   bash scripts/run_ddpm_frozen_autoencoder_500k.sh vae
#   bash scripts/run_ddpm_frozen_autoencoder_500k.sh evoformer

set -eo pipefail

MODEL_TYPE=${1:?MODEL_TYPE required: vae or evoformer}
if [[ "$MODEL_TYPE" != "vae" && "$MODEL_TYPE" != "evoformer" ]]; then
  echo "MODEL_TYPE must be vae or evoformer" >&2
  exit 1
fi

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova

if [[ "$MODEL_TYPE" == "vae" ]]; then
  CONFIG="configs/replogle_zeroshot/runs/hepg2/ddpm_frozen_vae_500k.toml"
  INFER_CONFIG="configs/replogle_zeroshot/inference/hepg2/infer_ddpm_vae.toml"
  OUTPUT_DIR="outputs/ddpm_frozen_vae_zeroshot_hepg2_500k"
  MASTER_PORT=${MASTER_PORT:-29780}
else
  CONFIG="configs/replogle_zeroshot/runs/hepg2/ddpm_frozen_evoformer_500k.toml"
  INFER_CONFIG="configs/replogle_zeroshot/inference/hepg2/infer_ddpm_evoformer.toml"
  OUTPUT_DIR="outputs/ddpm_frozen_evoformer_zeroshot_hepg2_500k"
  MASTER_PORT=${MASTER_PORT:-29781}
fi

source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

export PYTHONPATH="/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/src:${PYTHONPATH:-}"

# ============================================================================
# Step 1: Training
# ============================================================================

echo "=========================================="
echo "Step 1: Training DDPM (frozen $MODEL_TYPE)"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} torchrun \
  --standalone \
  --master_port="$MASTER_PORT" \
  --nproc_per_node=4 \
  -m perturbnova.cli.train \
  --config "$CONFIG"

echo ""
echo "Training complete!"
echo ""

# ============================================================================
# Step 2: Inference (with GPU)
# ============================================================================

echo "=========================================="
echo "Step 2: Inference"
echo "  Config: $INFER_CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} torchrun \
  --standalone \
  --master_port=$((MASTER_PORT + 100)) \
  --nproc_per_node=4 \
  -m perturbnova.cli.infer \
  --config "$INFER_CONFIG" \
  --output-dir "${OUTPUT_DIR}/inference"

echo ""
echo "Inference complete!"
echo ""

# ============================================================================
# Step 3: Cell Eval (CPU only, release GPU)
# ============================================================================

echo "=========================================="
echo "Step 3: Cell Evaluation (CPU)"
echo "=========================================="

# Release GPU memory
export CUDA_VISIBLE_DEVICES=""

python -m perturbnova.cli.cell_eval \
  --config "$INFER_CONFIG" \
  --output-dir "${OUTPUT_DIR}/inference"

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "  Output: $OUTPUT_DIR"
echo "  Inference: $OUTPUT_DIR/inference"
echo "  Cell Eval: $OUTPUT_DIR/inference/cell_eval"
echo "=========================================="
