#!/bin/bash
# MLP VAE Pipeline: Train + checkpoint sweep
# Usage: bash scripts/run_mlp.sh [SPLIT] [EXTRA_ARGS...]

set -e

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova

SPLIT=${1:-zeroshot_hepg2}
if [[ $# -gt 0 && "$1" != --* ]]; then
    shift
else
    SPLIT=zeroshot_hepg2
fi

echo "=========================================="
echo "MLP VAE Pipeline"
echo "Split: $SPLIT"
echo "=========================================="

# Four-GPU distributed run
source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export MASTER_PORT=${MASTER_PORT:-29613}

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master_port="$MASTER_PORT" --nproc_per_node=4 scripts/run_autoencoder_pipeline.py \
    --model-type vae \
    --split $SPLIT \
    --latent-dim 128 \
    --epochs 200 \
    --batch-size 256 \
    --lr 5e-4 \
    --validate-every 10 \
    --save-every 10 \
    --cell-eval-profile full \
    --num-threads 32 \
    --cell-eval-batch-size 500 \
    --cell-eval-skip-metrics pearson_edistance \
    --eval-checkpoints-every 10 \
    --eval-output-subdir checkpoint_eval \
    "$@"

echo ""
echo "Pipeline complete!"
