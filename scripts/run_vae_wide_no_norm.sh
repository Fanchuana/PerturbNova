#!/bin/bash
# Wider VAE ablation: no latent normalization, larger hidden layers, full flow.

set -e

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova

SPLIT=${1:-zeroshot_hepg2}
if [[ $# -gt 0 && "$1" != --* ]]; then
    shift
else
    SPLIT=zeroshot_hepg2
fi

echo "=========================================="
echo "Wide VAE Ablation"
echo "Split: $SPLIT"
echo "=========================================="

source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export MASTER_PORT=${MASTER_PORT:-29614}

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master_port="$MASTER_PORT" --nproc_per_node=4 scripts/run_autoencoder_pipeline.py \
    --model-type vae \
    --split "$SPLIT" \
    --latent-dim 256 \
    --vae-hidden-dims 2048,2048,2048,2048 \
    --no-vae-normalize-latent \
    --epochs 40 \
    --batch-size 256 \
    --lr 5e-4 \
    --validate-every 10 \
    --save-every 10 \
    --cell-eval-profile full \
    --num-threads 32 \
    --cell-eval-batch-size 500 \
    --cell-eval-skip-metrics pearson_edistance \
    "$@"

echo ""
echo "Pipeline complete!"
