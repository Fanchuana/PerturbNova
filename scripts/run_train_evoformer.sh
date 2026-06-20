#!/bin/bash
# Train Evoformer AE on Node 2 (4 GPUs)
# Usage: bash scripts/run_train_evoformer.sh

set -e

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova

echo "=========================================="
echo "Training Evoformer AE"
echo "=========================================="

conda run -n my_state torchrun --nproc_per_node=4 scripts/train_autoencoder.py \
    --config configs/autoencoder_train_evoformer.toml

echo ""
echo "Training complete! Check outputs in: ./outputs/autoencoder_train_evoformer_hepg2"
