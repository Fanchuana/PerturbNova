#!/bin/bash
# Train MLP VAE on Node 1 (4 GPUs)
# Usage: bash scripts/run_train_mlp.sh

set -e

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova

echo "=========================================="
echo "Training MLP VAE"
echo "=========================================="

conda run -n my_state torchrun --nproc_per_node=4 scripts/train_autoencoder.py \
    --config configs/autoencoder_train_mlp.toml

echo ""
echo "Training complete! Check outputs in: ./outputs/autoencoder_train_mlp_hepg2"
