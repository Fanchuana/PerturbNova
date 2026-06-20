#!/bin/bash
# Inference MLP VAE (after training)
# Usage: bash scripts/run_infer_mlp.sh

set -e

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova

echo "=========================================="
echo "Inference MLP VAE"
echo "=========================================="

conda run -n my_state torchrun --nproc_per_node=4 scripts/infer_autoencoder.py \
    --config configs/autoencoder_infer_mlp.toml

echo ""
echo "Inference complete! Check results in: ./outputs/autoencoder_infer_mlp_hepg2"
echo ""
echo "Cell eval results:"
cat ./outputs/autoencoder_infer_mlp_hepg2/cell_eval_results.json
