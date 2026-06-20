#!/bin/bash
# Inference Evoformer AE (after training)
# Usage: bash scripts/run_infer_evoformer.sh

set -e

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova

echo "=========================================="
echo "Inference Evoformer AE"
echo "=========================================="

conda run -n my_state torchrun --nproc_per_node=4 scripts/infer_autoencoder.py \
    --config configs/autoencoder_infer_evoformer.toml

echo ""
echo "Inference complete! Check results in: ./outputs/autoencoder_infer_evoformer_hepg2"
echo ""
echo "Cell eval results:"
cat ./outputs/autoencoder_infer_evoformer_hepg2/cell_eval_results.json
