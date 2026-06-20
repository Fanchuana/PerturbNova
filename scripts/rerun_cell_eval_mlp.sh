#!/bin/bash
# Rerun full cell-eval for the saved MLP VAE outputs.

set -e

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova

conda run --no-capture-output -n my_state python scripts/rerun_cell_eval.py \
    --output-dir outputs/autoencoder_vae_zeroshot_hepg2 \
    --profile full \
    --num-threads 64 \
    --batch-size 500 \
    "$@"
