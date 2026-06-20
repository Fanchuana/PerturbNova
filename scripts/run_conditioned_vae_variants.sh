#!/bin/bash
# Run one conditioned compression variant with the original VAE width/latent setting.
# Usage: bash scripts/run_conditioned_vae_variants.sh cond_vae [output_dir] [split]

set -eo pipefail

echo "Conditioned/VQ/delta VAE variants are disabled for the current DDPM prep stage."
echo "Use scripts/run_mlp.sh or scripts/run_evoformer.sh instead."
exit 1

MODEL_TYPE=${1:?MODEL_TYPE required: cond_ae, cond_vae, vqvae, or cond_delta_vae}
OUTPUT_DIR=${2:-outputs/autoencoder_${MODEL_TYPE}_zeroshot_hepg2_40ep}
SPLIT=${3:-zeroshot_hepg2}
MASTER_PORT=${MASTER_PORT:-29640}

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --master_port="$MASTER_PORT" \
  --nproc_per_node=4 \
  scripts/run_autoencoder_pipeline.py \
  --model-type "$MODEL_TYPE" \
  --split "$SPLIT" \
  --latent-dim 128 \
  --vae-hidden-dims 1024,1024,1024 \
  --vae-normalize-latent \
  --cond-embed-dim 64 \
  --vae-beta 1e-3 \
  --kl-warmup-epochs 10 \
  --vq-num-codes 512 \
  --vq-commitment-cost 0.25 \
  --delta-loss-weight 1.0 \
  --epochs 40 \
  --batch-size 256 \
  --lr 5e-4 \
  --validate-every 10 \
  --save-every 10 \
  --cell-eval-profile full \
  --num-threads 32 \
  --cell-eval-batch-size 500 \
  --cell-eval-skip-metrics pearson_edistance \
  --output-dir "$OUTPUT_DIR"
