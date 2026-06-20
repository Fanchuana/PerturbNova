#!/bin/bash
# 40-epoch smoke/full-flow suite after fixing zeroshot control handling.
# Runs one model per invocation so jobs can be placed on different nodes.
# Usage:
#   MASTER_PORT=29650 bash scripts/run_40ep_fixed_split_suite.sh vae
#   MASTER_PORT=29651 bash scripts/run_40ep_fixed_split_suite.sh evoformer

set -eo pipefail

MODEL_TYPE=${1:?MODEL_TYPE required: vae or evoformer}
shift
if [[ "$MODEL_TYPE" != "vae" && "$MODEL_TYPE" != "evoformer" ]]; then
  echo "Disabled model type: $MODEL_TYPE" >&2
  echo "Active encoder baselines are plain VAE autoencoder and Evoformer only." >&2
  exit 1
fi
SPLIT=${SPLIT:-zeroshot_hepg2}
if [[ $# -gt 0 && "$1" != --* ]]; then
  SPLIT=$1
  shift
fi
MASTER_PORT=${MASTER_PORT:-29650}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/autoencoder_${MODEL_TYPE}_${SPLIT}_40ep_fixedsplit}

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

ARGS=(
  --model-type "$MODEL_TYPE"
  --split "$SPLIT"
  --latent-dim 128
  --vae-hidden-dims 1024,1024,1024
  --vae-normalize-latent
  --epochs 40
  --batch-size 256
  --lr 5e-4
  --validate-every 10
  --save-every 10
  --cell-eval-profile full
  --num-threads 32
  --cell-eval-batch-size 500
  --cell-eval-skip-metrics pearson_edistance
  --output-dir "$OUTPUT_DIR"
  --ddp-timeout-minutes 120
)

if [[ "$MODEL_TYPE" == "evoformer" ]]; then
  ARGS+=(
    --evo-n-gene 10
    --evo-n-gene-feat 32
    --evo-n-pair-feat 16
    --evo-n-embed 1280
    --evo-num-blocks 6
  )
fi

if [[ "$MODEL_TYPE" == "cond_ae" || "$MODEL_TYPE" == "cond_vae" || "$MODEL_TYPE" == "vqvae" || "$MODEL_TYPE" == "cond_delta_vae" ]]; then
  ARGS+=(
    --cond-embed-dim 64
    --condition-use-cell-type
    --condition-use-perturbation
    --vae-beta 1e-3
    --kl-warmup-epochs 10
    --vq-num-codes 512
    --vq-commitment-cost 0.25
    --delta-loss-weight 1.0
  )
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --master_port="$MASTER_PORT" \
  --nproc_per_node=4 \
  scripts/run_autoencoder_pipeline.py \
  "${ARGS[@]}" \
  "$@"
