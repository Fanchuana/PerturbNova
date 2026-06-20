#!/bin/bash
# Four-GPU training only. Evaluation is run later in a single process.

set -eo pipefail

MODEL_TYPE=${1:?MODEL_TYPE required: vae or evoformer}
OUTPUT_DIR=${2:?OUTPUT_DIR required}
SPLIT=${3:-zeroshot_hepg2}
MASTER_PORT=${MASTER_PORT:-29630}

if [[ "$MODEL_TYPE" != "vae" && "$MODEL_TYPE" != "evoformer" ]]; then
  echo "Disabled model type: $MODEL_TYPE" >&2
  echo "Active encoder baselines are plain VAE autoencoder and Evoformer only." >&2
  exit 1
fi

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

COMMON_ARGS=(
  --model-type "$MODEL_TYPE"
  --split "$SPLIT"
  --latent-dim 128
  --epochs 200
  --batch-size 256
  --lr 5e-4
  --validate-every 10
  --save-every 10
  --skip-final-eval
  --output-dir "$OUTPUT_DIR"
  --ddp-timeout-minutes 120
)

if [[ "$MODEL_TYPE" == "evoformer" ]]; then
  COMMON_ARGS+=(
    --evo-n-gene 10
    --evo-n-gene-feat 32
    --evo-n-pair-feat 16
    --evo-n-embed 1280
    --evo-num-blocks 6
  )
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --master_port="$MASTER_PORT" \
  --nproc_per_node=4 \
  scripts/run_autoencoder_pipeline.py \
  "${COMMON_ARGS[@]}"
