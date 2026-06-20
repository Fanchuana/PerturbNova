#!/bin/bash
# Single-process checkpoint sweep with full cell-eval, then plot curves.

set -eo pipefail

MODEL_TYPE=${1:?MODEL_TYPE required: vae or evoformer}
OUTPUT_DIR=${2:?OUTPUT_DIR required}
SPLIT=${3:-zeroshot_hepg2}

if [[ "$MODEL_TYPE" != "vae" && "$MODEL_TYPE" != "evoformer" ]]; then
  echo "Disabled model type: $MODEL_TYPE" >&2
  echo "Active encoder baselines are plain VAE autoencoder and Evoformer only." >&2
  exit 1
fi

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

ARGS=(
  --model-type "$MODEL_TYPE"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  --latent-dim 128
  --epochs 200
  --eval-every 10
  --infer-batch-size 2048
  --cell-eval-profile full
  --cell-eval-skip-metrics pearson_edistance
  --num-threads 32
  --cell-eval-batch-size 500
  --prefix hepg2
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

CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoint_sweep.py "${ARGS[@]}"
