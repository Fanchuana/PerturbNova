#!/bin/bash
# Run autoencoder evaluation on Replogle dataset

set -e

# Configuration
DATA_CONFIG="/work/home/cryoem666/czx/project/OPUS-Cell-Refactored/configs/split/Replogle_Nadig_v2/zeroshot_hepg2.toml"
OUTPUT_DIR="./outputs/autoencoder_eval_hepg2"
EPOCHS=50
BATCH_SIZE=256
LR=1e-3
LATENT_DIM=128

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Autoencoder Evaluation"
echo "=========================================="
echo "Data config: $DATA_CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Latent dim: $LATENT_DIM"
echo "=========================================="

# Run evaluation
conda run -n my_state python scripts/eval_autoencoder.py \
    --data-config $DATA_CONFIG \
    --output-dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --latent-dim $LATENT_DIM

echo ""
echo "Evaluation complete! Results saved to: $OUTPUT_DIR/eval_results.json"
echo ""
cat $OUTPUT_DIR/eval_results.json
