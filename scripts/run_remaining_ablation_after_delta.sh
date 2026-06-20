#!/bin/bash
# Wait for the delta-auxiliary VAE run to finish, then run the remaining
# focused ablations sequentially on the same 4 GPUs.

set -eo pipefail

echo "Ablation variants are disabled for the current DDPM prep stage."
echo "Active encoder baselines are plain VAE autoencoder and Evoformer only."
exit 1

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

while pgrep -f '[v]ae_delta_aux_zeroshot_hepg2_40ep_fixedsplit' >/dev/null; do
  sleep 60
done

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --master_port=29672 \
  --nproc_per_node=4 \
  scripts/run_autoencoder_pipeline.py \
  --model-type vae \
  --split zeroshot_hepg2 \
  --latent-dim 128 \
  --vae-hidden-dims 1024,1024,1024 \
  --no-vae-normalize-latent \
  --epochs 40 \
  --batch-size 256 \
  --lr 5e-4 \
  --validate-every 10 \
  --save-every 10 \
  --cell-eval-profile full \
  --num-threads 32 \
  --cell-eval-batch-size 500 \
  --cell-eval-skip-metrics pearson_edistance \
  --output-dir outputs/vae_no_norm_zeroshot_hepg2_40ep_fixedsplit \
  --ddp-timeout-minutes 120

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --master_port=29673 \
  --nproc_per_node=4 \
  scripts/run_autoencoder_pipeline.py \
  --model-type cond_ae \
  --split zeroshot_hepg2 \
  --latent-dim 128 \
  --vae-hidden-dims 1024,1024,1024 \
  --vae-normalize-latent \
  --cond-embed-dim 64 \
  --condition-use-cell-type \
  --no-condition-use-perturbation \
  --epochs 40 \
  --batch-size 256 \
  --lr 5e-4 \
  --validate-every 10 \
  --save-every 10 \
  --cell-eval-profile full \
  --num-threads 32 \
  --cell-eval-batch-size 500 \
  --cell-eval-skip-metrics pearson_edistance \
  --output-dir outputs/cond_ae_cell_only_zeroshot_hepg2_40ep_fixedsplit \
  --ddp-timeout-minutes 120
