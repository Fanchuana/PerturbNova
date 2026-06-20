#!/bin/bash
# Focused ablation suite for diagnosing reconstruction vs perturbation signal.
# Usage:
#   MASTER_PORT=29670 bash scripts/run_next_ablation_suite.sh
#
# Runs three new 40-epoch experiments, compared against the existing VAE baseline:
#   1) vae + auxiliary delta loss
#   2) vae without latent normalization
#   3) cond_ae with cell-type only, no perturbation id

set -eo pipefail

echo "Ablation variants are disabled for the current DDPM prep stage."
echo "Active encoder baselines are plain VAE autoencoder and Evoformer only."
exit 1

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

run_one() {
  local name="$1"
  shift
  local master_port="$1"
  shift
  echo
  echo "============================================================"
  echo "Running: $name"
  echo "============================================================"
  MASTER_PORT="$master_port" OUTPUT_DIR="outputs/${name}_zeroshot_hepg2_40ep_fixedsplit" \
    bash scripts/run_40ep_fixed_split_suite.sh "$@"
}

run_one "vae_delta_aux" 29671 vae --delta-recon-loss-weight 0.3
run_one "vae_no_norm" 29672 vae --no-vae-normalize-latent
run_one "cond_ae_cell_only" 29673 cond_ae --no-condition-use-perturbation
