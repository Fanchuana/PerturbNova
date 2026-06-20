#!/bin/bash
# 200-epoch VAE vs Evoformer comparison after fixing zeroshot control handling.
# Train on g01n13/g01n14, then run single-process checkpoint sweeps and plot curves.
# Do not start this until the 40-epoch fixed-split suite looks sane.

set -eo pipefail

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
mkdir -p logs outputs

SPLIT=${SPLIT:-zeroshot_hepg2}
VAE_DIR=${VAE_DIR:-outputs/autoencoder_vae_${SPLIT}_200ep_fixedsplit}
EVO_DIR=${EVO_DIR:-outputs/autoencoder_evoformer_${SPLIT}_200ep_fixedsplit}

echo "[fixedsplit-200] starting train-only jobs"
MASTER_PORT=29660 nohup bash scripts/run_200ep_train_only.sh vae "$VAE_DIR" "$SPLIT" \
  > logs/vae_200ep_fixedsplit_train_g01n13.log 2>&1 &
VAE_PID=$!

ssh g01n14 "cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova && mkdir -p logs && MASTER_PORT=29661 nohup bash scripts/run_200ep_train_only.sh evoformer '$EVO_DIR' '$SPLIT' > logs/evoformer_200ep_fixedsplit_train_g01n14.log 2>&1 < /dev/null & echo \$!" \
  > logs/evoformer_200ep_fixedsplit_remote.pid

echo "[fixedsplit-200] VAE local pid: $VAE_PID"
echo "[fixedsplit-200] Evoformer remote pid: $(cat logs/evoformer_200ep_fixedsplit_remote.pid)"

wait "$VAE_PID"
echo "[fixedsplit-200] VAE training finished"

while ssh g01n14 "pgrep -af 'run_200ep_train_only.sh evoformer|run_autoencoder_pipeline.py --model-type evoformer' >/dev/null"; do
  echo "[fixedsplit-200] waiting for Evoformer training..."
  sleep 300
done
echo "[fixedsplit-200] Evoformer training finished"

echo "[fixedsplit-200] evaluating VAE checkpoints on g01n13"
nohup bash scripts/run_200ep_eval_sweep.sh vae "$VAE_DIR" "$SPLIT" \
  > logs/vae_200ep_fixedsplit_eval_g01n13.log 2>&1 &
VAE_EVAL_PID=$!

echo "[fixedsplit-200] evaluating Evoformer checkpoints on g01n14"
ssh g01n14 "cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova && nohup bash scripts/run_200ep_eval_sweep.sh evoformer '$EVO_DIR' '$SPLIT' > logs/evoformer_200ep_fixedsplit_eval_g01n14.log 2>&1 < /dev/null & echo \$!" \
  > logs/evoformer_200ep_fixedsplit_eval_remote.pid

wait "$VAE_EVAL_PID"
echo "[fixedsplit-200] VAE eval finished"

while ssh g01n14 "pgrep -af 'run_200ep_eval_sweep.sh evoformer|eval_checkpoint_sweep.py --model-type evoformer' >/dev/null"; do
  echo "[fixedsplit-200] waiting for Evoformer eval..."
  sleep 300
done
echo "[fixedsplit-200] Evoformer eval finished"

source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

python scripts/plot_checkpoint_curves.py \
  --vae-dir "$VAE_DIR" \
  --evo-dir "$EVO_DIR" \
  --output outputs/checkpoint_curves_200ep_fixedsplit.pdf \
  --csv-output outputs/checkpoint_curves_200ep_fixedsplit.csv

python - <<'PY'
from pathlib import Path
import pandas as pd

csv_path = Path("outputs/checkpoint_curves_200ep_fixedsplit.csv")
df = pd.read_csv(csv_path)
last = df.sort_values("epoch").groupby("model", as_index=False).tail(1).assign(selection="epoch_200")
best = df.sort_values("r2", ascending=False).groupby("model", as_index=False).head(1).assign(selection="best_r2")
out = Path("outputs/performance_table_200ep_fixedsplit.csv")
summary = pd.concat([last, best], ignore_index=True)
summary.to_csv(out, index=False)
print(out)
print(summary.to_string(index=False))
PY

echo "[fixedsplit-200] all done"
