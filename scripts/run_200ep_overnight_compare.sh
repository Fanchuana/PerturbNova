#!/bin/bash
# Launch VAE and Evoformer 200-epoch training, evaluate checkpoint sweeps, and plot results.

set -eo pipefail

cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
mkdir -p logs outputs

VAE_DIR=outputs/autoencoder_vae_zeroshot_hepg2_200ep_rerun
EVO_DIR=outputs/autoencoder_evoformer_zeroshot_hepg2_200ep_rerun

echo "[overnight] starting train-only jobs"
nohup bash scripts/run_200ep_train_only.sh vae "$VAE_DIR" zeroshot_hepg2 \
  > logs/vae_200ep_rerun_train_g01n13.log 2>&1 &
VAE_PID=$!

ssh g01n14 "cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova && mkdir -p logs && nohup bash scripts/run_200ep_train_only.sh evoformer '$EVO_DIR' zeroshot_hepg2 > logs/evoformer_200ep_rerun_train_g01n14.log 2>&1 < /dev/null & echo \$!" \
  > logs/evoformer_200ep_rerun_remote.pid

echo "[overnight] VAE local pid: $VAE_PID"
echo "[overnight] Evoformer remote pid: $(cat logs/evoformer_200ep_rerun_remote.pid)"

wait "$VAE_PID"
echo "[overnight] VAE training finished"

while ssh g01n14 "pgrep -af 'run_200ep_train_only.sh evoformer|run_autoencoder_pipeline.py --model-type evoformer' >/dev/null"; do
  echo "[overnight] waiting for Evoformer training..."
  sleep 300
done
echo "[overnight] Evoformer training finished"

echo "[overnight] evaluating VAE checkpoints on g01n13"
nohup bash scripts/run_200ep_eval_sweep.sh vae "$VAE_DIR" zeroshot_hepg2 \
  > logs/vae_200ep_rerun_eval_g01n13.log 2>&1 &
VAE_EVAL_PID=$!

echo "[overnight] evaluating Evoformer checkpoints on g01n14"
ssh g01n14 "cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova && nohup bash scripts/run_200ep_eval_sweep.sh evoformer '$EVO_DIR' zeroshot_hepg2 > logs/evoformer_200ep_rerun_eval_g01n14.log 2>&1 < /dev/null & echo \$!" \
  > logs/evoformer_200ep_rerun_eval_remote.pid

wait "$VAE_EVAL_PID"
echo "[overnight] VAE eval finished"

while ssh g01n14 "pgrep -af 'run_200ep_eval_sweep.sh evoformer|eval_checkpoint_sweep.py --model-type evoformer' >/dev/null"; do
  echo "[overnight] waiting for Evoformer eval..."
  sleep 300
done
echo "[overnight] Evoformer eval finished"

source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
conda activate my_state

python scripts/plot_checkpoint_curves.py \
  --vae-dir "$VAE_DIR" \
  --evo-dir "$EVO_DIR" \
  --output outputs/checkpoint_curves_200ep_rerun.pdf \
  --csv-output outputs/checkpoint_curves_200ep_rerun.csv

python - <<'PY'
from pathlib import Path
import pandas as pd

csv_path = Path("outputs/checkpoint_curves_200ep_rerun.csv")
df = pd.read_csv(csv_path)
last = df.sort_values("epoch").groupby("model", as_index=False).tail(1)
out = Path("outputs/performance_table_200ep_rerun.csv")
last.to_csv(out, index=False)
print(out)
print(last.to_string(index=False))
PY

echo "[overnight] all done"
