export CUDA_VISIBLE_DEVICES=0
NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/train.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/generated_configs/train/zeroshot/k562/frozen_1500k.toml

NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/infer.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/generated_configs/infer/zeroshot/k562/frozen_1500k.toml

NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/train.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/generated_configs/train/fewshot/k562/frozen_1500k.toml

NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/infer.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/generated_configs/infer/fewshot/k562/frozen_1500k.toml