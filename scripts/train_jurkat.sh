NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/train.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/generated_configs/train/zeroshot/jurkat/frozen_1500k.toml

NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/infer.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/generated_configs/infer/zeroshot/jurkat/frozen_1500k.toml

NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/train.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/generated_configs/train/fewshot/jurkat/frozen_1500k.toml

NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/infer.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/generated_configs/infer/fewshot/jurkat/frozen_1500k.toml