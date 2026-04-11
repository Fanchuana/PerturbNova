NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/train.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_20260410_ablation_network_v2/generated_configs/train/zeroshot/hepg2/frozen_1500k.toml

NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/infer.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_20260410_ablation_network_v2/generated_configs/infer/zeroshot/hepg2/frozen_1500k.toml

NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/train.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_20260410_ablation_network_v2/generated_configs/train/fewshot/hepg2/frozen_1500k.toml

NPROC_PER_NODE=1 bash /mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/scripts/infer.sh \
/mnt/shared-storage-user/lvying/s2-project/virtual_cell/PerturbNova/experiments/replogle_20260410_ablation_network_v2/generated_configs/infer/fewshot/hepg2/frozen_1500k.toml