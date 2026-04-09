#!/bin/bash
#SBATCH --job-name=pn_zeroshot_v1_500k
#SBATCH --partition=normal
#SBATCH --nodelist=g01n04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/runtime_logs/sbatch/pn_zeroshot_v1_500k.%j.out
#SBATCH --error=/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/runtime_logs/sbatch/pn_zeroshot_v1_500k.%j.err

export TASK=zeroshot
export VARIANT=v1_500k

bash /work/home/cryoem666/xyf/temp/pycharm/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/scripts/on_node_task_variant.sh
