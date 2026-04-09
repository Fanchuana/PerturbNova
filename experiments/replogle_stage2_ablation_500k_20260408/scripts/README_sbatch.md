# sbatch submission mode

This experiment package also supports a state/experiment-style sbatch workflow.

## One job = one task × one training strategy
Each sbatch job requests 1 node with 4 GPUs and runs the 4 splits in parallel:
- hepg2 -> GPU 0
- jurkat -> GPU 1
- k562 -> GPU 2
- rpe1 -> GPU 3

Each split runs:
1. train
2. infer
3. cell_eval

## Submit fewshot
```bash
cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408/scripts
bash batch_submit_fewshot.sh
```

## Submit zeroshot
```bash
bash batch_submit_zeroshot.sh
```

## Submit all 6 jobs
```bash
bash batch_submit_all.sh
```
