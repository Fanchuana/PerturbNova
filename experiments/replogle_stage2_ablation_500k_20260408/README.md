# replogle_stage2_ablation_500k_20260408

## Goal
比较 3 个 stage2 方案在相同训练预算下的性能：

1. `frozen_500k`：冻结 VAE，纯 diffusion stage2
2. `unfrozen_500k`：不冻结 VAE，joint diffusion + VAE
3. `v1_500k`：不冻结 VAE + 多组合 loss (`xstart_mse`, `effect_batch_mse`, `effect_cosine`)

维度：
- 4 splits: `hepg2`, `jurkat`, `k562`, `rpe1`
- 2 tasks: `zeroshot`, `fewshot`
- 共 24 个 stage2 训练实验

## Unified hyperparameters
- `batch_size = 256`
- `microbatch_size = 256`
- `max_steps = 500000`
- `lr_anneal_steps = 500000`
- `log_every_steps = 100`
- `save_every_steps = 50000`
- `schedule_sampler = uniform`
- `seed = 42`
- `evaluation.enabled = false`
- infer 统一 `checkpoint.use_ema = false`
- infer 统一 `diffusion.timestep_respacing = ""`

## SLURM plan
主训练节点：
- job `146990` -> g01n12
- job `146991` -> g01n13

备用节点：
- job `147251` -> g01n14
- job `147252` -> g01n15

推荐策略：
- Phase 0: 用 g01n12/g01n13 各跑 1 个 task 的 stage1 bootstrap（4-split 并行）
- Phase 1: 用这两个整节点分 3 个 wave 跑完 24 个 stage2 ablation
- Phase 2: 用同样的 group 脚本跑 infer + cell_eval

## Layout
- `generated_configs/train/...`
- `generated_configs/infer/...`
- `scripts/`
- `run_matrix.csv`
- `LOG.md`
- outputs root: `/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/outputs/experiments/replogle_stage2_ablation_500k_20260408`
