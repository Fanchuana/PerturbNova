# Experiment log: replogle_stage2_ablation_500k_20260408

## 2026-04-08
- 建立正式 500k-step ablation 实验目录。
- 已确认 inference 历史问题：PerturbNova 默认使用 EMA，而旧 Squidiff workflow 使用 raw model；本实验统一改为 `use_ema = false`。
- 当前仓库里仅观察到 `zeroshot/hepg2/stage1` 输出；其余 split/task 需先补 stage1 bootstrap。
- 主训练资源：`146990` (g01n12), `146991` (g01n13)
- 备用资源：`147251` (g01n14), `147252` (g01n15)

## TODO
- [ ] Stage1 bootstrap (8 runs)
- [ ] Stage2 ablation (24 runs)
- [ ] Infer + cell_eval (24 runs)
- [ ] 汇总结果表
