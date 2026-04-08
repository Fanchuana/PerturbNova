# PerturbNova

`PerturbNova` is a refactored training and inference project for the FiLM-based single-cell perturbation diffusion workflow in this repository.

## What Changed

- Hardcoded paths, fold mappings, and dataset column assumptions were removed and replaced with TOML configs.
- Training, inference, dataset preparation, checkpointing, and model construction were decoupled.
- DDP multi-GPU training and distributed multi-GPU inference are supported through `torchrun`.
- Model naming and data field naming were normalized around `perturbation`, `batch`, and `cell_type`.
- Training artifacts now save the full config snapshot and categorical vocabularies together with checkpoints.

## Design Notes

- The task-specific stack is new and isolated under `PerturbNova/src/perturbnova`.
- The diffusion kernel, timestep respacing, and utility NN primitives are now fully internalized inside `PerturbNova/core`, so the package can run without importing the sibling `Squidiff` project.
- The previous hidden on-the-fly VAE path dependency is removed. Training feature space is now explicit:
  - direct gene space: `feature_space.source = "X"`
  - latent space already stored in `obsm`: `feature_space.source = "obsm"`

If you still need latent-to-expression decoding at inference time, enable the optional decoder hook in the inference config.

## Quick Start

Use the provided scripts from the `PerturbNova` project directory or from anywhere with absolute paths. Both scripts now require an explicit config path.

```bash
bash PerturbNova/scripts/train.sh PerturbNova/configs/replogle_fewshot/runs/hepg2/stage1.toml
bash PerturbNova/scripts/infer.sh PerturbNova/configs/replogle_fewshot/inference/hepg2/infer.toml
```

For multi-GPU:

```bash
NPROC_PER_NODE=4 bash PerturbNova/scripts/train.sh PerturbNova/configs/replogle_zeroshot/runs/hepg2/stage1.toml
NPROC_PER_NODE=4 bash PerturbNova/scripts/infer.sh PerturbNova/configs/replogle_zeroshot/inference/hepg2/infer.toml
```

The scripts activate the `my_state` conda environment and export only `PerturbNova/src` to `PYTHONPATH`.

## Log Rendering

Training and inference logs can be switched under `[experiment]`:

```toml
[experiment]
log_render = "compact"   # current minimal logs
log_progress_width = 28  # used by rich mode
```

Available modes:

- `compact`: keep the current short single-line logs
- `rich`: colored terminal logs with stage labels, progress bars, and highlighted metrics

## Inference Split Modes

Inference now supports two input patterns:

- pass a pre-sliced `test_only.h5ad` directly through `input.data_path`
- pass the full dataset and let inference slice `train` / `val` / `test` from the same split TOML used in training

Example:

```toml
[input]
data_path = "/path/to/full_dataset.h5ad"
reference_data_path = ""

[input.split]
subset = "test"
dataset_config_path = "/path/to/data_or_run_config.toml"
```

Behavior:

- if `input.split.subset` is empty, inference keeps the old behavior and uses `input.data_path` as-is
- if `input.split.subset` is set to `train`, `val`, or `test`, inference reads the full dataset and slices that subset before sampling
- if `input.split.dataset_config_path` is empty, inference falls back to the dataset config saved inside the training checkpoint

## Inference Cell Eval

`scripts/infer.sh` can now run `cell_eval` automatically after sampling. Enable it in the infer TOML:

```toml
[cell_eval]
enabled = true
outdir = "/path/to/output/cell_eval"
profile = "full"
pert_col = "gene"
celltype_col = "cell_line"
control_pert = "non-targeting"
```

Behavior:

- if `cell_eval.enabled = true`, inference always saves an exact real copy for the sampled subset
- if the sampled subset does not contain control cells, the post-infer evaluator backfills control cells from `input.reference_data_path`, then `input.data_path`, then the training dataset path
- evaluation is run on the model feature space used for training, so HVG / `obsm["X_hvg"]` workflows are supported directly

Inference is now split-specific rather than stage-specific:

- `stage1` cannot be used for perturbation prediction because it only trains the VAE
- each split has one infer config under `inference/<split>/infer.toml`
- in that infer config, set `checkpoint.path` to the final diffusion checkpoint you want to evaluate, whether it came from `stage2` or `stage2_unfrozen_vae`

## Replogle few-shot migration example

This project now includes a hierarchical Replogle config tree under:

- `PerturbNova/configs/replogle/common/`
- `PerturbNova/configs/replogle/data/source/`
- `PerturbNova/configs/replogle/data/views/`
- `PerturbNova/configs/replogle/model/`
- `PerturbNova/configs/replogle/training/`
- `PerturbNova/configs/replogle/inference/`
- `PerturbNova/configs/replogle_fewshot/runs/`
- `PerturbNova/configs/replogle_fewshot/inference/`
- `PerturbNova/configs/replogle_zeroshot/runs/`
- `PerturbNova/configs/replogle_zeroshot/inference/`

Batch launch:

```bash
bash PerturbNova/scripts/train_replogle_fewshot_all.sh
```

Useful modes:

```bash
# sequential, one split after another, each split can use multiple GPUs
RUN_NAME=stage2 RUN_MODE=sequential TRAIN_NPROC_PER_NODE=4 bash PerturbNova/scripts/train_replogle_fewshot_all.sh

# four splits at once, one GPU per split
RUN_NAME=stage2 RUN_MODE=parallel_single_gpu bash PerturbNova/scripts/train_replogle_fewshot_all.sh
```

## Replogle zero-shot migration example

Zero-shot runs are now available under:

- `PerturbNova/configs/replogle_zeroshot/runs/`
- `PerturbNova/configs/replogle_zeroshot/inference/`

Batch launch:

```bash
bash PerturbNova/scripts/train_replogle_zeroshot_all.sh
```

Useful modes:

```bash
RUN_NAME=stage1 RUN_MODE=parallel_single_gpu bash PerturbNova/scripts/train_replogle_zeroshot_all.sh
PIPELINE_MODE=stage1_then_stage2 RUN_MODE=sequential TRAIN_NPROC_PER_NODE=4 bash PerturbNova/scripts/train_replogle_zeroshot_pipeline_all.sh
PIPELINE_MODE=stage1_then_stage2_unfrozen_vae RUN_MODE=sequential TRAIN_NPROC_PER_NODE=4 bash PerturbNova/scripts/train_replogle_zeroshot_pipeline_all.sh
```

## Config Layers

The intended ownership is:

- `replogle/common/`: experiment-wide defaults such as seed and distributed backend
- `replogle/data/source/`: raw state-style dataset configs with data source, dataset name, task type, and train/val/test split rules
- `replogle/data/views/`: training-side data view config such as feature space, HVG usage, obs keys, and control settings
- `replogle/model/`: model and diffusion architecture defaults
- `replogle/training/`: stage-specific optimization behavior such as `stage1`, `stage2`, and `stage2_unfrozen_vae`
- `replogle/inference/`: shared inference defaults
- `replogle_fewshot/runs/`: final runnable few-shot training configs
- `replogle_fewshot/inference/`: final runnable few-shot inference configs
- `replogle_zeroshot/runs/`: final runnable zero-shot training configs
- `replogle_zeroshot/inference/`: final runnable zero-shot inference configs

`PerturbNova` supports both stage-oriented and mode-oriented training semantics. The user-facing recommended flow is:

- `stage1`: train only the VAE
- `stage2`: train diffusion with VAE frozen
- `stage2_unfrozen_vae`: train diffusion while continuing to update VAE

The stage normalization is implemented in [config.py](/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/src/perturbnova/config.py#L259).

VAE options still live under `[vae]`:

```toml
[vae]
enabled = true
checkpoint_path = "/path/to/vae_checkpoint.pt"
pretrained_state_dir = "/path/to/scimilarity_dir"
latent_dim = 128
freeze = true
reconstruction_loss_weight = 0.0
decode_predictions = true
```

Initialization behavior:

- If `checkpoint_path` is set, the VAE loads a full checkpoint.
- Else if `pretrained_state_dir` is set, the VAE initializes from scimilarity-style `encoder.ckpt` and `decoder.ckpt`, dropping the first encoder layer and the last decoder layer to match the legacy `scDiffusion` workflow.

Example configs:

- stage1 from scimilarity init: `PerturbNova/configs/replogle_fewshot/runs/hepg2/stage1.toml`
- stage2 with frozen VAE: `PerturbNova/configs/replogle_fewshot/runs/hepg2/stage2.toml`
- stage2 with unfrozen VAE: `PerturbNova/configs/replogle_fewshot/runs/hepg2/stage2_unfrozen_vae.toml`

Batch helpers:

```bash
# run one stage across four splits
RUN_NAME=stage1 RUN_MODE=parallel_single_gpu bash PerturbNova/scripts/train_replogle_fewshot_all.sh

# full two-stage pipeline: stage1 then stage2
PIPELINE_MODE=stage1_then_stage2 RUN_MODE=sequential TRAIN_NPROC_PER_NODE=4 bash PerturbNova/scripts/train_replogle_fewshot_pipeline_all.sh

# stage1, then stage2 while continuing to update the VAE
PIPELINE_MODE=stage1_then_stage2_unfrozen_vae RUN_MODE=sequential TRAIN_NPROC_PER_NODE=4 bash PerturbNova/scripts/train_replogle_fewshot_pipeline_all.sh
```
