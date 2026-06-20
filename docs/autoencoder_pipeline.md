# Autoencoder Pipeline

## Overview

Unified pipeline script for training, inference, and cell evaluation of autoencoders.

## Quick Start

### Node 1: MLP VAE
```bash
cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
bash scripts/run_mlp.sh
```

### Node 2: Evoformer AE
```bash
cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
bash scripts/run_evoformer.sh
```

## Usage Examples

### Different Cell Types
```bash
# hepg2 (default)
bash scripts/run_mlp.sh zeroshot_hepg2

# jurkat
bash scripts/run_mlp.sh zeroshot_jurkat

# k562
bash scripts/run_mlp.sh zeroshot_k562

# rpe1
bash scripts/run_mlp.sh zeroshot_rpe1
```

### Fewshot Splits
```bash
bash scripts/run_mlp.sh fewshot_hepg2
bash scripts/run_mlp.sh fewshot_jurkat
bash scripts/run_mlp.sh fewshot_k562
bash scripts/run_mlp.sh fewshot_rpe1
```

### Custom Parameters
```bash
# Custom epochs and learning rate
bash scripts/run_mlp.sh zeroshot_hepg2 --epochs 50 --lr 1e-3

# Custom output directory
bash scripts/run_mlp.sh zeroshot_hepg2 --output-dir ./outputs/my_experiment

# Skip training, only inference
bash scripts/run_mlp.sh zeroshot_hepg2 --skip-training --checkpoint-path ./outputs/autoencoder_vae_zeroshot_hepg2/best_model.pt
```

### Direct Python Usage
```bash
# Single GPU
python scripts/run_autoencoder_pipeline.py \
    --model-type vae \
    --split zeroshot_hepg2 \
    --epochs 100

# Multi-GPU
torchrun --nproc_per_node=4 scripts/run_autoencoder_pipeline.py \
    --model-type evoformer \
    --split zeroshot_jurkat \
    --epochs 150 \
    --lr 1e-3
```

## Command Line Arguments

### Data Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | replogle_concat.h5ad | Path to h5ad data file |
| `--split` | zeroshot_hepg2 | Split name |
| `--split-config` | None | Full path to split config (overrides --split) |

### Model Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | (required) | `vae` or `evoformer` |
| `--latent-dim` | 128 | Latent dimension |

### Evoformer Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--evo-n-gene` | 10 | Number of gene groups |
| `--evo-n-gene-feat` | 32 | Gene feature dimension |
| `--evo-n-pair-feat` | 16 | Pair feature dimension |
| `--evo-n-embed` | 1280 | Hidden embedding dimension |
| `--evo-num-blocks` | 6 | Number of Evoformer blocks |

### Training Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 256 | Batch size per GPU |
| `--lr` | 5e-4 | Learning rate |
| `--validate-every` | 5 | Validate every N epochs |
| `--save-every` | 20 | Save checkpoint every N epochs |
| `--num-workers` | 4 | Number of data loading workers |

### Inference Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--infer-batch-size` | 512 | Batch size for inference |
| `--pert-col` | gene | Perturbation column name |
| `--control-label` | non-targeting | Control label |

### Output Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | auto | Output directory |
| `--skip-training` | False | Skip training |
| `--checkpoint-path` | None | Checkpoint path for inference |

## Output Structure

```
outputs/autoencoder_{model_type}_{split}/
├── best_model.pt              # Best model (by validation R2)
├── final_model.pt             # Final model
├── checkpoint_epoch_*.pt      # Periodic checkpoints
├── training_history.json      # Training metrics history
├── config.json                # Pipeline configuration
├── reconstructed.h5ad         # Reconstructed test data
├── results.json               # Summary results
└── cell_eval_detailed.json    # Detailed cell evaluation
```

## Metrics

### Training Validation
- **MMD**: Maximum Mean Discrepancy (lower is better)
- **R2**: R-squared score (higher is better)

### Final Results
- **Overall Metrics**: mse, mean_gene_corr, overall_corr, r2
- **Cell Eval Metrics**: mean_pert_corr, median_pert_corr

## Split Names

Available splits in `/work/home/cryoem666/czx/project/OPUS-Cell-Refactored/configs/split/Replogle_Nadig_v2/`:

| Split Name | Description |
|------------|-------------|
| `zeroshot_hepg2` | HepG2 cell line as test |
| `zeroshot_jurkat` | Jurkat cell line as test |
| `zeroshot_k562` | K562 cell line as test |
| `zeroshot_rpe1` | RPE1 cell line as test |
| `fewshot_hepg2` | HepG2 fewshot split |
| `fewshot_jurkat` | Jurkat fewshot split |
| `fewshot_k562` | K562 fewshot split |
| `fewshot_rpe1` | RPE1 fewshot split |

## Example Workflow

```bash
# 1. Train both models in parallel on two nodes

# Node 1
bash scripts/run_mlp.sh zeroshot_hepg2 --epochs 100

# Node 2
bash scripts/run_evoformer.sh zeroshot_hepg2 --epochs 100

# 2. After training, compare results

# MLP VAE results
cat outputs/autoencoder_vae_zeroshot_hepg2/results.json

# Evoformer AE results
cat outputs/autoencoder_evoformer_zeroshot_hepg2/results.json

# 3. Try different cell types
bash scripts/run_mlp.sh zeroshot_jurkat
bash scripts/run_evoformer.sh zeroshot_jurkat
```
