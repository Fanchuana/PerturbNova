# Autoencoder Evaluation Workflow

## Overview

This workflow trains two types of autoencoders (MLP VAE and Evoformer AE) and evaluates them using:
1. **During training**: MMD and R2 on validation set (test set)
2. **After training**: Cell eval on test set

## Workflow

```
Node 1 (4 GPUs)                    Node 2 (4 GPUs)
    │                                   │
    ▼                                   ▼
┌─────────────────┐               ┌─────────────────┐
│  Train MLP VAE  │               │Train Evoformer AE│
│  (with val MMD/R2)│             │  (with val MMD/R2)│
└────────┬────────┘               └────────┬────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐               ┌─────────────────┐
│  Infer MLP VAE  │               │Infer Evoformer AE│
│  + Cell Eval    │               │  + Cell Eval     │
└────────┬────────┘               └────────┬────────┘
         │                                   │
         └───────────────┬───────────────────┘
                         ▼
              ┌─────────────────┐
              │ Compare Results │
              └─────────────────┘
```

## Quick Start

### Step 1: Training (Run on two nodes in parallel)

**Node 1 - Train MLP VAE:**
```bash
cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
bash scripts/run_train_mlp.sh
```

**Node 2 - Train Evoformer AE:**
```bash
cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
bash scripts/run_train_evoformer.sh
```

### Step 2: Inference (After training completes)

**Node 1 - Infer MLP VAE:**
```bash
bash scripts/run_infer_mlp.sh
```

**Node 2 - Infer Evoformer AE:**
```bash
bash scripts/run_infer_evoformer.sh
```

## Config Files

### Training Configs

| Config | Description |
|--------|-------------|
| `configs/autoencoder_train_mlp.toml` | MLP VAE training config |
| `configs/autoencoder_train_evoformer.toml` | Evoformer AE training config |

### Inference Configs

| Config | Description |
|--------|-------------|
| `configs/autoencoder_infer_mlp.toml` | MLP VAE inference config |
| `configs/autoencoder_infer_evoformer.toml` | Evoformer AE inference config |

## Output Structure

```
outputs/
├── autoencoder_train_mlp_hepg2/
│   ├── best_model.pt              # Best model checkpoint (by R2)
│   ├── final_model.pt             # Final model checkpoint
│   ├── training_history.json      # Training history (loss, MMD, R2)
│   └── train_config.json          # Training config
│
├── autoencoder_train_evoformer_hepg2/
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── training_history.json
│   └── train_config.json
│
├── autoencoder_infer_mlp_hepg2/
│   ├── reconstructed.h5ad         # Reconstructed test data
│   ├── cell_eval_results.json     # Cell evaluation results
│   └── inference_results.json     # All metrics
│
└── autoencoder_infer_evoformer_hepg2/
    ├── reconstructed.h5ad
    ├── cell_eval_results.json
    └── inference_results.json
```

## Metrics

### During Training (Validation)

- **MMD**: Maximum Mean Discrepancy between true and reconstructed distributions
- **R2**: R-squared score of reconstruction

### After Inference (Cell Eval)

- **Overall Metrics**:
  - `mse`: Mean squared error
  - `mean_gene_corr`: Mean correlation per gene
  - `overall_corr`: Overall correlation
  - `r2`: R-squared score

- **Cell Eval Metrics**:
  - `mean_pert_corr`: Mean correlation per perturbation
  - `median_pert_corr`: Median correlation per perturbation
  - `num_perturbations`: Number of evaluated perturbations

## Customization

### Change Cell Type

Edit the split config path in the TOML files:

```toml
[data]
split_config = "/work/home/cryoem666/czx/project/OPUS-Cell-Refactored/configs/split/Replogle_Nadig_v2/zeroshot_jurkat.toml"
```

Available splits:
- `zeroshot_hepg2.toml`
- `zeroshot_jurkat.toml`
- `zeroshot_k562.toml`
- `zeroshot_rpe1.toml`

### Change Latent Dimension

```toml
[model]
latent_dim = 64  # or 256, 512, etc.
```

### Change Training Parameters

```toml
[training]
batch_size = 128
epochs = 200
lr = 0.001
```

## Monitoring Training

Training logs show:
```
Epoch 1/100
  Train Loss: 1.2345e-01
  LR: 5.00e-04

Epoch 5/100
  Train Loss: 9.8765e-02
  Val MMD: 0.1234
  Val R2: 0.5678
  Saved best model (R2=0.5678)
```

## Comparing Results

After both inferences complete, compare:

```bash
# MLP VAE results
cat outputs/autoencoder_infer_mlp_hepg2/cell_eval_results.json

# Evoformer AE results
cat outputs/autoencoder_infer_evoformer_hepg2/cell_eval_results.json
```

Look at:
- `mean_pert_corr`: Higher is better
- `median_pert_corr`: Higher is better
- `r2`: Higher is better
- `mse`: Lower is better
