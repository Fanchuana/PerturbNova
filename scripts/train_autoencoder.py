#!/usr/bin/env python3
"""
Train Autoencoder with validation (MMD and R2).

This script trains MLP VAE or Evoformer AE with validation on test set.
Validation metrics: MMD and R2.

Usage:
    # Single GPU
    python scripts/train_autoencoder.py --config configs/autoencoder_train.toml

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/train_autoencoder.py --config configs/autoencoder_train.toml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perturbnova.vae import VAE
from perturbnova.evoformer_ae import EvoformerAutoencoder, build_evoformer_ae_module


def setup_distributed():
    """Setup distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class AnnDataset(Dataset):
    """Dataset wrapper for AnnData."""

    def __init__(self, adata: sc.AnnData):
        if hasattr(adata.X, 'toarray'):
            self.X = adata.X.toarray().astype(np.float32)
        else:
            self.X = np.array(adata.X, dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32)


def load_config(config_path: str) -> dict:
    """Load TOML config file."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def load_data(data_path: str, split_config: Optional[str] = None) -> Dict[str, sc.AnnData]:
    """Load and split data."""
    print(f"Loading data from {data_path}")
    adata = sc.read_h5ad(data_path)

    if split_config is None:
        # Random split if no config provided
        n = adata.shape[0]
        indices = np.random.permutation(n)
        n_test = int(n * 0.1)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        result = {
            'train': adata[train_idx].copy(),
            'test': adata[test_idx].copy(),
        }
    else:
        # Load split config
        split = load_config(split_config)

        # Get cell type column
        cell_type_col = 'cell_line'
        for col in ['cell_line', 'cell_type', 'CellType']:
            if col in adata.obs.columns:
                cell_type_col = col
                break

        # Handle zeroshot splits
        test_indices = set()
        if 'zeroshot' in split:
            for key, role in split['zeroshot'].items():
                parts = key.split('.')
                cell_type = parts[-1] if len(parts) > 1 else parts[0]

                if role == 'test':
                    mask = adata.obs[cell_type_col].str.lower() == cell_type.lower()
                    test_indices.update(adata[mask].obs_names)

        # Split data
        test_mask = adata.obs_names.isin(test_indices)
        result = {
            'train': adata[~test_mask].copy(),
            'test': adata[test_mask].copy(),
        }

    print(f"Train: {result['train'].shape[0]} cells")
    print(f"Test: {result['test'].shape[0]} cells")

    return result


def compute_mmd(source: torch.Tensor, target: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> float:
    """Compute Maximum Mean Discrepancy (MMD) between two distributions."""
    source = source.float()
    target = target.float()
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)
    bandwidth = torch.sum(l2_distance.detach()) / float((source.size(0) + target.size(0)) ** 2)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernel_values = [torch.exp(-l2_distance / (bandwidth * (kernel_mul**i))) for i in range(kernel_num)]
    kernels = sum(kernel_values)
    source_count = source.size(0)
    xx = kernels[:source_count, :source_count]
    yy = kernels[source_count:, source_count:]
    xy = kernels[:source_count, source_count:]
    yx = kernels[source_count:, :source_count]
    return float(torch.mean(xx + yy - xy - yx).item())


def compute_r2(true: np.ndarray, pred: np.ndarray) -> float:
    """Compute R2 score."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    model_type: str = "vae",
    max_batches: int = 10,
) -> Dict[str, float]:
    """Validate model and compute MMD and R2."""
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_idx, x in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            x = x.to(device)

            if model_type == "vae":
                latent = model.module.encoder(x) if isinstance(model, DDP) else model.encoder(x)
                reconstructed = model.module.decoder(latent) if isinstance(model, DDP) else model.decoder(latent)
            else:
                if isinstance(model, DDP):
                    latent = model.module.encode(x)
                    reconstructed = model.module.decode(latent)
                else:
                    latent = model.encode(x)
                    reconstructed = model.decode(latent)

            all_true.append(x.cpu())
            all_pred.append(reconstructed.cpu())

    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)

    # Compute MMD
    mmd = compute_mmd(all_true, all_pred)

    # Compute R2
    r2 = compute_r2(all_true.numpy(), all_pred.numpy())

    return {'mmd': mmd, 'r2': r2}


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str = "vae",
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for x in train_loader:
        x = x.to(device)

        if model_type == "vae":
            if isinstance(model, DDP):
                latent = model.module.encoder(x)
                reconstructed = model.module.decoder(latent)
            else:
                latent = model.encoder(x)
                reconstructed = model.decoder(latent)
            loss = F.mse_loss(reconstructed, x)
        else:
            if isinstance(model, DDP):
                loss, _ = model.module.compute_autoencoder_loss(x)
            else:
                loss, _ = model.compute_autoencoder_loss(x)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder with validation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("="*60)
        print("Autoencoder Training with Validation")
        print("="*60)

    # Load data
    data = load_data(
        config['data']['path'],
        config['data'].get('split_config'),
    )

    n_genes = data['train'].shape[1]

    # Create datasets
    train_dataset = AnnDataset(data['train'])
    test_dataset = AnnDataset(data['test'])

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=test_sampler,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
    )

    # Build model
    model_type = config['model']['type']
    latent_dim = config['model']['latent_dim']

    if rank == 0:
        print(f"Model type: {model_type}")
        print(f"Number of genes: {n_genes}")
        print(f"Latent dim: {latent_dim}")

    if model_type == "vae":
        model = VAE(
            num_genes=n_genes,
            device=str(device),
            hidden_dim=latent_dim,
        ).to(device)
    else:
        evo_config = config['model'].get('evoformer', {})
        evo_config['enabled'] = True
        evo_config['latent_dim'] = latent_dim
        evo_config['freeze'] = False

        model = build_evoformer_ae_module(evo_config, input_dim=n_genes, device=device)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Count parameters
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6,
    )

    # Training loop
    epochs = config['training']['epochs']
    validate_every = config['training'].get('validate_every', 5)
    save_every = config['training'].get('save_every', 10)
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_r2 = -float('inf')
    history = {'train_loss': [], 'val_mmd': [], 'val_r2': []}

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, model_type)
        scheduler.step()

        if rank == 0:
            history['train_loss'].append(train_loss)
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4e}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Validate
        if (epoch + 1) % validate_every == 0:
            val_metrics = validate(model, test_loader, device, model_type)

            if rank == 0:
                history['val_mmd'].append(val_metrics['mmd'])
                history['val_r2'].append(val_metrics['r2'])

                print(f"  Val MMD: {val_metrics['mmd']:.4f}")
                print(f"  Val R2: {val_metrics['r2']:.4f}")

                # Save best model
                if val_metrics['r2'] > best_r2:
                    best_r2 = val_metrics['r2']
                    best_path = output_dir / "best_model.pt"
                    torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), best_path)
                    print(f"  Saved best model (R2={best_r2:.4f})")

        # Save checkpoint
        if rank == 0 and (epoch + 1) % save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
            }, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

    # Save final model and history
    if rank == 0:
        final_path = output_dir / "final_model.pt"
        torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), final_path)
        print(f"\nSaved final model to {final_path}")

        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to {history_path}")

        # Save config
        config_save_path = output_dir / "train_config.json"
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nBest validation R2: {best_r2:.4f}")
        print("="*60)
        print("Training complete!")
        print("="*60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
