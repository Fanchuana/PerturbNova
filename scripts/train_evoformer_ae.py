#!/usr/bin/env python3
"""
Training script for Evoformer Autoencoder on single-cell RNA-seq data.

This script demonstrates how to train the Evoformer-based autoencoder
as a standalone model or as part of PerturbNova pipeline.

Usage:
    python train_evoformer_ae.py --config configs/evoformer_ae.toml
    python train_evoformer_ae.py --data_path /path/to/data.h5ad
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perturbnova.evoformer_ae import EvoformerAutoencoder


class SingleCellDataset(Dataset):
    """Dataset for single-cell RNA-seq data."""

    def __init__(
        self,
        data_path: str,
        n_gene_total: int = 20074,
        n_cells: int = 256,
        mask_ratio: float = 0.15,
    ):
        """
        Args:
            data_path: Path to .h5ad file or directory of .h5ad files
            n_gene_total: Total number of genes
            n_cells: Number of cells to sample per batch
            mask_ratio: Ratio of genes to mask for pretraining
        """
        import scanpy as sc

        self.n_gene_total = n_gene_total
        self.n_cells = n_cells
        self.mask_ratio = mask_ratio

        # Load data
        print(f"Loading data from {data_path}")
        if os.path.isdir(data_path):
            # Load multiple files
            self.files = [
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith('.h5ad')
            ]
            print(f"Found {len(self.files)} .h5ad files")
        else:
            self.files = [data_path]

        # Preload first file to get gene count
        adata = sc.read_h5ad(self.files[0])
        if hasattr(adata.X, 'toarray'):
            self.n_genes = adata.X.shape[1]
        else:
            self.n_genes = adata.X.shape[1]
        print(f"Number of genes: {self.n_genes}")

    def __len__(self):
        return len(self.files) * 100  # Virtual length

    def __getitem__(self, idx):
        import scanpy as sc

        # Select random file
        file_idx = idx % len(self.files)
        adata = sc.read_h5ad(self.files[file_idx])

        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray().astype(np.float32)
        else:
            X = np.array(adata.X, dtype=np.float32)

        # Random cell sampling
        n_cells_total = X.shape[0]
        if n_cells_total > self.n_cells:
            cell_idx = np.random.choice(n_cells_total, self.n_cells, replace=False)
            X = X[cell_idx]
        else:
            # Pad with zeros if not enough cells
            pad = np.zeros((self.n_cells - n_cells_total, X.shape[1]), dtype=np.float32)
            X = np.concatenate([X, pad], axis=0)

        # Normalize: library size -> 10k -> log1p
        cell_sum = X.sum(axis=1, keepdims=True) + 1e-8
        X = X / cell_sum * 1e4
        X = np.log1p(X)

        # Create masked version
        n_mask = int(self.n_genes * self.mask_ratio)
        mask_idx = np.random.choice(self.n_genes, n_mask, replace=False)
        mask = np.zeros(self.n_genes, dtype=bool)
        mask[mask_idx] = True

        # Apply mask (set masked genes to 0)
        X_masked = X.copy()
        X_masked[:, mask] = 0

        # For now, use mean across cells as single sample
        # (In production, you'd want to handle batches differently)
        X_mean = X.mean(axis=0)
        X_masked_mean = X_masked.mean(axis=0)

        return (
            torch.tensor(X_masked_mean, dtype=torch.float32),
            torch.tensor(X_mean, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
        )


def train_epoch(
    model: EvoformerAutoencoder,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    start_time = time.time()

    for batch_idx, (sc_data_masked, sc_data_label, mask) in enumerate(dataloader):
        sc_data_masked = sc_data_masked.to(device)
        sc_data_label = sc_data_label.to(device)
        mask = mask.to(device)

        # Forward pass
        loss, results = model.compute_pretrain_loss(
            sc_data_masked,
            sc_data_label,
            mask=mask,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4e} "
                f"Time: {elapsed:.2f}s"
            )

            # Show sample predictions
            if batch_idx % (log_interval * 5) == 0:
                print(f"  True (first 10): {results['true'][0, :10].detach().cpu().numpy()}")
                print(f"  Pred (first 10): {results['pred'][0, :10].detach().cpu().numpy()}")

            start_time = time.time()

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def validate(
    model: EvoformerAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
):
    """Validate the model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for sc_data_masked, sc_data_label, mask in dataloader:
            sc_data_masked = sc_data_masked.to(device)
            sc_data_label = sc_data_label.to(device)
            mask = mask.to(device)

            loss, _ = model.compute_pretrain_loss(
                sc_data_masked,
                sc_data_label,
                mask=mask,
            )

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Evoformer Autoencoder")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .h5ad file or directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/evoformer_ae",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--n_gene_total",
        type=int,
        default=20074,
        help="Total number of genes",
    )
    parser.add_argument(
        "--n_gene",
        type=int,
        default=100,
        help="Number of gene groups",
    )
    parser.add_argument(
        "--n_gene_feat",
        type=int,
        default=32,
        help="Gene feature dimension",
    )
    parser.add_argument(
        "--n_pair_feat",
        type=int,
        default=16,
        help="Pair feature dimension",
    )
    parser.add_argument(
        "--n_embed",
        type=int,
        default=1280,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num_evoformer_blocks",
        type=int,
        default=6,
        help="Number of Evoformer blocks",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent dimension",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.15,
        help="Mask ratio for pretraining",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Logging interval",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Checkpoint saving interval",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create model
    model = EvoformerAutoencoder(
        n_gene_total=args.n_gene_total,
        n_gene=args.n_gene,
        n_gene_feat=args.n_gene_feat,
        n_pair_feat=args.n_pair_feat,
        n_embed=args.n_embed,
        num_evoformer_blocks=args.num_evoformer_blocks,
        latent_dim=args.latent_dim,
    )
    model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 0
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")
    else:
        start_epoch = 0

    # Create dataset and dataloader
    dataset = SingleCellDataset(
        data_path=args.data_path,
        n_gene_total=args.n_gene_total,
        mask_ratio=args.mask_ratio,
    )

    # Split into train/val
    n_val = max(1, len(dataset) // 10)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, args.log_interval
        )
        print(f"Train Loss: {train_loss:.4e}")

        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4e}")

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.2e}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Saved final model to {final_path}")

    # Save model config
    config = {
        "n_gene_total": args.n_gene_total,
        "n_gene": args.n_gene,
        "n_gene_feat": args.n_gene_feat,
        "n_pair_feat": args.n_pair_feat,
        "n_embed": args.n_embed,
        "num_evoformer_blocks": args.num_evoformer_blocks,
        "latent_dim": args.latent_dim,
    }
    import json
    config_path = os.path.join(args.output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved model config to {config_path}")


if __name__ == "__main__":
    main()
