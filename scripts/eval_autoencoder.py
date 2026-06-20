#!/usr/bin/env python3
"""
Evaluate Autoencoder performance by comparing reconstruction quality.

This script:
1. Trains MLP VAE and Evoformer AE on training data
2. Reconstructs test data using both autoencoders
3. Runs cell_eval to compare reconstruction quality

Usage:
    python scripts/eval_autoencoder.py \
        --data-config /work/home/cryoem666/czx/project/OPUS-Cell-Refactored/configs/split/Replogle_Nadig_v2/zeroshot_hepg2.toml \
        --output-dir ./outputs/autoencoder_eval
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
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perturbnova.vae import VAE, build_vae_module
from perturbnova.evoformer_ae import EvoformerAutoencoder, build_evoformer_ae_module


class AnnDataset(Dataset):
    """Dataset wrapper for AnnData."""

    def __init__(self, adata: sc.AnnData, n_cells: Optional[int] = None):
        """
        Args:
            adata: AnnData object
            n_cells: Number of cells to sample per epoch (None = use all)
        """
        if hasattr(adata.X, 'toarray'):
            self.X = adata.X.toarray().astype(np.float32)
        else:
            self.X = np.array(adata.X, dtype=np.float32)

        self.n_cells = n_cells
        self.n_total = self.X.shape[0]

    def __len__(self):
        return self.n_total if self.n_cells is None else self.n_cells

    def __getitem__(self, idx):
        if self.n_cells is not None:
            idx = np.random.randint(0, self.n_total)
        return torch.tensor(self.X[idx], dtype=torch.float32)


def load_data(data_config_path: str) -> Dict[str, sc.AnnData]:
    """Load data based on config file."""
    import tomllib

    with open(data_config_path, 'rb') as f:
        config = tomllib.load(f)

    # Load main dataset
    dataset_path = list(config['datasets'].values())[0]
    print(f"Loading data from {dataset_path}")
    adata = sc.read_h5ad(dataset_path)

    # Split based on config
    result = {}

    # Get cell type column (usually 'cell_line' or 'cell_type')
    obs_keys = config.get('obs_keys', {})
    cell_type_col = obs_keys.get('cell_type', 'cell_line')
    pert_col = obs_keys.get('perturbation', 'gene')

    if cell_type_col not in adata.obs.columns:
        # Try common alternatives
        for col in ['cell_line', 'cell_type', 'CellType']:
            if col in adata.obs.columns:
                cell_type_col = col
                break

    print(f"Using cell type column: {cell_type_col}")
    print(f"Cell types: {adata.obs[cell_type_col].unique()}")

    # Handle zeroshot splits
    if 'zeroshot' in config:
        for key, role in config['zeroshot'].items():
            # Parse "replogle.hepg2" -> cell_type = "hepg2"
            parts = key.split('.')
            cell_type = parts[-1] if len(parts) > 1 else parts[0]

            # Find cells of this type
            mask = adata.obs[cell_type_col].str.lower() == cell_type.lower()
            subset = adata[mask].copy()

            if role == 'test':
                result['test'] = subset
                print(f"Test set ({cell_type}): {subset.shape[0]} cells")
            elif role == 'val':
                result['val'] = subset
                print(f"Val set ({cell_type}): {subset.shape[0]} cells")

    # Training set = everything not in test/val
    test_indices = set()
    if 'test' in result:
        test_indices.update(result['test'].obs_names)
    if 'val' in result:
        test_indices.update(result['val'].obs_names)

    train_mask = ~adata.obs_names.isin(test_indices)
    result['train'] = adata[train_mask].copy()
    print(f"Train set: {result['train'].shape[0]} cells")

    return result


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    log_interval: int = 10,
    model_type: str = "vae",
) -> Dict[str, list]:
    """Train an autoencoder model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)

            if model_type == "vae":
                # VAE forward
                latent = model.encoder(x)
                reconstructed = model.decoder(latent)
                loss = nn.MSELoss()(reconstructed, x)
            else:
                # Evoformer AE forward
                loss, _ = model.compute_autoencoder_loss(x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)

                if model_type == "vae":
                    latent = model.encoder(x)
                    reconstructed = model.decoder(latent)
                    loss = nn.MSELoss()(reconstructed, x)
                else:
                    loss, _ = model.compute_autoencoder_loss(x)

                val_losses.append(loss.item())

        scheduler.step()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % log_interval == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4e}, val_loss={val_loss:.4e}")

    return history


def reconstruct_with_autoencoder(
    model: nn.Module,
    data: sc.AnnData,
    device: torch.device,
    batch_size: int = 256,
    model_type: str = "vae",
) -> np.ndarray:
    """Reconstruct data using autoencoder."""
    if hasattr(data.X, 'toarray'):
        X = data.X.toarray().astype(np.float32)
    else:
        X = np.array(data.X, dtype=np.float32)

    model.eval()
    outputs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)

            if model_type == "vae":
                latent = model.encoder(batch)
                reconstructed = model.decoder(latent)
            else:
                reconstructed = model.decode(model.encode(batch))

            outputs.append(reconstructed.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute reconstruction metrics."""
    # MSE
    mse = np.mean((true - pred) ** 2)

    # Pearson correlation per gene
    gene_corrs = []
    for i in range(true.shape[1]):
        if np.std(true[:, i]) > 0 and np.std(pred[:, i]) > 0:
            corr = np.corrcoef(true[:, i], pred[:, i])[0, 1]
            gene_corrs.append(corr)
    mean_gene_corr = np.mean(gene_corrs) if gene_corrs else 0.0

    # Overall correlation
    true_flat = true.flatten()
    pred_flat = pred.flatten()
    overall_corr = np.corrcoef(true_flat, pred_flat)[0, 1] if np.std(true_flat) > 0 and np.std(pred_flat) > 0 else 0.0

    # R2 score
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'mse': float(mse),
        'mean_gene_corr': float(mean_gene_corr),
        'overall_corr': float(overall_corr),
        'r2': float(r2),
    }


def run_cell_eval_simple(
    true_adata: sc.AnnData,
    pred_adata: sc.AnnData,
    control_label: str = "non-targeting",
    pert_col: str = "gene",
) -> Dict[str, float]:
    """Run simplified cell evaluation."""
    # Get unique perturbations (excluding control)
    perts = true_adata.obs[pert_col].unique()
    perts = [p for p in perts if p != control_label]

    metrics_per_pert = {}

    for pert in perts:
        # Get cells for this perturbation
        true_mask = true_adata.obs[pert_col] == pert
        pred_mask = pred_adata.obs[pert_col] == pert

        if true_mask.sum() == 0 or pred_mask.sum() == 0:
            continue

        true_cells = true_adata[true_mask].X
        pred_cells = pred_adata[pred_mask].X

        if hasattr(true_cells, 'toarray'):
            true_cells = true_cells.toarray()
        if hasattr(pred_cells, 'toarray'):
            pred_cells = pred_cells.toarray()

        # Compute mean expression per perturbation
        true_mean = np.mean(true_cells, axis=0)
        pred_mean = np.mean(pred_cells, axis=0)

        # Correlation of mean profiles
        if np.std(true_mean) > 0 and np.std(pred_mean) > 0:
            corr = np.corrcoef(true_mean, pred_mean)[0, 1]
            metrics_per_pert[pert] = float(corr)

    if not metrics_per_pert:
        return {'mean_pert_corr': 0.0, 'median_pert_corr': 0.0}

    corrs = list(metrics_per_pert.values())
    return {
        'mean_pert_corr': float(np.mean(corrs)),
        'median_pert_corr': float(np.median(corrs)),
        'num_perturbations': len(corrs),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Autoencoder performance")
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to data split config (e.g., zeroshot_hepg2.toml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/autoencoder_eval",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent dimension",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and load existing checkpoints",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    data = load_data(args.data_config)

    train_adata = data['train']
    test_adata = data['test']
    n_genes = train_adata.shape[1]
    print(f"Number of genes: {n_genes}")

    # Create datasets
    train_dataset = AnnDataset(train_adata, n_cells=10000)
    val_dataset = AnnDataset(test_adata, n_cells=2000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ================================================================
    # Train MLP VAE
    # ================================================================
    print("\n" + "="*60)
    print("Training MLP VAE...")
    print("="*60)

    vae_model = VAE(
        num_genes=n_genes,
        device=str(device),
        hidden_dim=args.latent_dim,
    ).to(device)

    vae_path = os.path.join(args.output_dir, "vae_model.pt")
    if not args.skip_training or not os.path.exists(vae_path):
        vae_history = train_autoencoder(
            vae_model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            model_type="vae",
        )
        torch.save(vae_model.state_dict(), vae_path)
        print(f"Saved VAE model to {vae_path}")
    else:
        vae_model.load_state_dict(torch.load(vae_path, map_location=device))
        print(f"Loaded VAE model from {vae_path}")

    # ================================================================
    # Train Evoformer AE
    # ================================================================
    print("\n" + "="*60)
    print("Training Evoformer AE...")
    print("="*60)

    evo_config = {
        'enabled': True,
        'latent_dim': args.latent_dim,
        'freeze': False,
        'n_gene': max(10, min(200, n_genes // 200)),
        'n_gene_feat': 32,
        'n_pair_feat': 16,
        'n_embed': 1280,
        'num_evoformer_blocks': 6,
    }

    evo_model = build_evoformer_ae_module(evo_config, input_dim=n_genes, device=device)

    evo_path = os.path.join(args.output_dir, "evoformer_model.pt")
    if not args.skip_training or not os.path.exists(evo_path):
        evo_history = train_autoencoder(
            evo_model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            model_type="evoformer",
        )
        torch.save(evo_model.state_dict(), evo_path)
        print(f"Saved Evoformer AE model to {evo_path}")
    else:
        evo_model.load_state_dict(torch.load(evo_path, map_location=device))
        print(f"Loaded Evoformer AE model from {evo_path}")

    # ================================================================
    # Reconstruct test data
    # ================================================================
    print("\n" + "="*60)
    print("Reconstructing test data...")
    print("="*60)

    # Get true test data
    if hasattr(test_adata.X, 'toarray'):
        true_X = test_adata.X.toarray().astype(np.float32)
    else:
        true_X = np.array(test_adata.X, dtype=np.float32)

    # Reconstruct with VAE
    print("Reconstructing with VAE...")
    vae_reconstructed = reconstruct_with_autoencoder(
        vae_model, test_adata, device, model_type="vae"
    )

    # Reconstruct with Evoformer AE
    print("Reconstructing with Evoformer AE...")
    evo_reconstructed = reconstruct_with_autoencoder(
        evo_model, test_adata, device, model_type="evoformer"
    )

    # ================================================================
    # Compute metrics
    # ================================================================
    print("\n" + "="*60)
    print("Computing metrics...")
    print("="*60)

    # Overall reconstruction metrics
    vae_metrics = compute_metrics(true_X, vae_reconstructed)
    evo_metrics = compute_metrics(true_X, evo_reconstructed)

    print("\nMLP VAE Reconstruction Metrics:")
    for k, v in vae_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nEvoformer AE Reconstruction Metrics:")
    for k, v in evo_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Cell eval metrics (perturbation-level)
    print("\n" + "="*60)
    print("Running cell evaluation...")
    print("="*60)

    # Create AnnData objects with reconstructed data
    vae_adata = test_adata.copy()
    vae_adata.X = vae_reconstructed

    evo_adata = test_adata.copy()
    evo_adata.X = evo_reconstructed

    # Determine perturbation column
    pert_col = 'gene'
    for col in ['gene', 'perturbation', 'pert']:
        if col in test_adata.obs.columns:
            pert_col = col
            break

    # Determine control label
    control_label = 'non-targeting'
    for label in ['non-targeting', 'control', 'non_targeting', 'NT']:
        if label in test_adata.obs[pert_col].values:
            control_label = label
            break

    print(f"Using perturbation column: {pert_col}")
    print(f"Using control label: {control_label}")

    vae_cell_eval = run_cell_eval_simple(test_adata, vae_adata, control_label, pert_col)
    evo_cell_eval = run_cell_eval_simple(test_adata, evo_adata, control_label, pert_col)

    print("\nMLP VAE Cell Eval Metrics:")
    for k, v in vae_cell_eval.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nEvoformer AE Cell Eval Metrics:")
    for k, v in evo_cell_eval.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ================================================================
    # Save results
    # ================================================================
    results = {
        'vae': {
            'reconstruction_metrics': vae_metrics,
            'cell_eval_metrics': vae_cell_eval,
        },
        'evoformer': {
            'reconstruction_metrics': evo_metrics,
            'cell_eval_metrics': evo_cell_eval,
        },
        'config': {
            'data_config': args.data_config,
            'n_genes': n_genes,
            'latent_dim': args.latent_dim,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
        },
    }

    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # ================================================================
    # Print summary
    # ================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Metric':<25} {'MLP VAE':<15} {'Evoformer AE':<15} {'Winner':<10}")
    print("-" * 65)

    for metric in ['mse', 'mean_gene_corr', 'overall_corr', 'r2']:
        vae_val = vae_metrics[metric]
        evo_val = evo_metrics[metric]

        if metric == 'mse':
            winner = 'MLP VAE' if vae_val < evo_val else 'Evoformer AE'
        else:
            winner = 'MLP VAE' if vae_val > evo_val else 'Evoformer AE'

        print(f"{metric:<25} {vae_val:<15.4f} {evo_val:<15.4f} {winner:<10}")

    print(f"\n{'Cell Eval Metric':<25} {'MLP VAE':<15} {'Evoformer AE':<15} {'Winner':<10}")
    print("-" * 65)

    for metric in ['mean_pert_corr', 'median_pert_corr']:
        vae_val = vae_cell_eval.get(metric, 0)
        evo_val = evo_cell_eval.get(metric, 0)

        winner = 'MLP VAE' if vae_val > evo_val else 'Evoformer AE'
        print(f"{metric:<25} {vae_val:<15.4f} {evo_val:<15.4f} {winner:<10}")

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
