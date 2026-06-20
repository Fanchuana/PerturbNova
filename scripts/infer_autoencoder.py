#!/usr/bin/env python3
"""
Inference and Cell Eval for Autoencoder.

This script:
1. Loads a trained autoencoder
2. Reconstructs test data
3. Runs cell evaluation

Usage:
    # Single GPU
    python scripts/infer_autoencoder.py --config configs/autoencoder_infer.toml

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/infer_autoencoder.py --config configs/autoencoder_infer.toml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perturbnova.vae import VAE
from perturbnova.evoformer_ae import EvoformerAutoencoder, build_evoformer_ae_module


def setup_distributed():
    """Setup distributed inference."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class AnnDataset(Dataset):
    """Dataset wrapper for AnnData."""

    def __init__(self, adata: sc.AnnData):
        if hasattr(adata.X, 'toarray'):
            self.X = adata.X.toarray().astype(np.float32)
        else:
            self.X = np.array(adata.X, dtype=np.float32)
        self.obs_names = adata.obs_names.tolist()
        self.obs = adata.obs.copy()

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


def load_test_data(data_path: str, split_config: str, cell_type: Optional[str] = None) -> sc.AnnData:
    """Load test data based on split config."""
    print(f"Loading data from {data_path}")
    adata = sc.read_h5ad(data_path)

    split = load_config(split_config)

    # Get cell type column
    cell_type_col = 'cell_line'
    for col in ['cell_line', 'cell_type', 'CellType']:
        if col in adata.obs.columns:
            cell_type_col = col
            break

    # Get test cell types
    test_cell_types = []
    if 'zeroshot' in split:
        for key, role in split['zeroshot'].items():
            if role == 'test':
                parts = key.split('.')
                ct = parts[-1] if len(parts) > 1 else parts[0]
                test_cell_types.append(ct)

    if cell_type:
        # Use specified cell type
        mask = adata.obs[cell_type_col].str.lower() == cell_type.lower()
    elif test_cell_types:
        # Use all test cell types
        mask = adata.obs[cell_type_col].str.lower().isin([ct.lower() for ct in test_cell_types])
    else:
        raise ValueError("No test cell types found in config")

    test_adata = adata[mask].copy()
    print(f"Test data: {test_adata.shape[0]} cells, {test_adata.shape[1]} genes")
    print(f"Cell types: {test_adata.obs[cell_type_col].unique()}")

    return test_adata


def build_model(config: dict, n_genes: int, device: torch.device) -> nn.Module:
    """Build model based on config."""
    model_type = config['model']['type']
    latent_dim = config['model']['latent_dim']

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
        evo_config['freeze'] = True  # Freeze for inference

        model = build_evoformer_ae_module(evo_config, input_dim=n_genes, device=device)

    return model


def reconstruct(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    model_type: str = "vae",
    rank: int = 0,
) -> np.ndarray:
    """Reconstruct data using model."""
    model.eval()
    all_reconstructed = []

    with torch.no_grad():
        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)

            if model_type == "vae":
                if isinstance(model, DDP):
                    latent = model.module.encoder(x)
                    reconstructed = model.module.decoder(latent)
                else:
                    latent = model.encoder(x)
                    reconstructed = model.decoder(latent)
            else:
                if isinstance(model, DDP):
                    latent = model.module.encode(x)
                    reconstructed = model.module.decode(latent)
                else:
                    latent = model.encode(x)
                    reconstructed = model.decode(latent)

            # Clamp to reasonable range
            reconstructed = torch.clamp(reconstructed, min=0.0, max=10.0)

            all_reconstructed.append(reconstructed.cpu().numpy())

            if rank == 0 and (batch_idx + 1) % 10 == 0:
                print(f"  Reconstructed {batch_idx + 1} batches...")

    return np.concatenate(all_reconstructed, axis=0)


def run_cell_eval(
    true_adata: sc.AnnData,
    pred_adata: sc.AnnData,
    output_dir: Path,
    pert_col: str = "gene",
    control_label: str = "non-targeting",
) -> Dict[str, float]:
    """Run cell evaluation."""
    print("\nRunning cell evaluation...")

    # Get unique perturbations (excluding control)
    perts = true_adata.obs[pert_col].unique()
    perts = [p for p in perts if p != control_label]
    print(f"Found {len(perts)} perturbations")

    metrics_per_pert = {}

    for i, pert in enumerate(perts):
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

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(perts)} perturbations...")

    # Compute summary metrics
    if not metrics_per_pert:
        summary = {'mean_pert_corr': 0.0, 'median_pert_corr': 0.0, 'num_perturbations': 0}
    else:
        corrs = list(metrics_per_pert.values())
        summary = {
            'mean_pert_corr': float(np.mean(corrs)),
            'median_pert_corr': float(np.median(corrs)),
            'std_pert_corr': float(np.std(corrs)),
            'min_pert_corr': float(np.min(corrs)),
            'max_pert_corr': float(np.max(corrs)),
            'num_perturbations': len(corrs),
        }

    # Save detailed results
    results = {
        'summary': summary,
        'per_perturbation': metrics_per_pert,
    }

    results_path = output_dir / "cell_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved cell eval results to {results_path}")

    return summary


def compute_overall_metrics(true_X: np.ndarray, pred_X: np.ndarray) -> Dict[str, float]:
    """Compute overall reconstruction metrics."""
    # MSE
    mse = float(np.mean((true_X - pred_X) ** 2))

    # Pearson correlation per gene
    gene_corrs = []
    for i in range(true_X.shape[1]):
        if np.std(true_X[:, i]) > 0 and np.std(pred_X[:, i]) > 0:
            corr = np.corrcoef(true_X[:, i], pred_X[:, i])[0, 1]
            gene_corrs.append(corr)
    mean_gene_corr = float(np.mean(gene_corrs)) if gene_corrs else 0.0

    # Overall correlation
    true_flat = true_X.flatten()
    pred_flat = pred_X.flatten()
    overall_corr = float(np.corrcoef(true_flat, pred_flat)[0, 1]) if np.std(true_flat) > 0 and np.std(pred_flat) > 0 else 0.0

    # R2 score
    ss_res = np.sum((true_X - pred_X) ** 2)
    ss_tot = np.sum((true_X - true_X.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mse': mse,
        'mean_gene_corr': mean_gene_corr,
        'overall_corr': overall_corr,
        'r2': r2,
    }


def main():
    parser = argparse.ArgumentParser(description="Autoencoder Inference and Cell Eval")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--cell-type", type=str, default=None, help="Specific cell type to evaluate")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("="*60)
        print("Autoencoder Inference and Cell Eval")
        print("="*60)

    # Load test data
    test_adata = load_test_data(
        config['data']['path'],
        config['data']['split_config'],
        cell_type=args.cell_type,
    )

    n_genes = test_adata.shape[1]

    # Create dataset and loader
    test_dataset = AnnDataset(test_adata)
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if world_size > 1 else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        sampler=test_sampler,
        num_workers=config['inference'].get('num_workers', 4),
        pin_memory=True,
    )

    # Build and load model
    model = build_model(config, n_genes, device)

    checkpoint_path = config['inference']['checkpoint_path']
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Wrap with DDP for distributed inference
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Reconstruct
    if rank == 0:
        print("\nReconstructing test data...")

    model_type = config['model']['type']
    reconstructed = reconstruct(model, test_loader, device, model_type, rank)

    # Gather results from all ranks
    if world_size > 1:
        # Gather all reconstructed data
        gathered = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, reconstructed)
        if rank == 0:
            reconstructed = np.concatenate(gathered, axis=0)

    if rank == 0:
        # Create output directory
        output_dir = Path(config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get true data
        if hasattr(test_adata.X, 'toarray'):
            true_X = test_adata.X.toarray().astype(np.float32)
        else:
            true_X = np.array(test_adata.X, dtype=np.float32)

        # Compute overall metrics
        print("\nComputing overall metrics...")
        overall_metrics = compute_overall_metrics(true_X, reconstructed)

        print("\nOverall Reconstruction Metrics:")
        for k, v in overall_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Create predicted AnnData
        pred_adata = test_adata.copy()
        pred_adata.X = reconstructed

        # Save reconstructed data
        pred_path = output_dir / "reconstructed.h5ad"
        pred_adata.write_h5ad(pred_path)
        print(f"\nSaved reconstructed data to {pred_path}")

        # Run cell eval
        pert_col = config['evaluation'].get('pert_col', 'gene')
        control_label = config['evaluation'].get('control_label', 'non-targeting')

        cell_eval_metrics = run_cell_eval(test_adata, pred_adata, output_dir, pert_col, control_label)

        print("\nCell Evaluation Metrics:")
        for k, v in cell_eval_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Save all results
        results = {
            'overall_metrics': overall_metrics,
            'cell_eval_metrics': cell_eval_metrics,
            'config': config,
        }

        results_path = output_dir / "inference_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved all results to {results_path}")

        print("\n" + "="*60)
        print("Inference complete!")
        print("="*60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
