#!/usr/bin/env python3
"""
Unified Autoencoder Pipeline: Training + Inference + Cell Eval

Usage:
    # MLP VAE
    torchrun --nproc_per_node=4 scripts/run_autoencoder_pipeline.py --model-type vae --split zeroshot_hepg2

    # Evoformer AE
    torchrun --nproc_per_node=4 scripts/run_autoencoder_pipeline.py --model-type evoformer --split zeroshot_hepg2
"""

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perturbnova.vae import ConditionalAutoencoder, ConditionalDeltaVAE, ConditionalVAE, ConditionalVQVAE, VAE
from perturbnova.evoformer_ae import EvoformerAutoencoder, build_evoformer_ae_module


# ============================================================================
# Distributed
# ============================================================================

def setup_distributed(timeout_minutes: int = 60, initialize: bool = True):
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        if initialize:
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=timeout_minutes),
            )
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


# ============================================================================
# Dataset
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        cell_type_ids: np.ndarray | None = None,
        pert_ids: np.ndarray | None = None,
        target_delta: np.ndarray | None = None,
        target_delta_ids: np.ndarray | None = None,
        control_baseline: np.ndarray | None = None,
    ):
        self.X = X
        self.cell_type_ids = cell_type_ids
        self.pert_ids = pert_ids
        self.target_delta = target_delta
        self.target_delta_ids = target_delta_ids
        self.control_baseline = control_baseline

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.cell_type_ids is None or self.pert_ids is None:
            return torch.from_numpy(self.X[idx].copy())
        item = {
            "x": torch.from_numpy(self.X[idx].copy()),
            "cell_type_id": torch.tensor(int(self.cell_type_ids[idx]), dtype=torch.long),
            "pert_id": torch.tensor(int(self.pert_ids[idx]), dtype=torch.long),
        }
        if self.target_delta is not None:
            delta_idx = int(self.target_delta_ids[idx]) if self.target_delta_ids is not None else idx
            item["target_delta_id"] = torch.tensor(delta_idx, dtype=torch.long)
            item["target_delta"] = torch.from_numpy(self.target_delta[delta_idx].copy())
            if self.control_baseline is not None:
                item["control_baseline"] = torch.from_numpy(self.control_baseline[delta_idx].copy())
        return item


# ============================================================================
# Data Loading
# ============================================================================

def load_toml(path: str) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, 'rb') as f:
        return tomllib.load(f)


def _decode_h5_strings(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _read_h5ad_column(group: h5py.Group, column: str):
    obj = group[column]
    if isinstance(obj, h5py.Group) and obj.attrs.get("encoding-type") == "categorical":
        categories = _decode_h5_strings(obj["categories"][:])
        codes = obj["codes"][:]
        values = pd.Categorical.from_codes(codes, categories=categories)
        return values
    values = obj[:]
    if values.dtype.kind in {"O", "S"}:
        return _decode_h5_strings(values)
    return values


def _read_h5ad_obs_minimal(f: h5py.File, columns: list[str]) -> pd.DataFrame:
    obs_group = f["obs"]
    index_key = obs_group.attrs.get("_index", None)
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")

    index = None
    if index_key and index_key in obs_group:
        index = _read_h5ad_column(obs_group, index_key)

    data = {}
    for column in columns:
        if column in obs_group:
            data[column] = _read_h5ad_column(obs_group, column)

    return pd.DataFrame(data, index=index)


def _read_hvg_direct(data_path: str, rank: int = 0) -> Dict[str, Any]:
    """Read only the HVG matrix and required metadata from an h5ad file."""
    start = time.time()
    if is_main_process(rank):
        print("  Fast path: reading obsm/X_hvg directly with h5py", flush=True)

    with h5py.File(data_path, "r") as f:
        if "obsm/X_hvg" not in f:
            raise KeyError("obsm/X_hvg not found")

        X = np.asarray(f["obsm/X_hvg"], dtype=np.float32)

        obs_columns = ["cell_line", "cell_type", "CellType", "gene", "target"]
        obs = _read_h5ad_obs_minimal(f, obs_columns)

        if "var/highly_variable" in f and "var/gene_name_index" in f:
            hvg_mask = f["var/highly_variable"][:].astype(bool)
            all_var_names = np.asarray(_decode_h5_strings(f["var/gene_name_index"][:]), dtype=object)
            var_names = all_var_names[hvg_mask].tolist()
        else:
            var_names = [f"gene_{i}" for i in range(X.shape[1])]

    if len(var_names) != X.shape[1]:
        if is_main_process(rank):
            print(
                f"  Warning: HVG name count ({len(var_names)}) does not match X_hvg width ({X.shape[1]}), using generic names",
                flush=True,
            )
        var_names = [f"gene_{i}" for i in range(X.shape[1])]

    if is_main_process(rank):
        print(f"  X_hvg loaded in {time.time()-start:.1f}s: {X.shape}", flush=True)

    return {"X": X, "obs": obs, "var_names": var_names}


def get_split_config_path(split_name: str) -> str:
    base_dir = "/work/home/cryoem666/czx/project/OPUS-Cell-Refactored/configs/split/Replogle_Nadig_v2"
    return os.path.join(base_dir, f"{split_name}.toml")


def load_data(data_path: str, split_config: str, rank: int = 0) -> Dict[str, Any]:
    """Load HVG data and split."""
    if is_main_process(rank):
        print(f"\n{'='*70}")
        print(f"Loading data...")
        print(f"  File: {data_path}")
        print(f"  Size: {os.path.getsize(data_path) / (1024**3):.1f} GB")
        print(f"{'='*70}", flush=True)

    start = time.time()

    try:
        hvg_data = _read_hvg_direct(data_path, rank)
        X = hvg_data["X"]
        obs = hvg_data["obs"]
        var_names = hvg_data["var_names"]
    except Exception as exc:
        if is_main_process(rank):
            print(f"  Fast HVG read failed ({exc}); falling back to scanpy.read_h5ad", flush=True)

        # Fallback: all ranks load the same data.
        adata = sc.read_h5ad(data_path)

        if 'X_hvg' in adata.obsm:
            X = adata.obsm['X_hvg']
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            if 'highly_variable' in adata.var.columns:
                var_names = adata.var_names[adata.var['highly_variable']].tolist()
            else:
                var_names = [f"gene_{i}" for i in range(X.shape[1])]
        else:
            if 'highly_variable' not in adata.var.columns:
                raise ValueError("No X_hvg or highly_variable column found!") from exc
            adata_hvg = adata[:, adata.var['highly_variable']]
            X = adata_hvg.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            var_names = adata_hvg.var_names.tolist()

        obs = adata.obs.copy()
        del adata

    if is_main_process(rank):
        print(f"  Loaded in {time.time()-start:.1f}s", flush=True)

    # Get split config
    split = load_toml(split_config)
    cell_type_col = 'cell_line'
    for col in ['cell_line', 'cell_type', 'CellType']:
        if col in obs.columns:
            cell_type_col = col
            break

    # Get test cell types
    test_cell_types = []
    if 'zeroshot' in split:
        for key, role in split['zeroshot'].items():
            if role == 'test':
                parts = key.split('.')
                test_cell_types.append(parts[-1])

    if is_main_process(rank):
        print(f"  Cell type: {cell_type_col}", flush=True)
        print(f"  Test types: {test_cell_types}", flush=True)

    pert_col = 'gene' if 'gene' in obs.columns else 'target'
    control_label = 'non-targeting'

    # Split. Match the main PerturbNova zeroshot_holdout behavior:
    # target-cell-line controls are kept in training and also retained in test
    # as cell_eval anchors; only target perturbed cells are truly held out.
    train_mask = None
    test_mask = None
    if test_cell_types:
        cell_values_lower = obs[cell_type_col].astype(str).str.lower()
        is_target = cell_values_lower.isin([ct.lower() for ct in test_cell_types]).to_numpy()
        pert_values = obs[pert_col].astype(str).to_numpy()
        is_control = pert_values == control_label
        target_controls = is_target & is_control
        target_perturbed = is_target & ~is_control
        test_mask = target_perturbed | target_controls
        train_mask = (~test_mask) | target_controls
    else:
        n = X.shape[0]
        test_mask = np.zeros(n, dtype=bool)
        test_mask[np.random.permutation(n)[:int(n*0.1)]] = True
        train_mask = ~test_mask

    all_cell_values = obs[cell_type_col].astype(str).fillna("__missing__").to_numpy()
    all_pert_values = obs[pert_col].astype(str).fillna("__missing__").to_numpy()
    cell_categories = sorted(pd.unique(all_cell_values).tolist())
    pert_categories = sorted(pd.unique(all_pert_values).tolist())
    cell_to_id = {value: idx for idx, value in enumerate(cell_categories)}
    pert_to_id = {value: idx for idx, value in enumerate(pert_categories)}
    all_cell_ids = np.asarray([cell_to_id[value] for value in all_cell_values], dtype=np.int64)
    all_pert_ids = np.asarray([pert_to_id[value] for value in all_pert_values], dtype=np.int64)

    train_X = X[train_mask].copy()
    test_X = X[test_mask].copy()
    train_obs = obs.loc[train_mask].copy()
    test_obs = obs.loc[test_mask].copy()
    train_cell_ids = all_cell_ids[train_mask].copy()
    test_cell_ids = all_cell_ids[test_mask].copy()
    train_pert_ids = all_pert_ids[train_mask].copy()
    test_pert_ids = all_pert_ids[test_mask].copy()
    n_genes = X.shape[1]

    del X

    train_cell_values = train_obs[cell_type_col].astype(str).fillna("__missing__").to_numpy()
    train_pert_values = train_obs[pert_col].astype(str).fillna("__missing__").to_numpy()
    global_control = train_X[train_pert_values == control_label].mean(axis=0) if np.any(train_pert_values == control_label) else train_X.mean(axis=0)
    control_by_cell = {}
    for cell in cell_categories:
        cell_control_mask = (train_cell_values == cell) & (train_pert_values == control_label)
        if np.any(cell_control_mask):
            control_by_cell[cell] = train_X[cell_control_mask].mean(axis=0)

    delta_lookup = []
    control_baseline_lookup = []
    delta_index_by_pair = {}
    train_pair_groups = train_obs.groupby([cell_type_col, pert_col], sort=False).indices
    for (cell, pert), indices in train_pair_groups.items():
        cell = str(cell)
        pert = str(pert)
        control_mean = control_by_cell.get(cell, global_control)
        delta_index_by_pair[(cell_to_id[cell], pert_to_id[pert])] = len(delta_lookup)
        control_baseline_lookup.append(control_mean.astype(np.float32, copy=False))
        delta_lookup.append((train_X[indices].mean(axis=0) - control_mean).astype(np.float32, copy=False))

    control_baseline_lookup.append(global_control.astype(np.float32, copy=False))
    delta_lookup.append(np.zeros(n_genes, dtype=np.float32))
    default_delta_index = len(delta_lookup) - 1
    control_baseline_lookup = np.stack(control_baseline_lookup, axis=0).astype(np.float32, copy=False)
    delta_lookup = np.stack(delta_lookup, axis=0).astype(np.float32, copy=False)

    def build_delta_indices(cell_ids: np.ndarray, pert_ids: np.ndarray) -> np.ndarray:
        out = np.full(cell_ids.shape[0], default_delta_index, dtype=np.int64)
        for i, (cell_id, pert_id) in enumerate(zip(cell_ids, pert_ids)):
            out[i] = delta_index_by_pair.get((int(cell_id), int(pert_id)), default_delta_index)
        return out

    train_delta_ids = build_delta_indices(train_cell_ids, train_pert_ids)
    test_delta_ids = build_delta_indices(test_cell_ids, test_pert_ids)

    if is_main_process(rank):
        print(f"  HVG genes: {n_genes}", flush=True)
        print(f"  Train: {train_X.shape}", flush=True)
        print(f"  Test:  {test_X.shape}", flush=True)
        print(f"  Conditions: cell_types={len(cell_categories)}, perturbations={len(pert_categories)}", flush=True)
        print(f"✓ Data ready in {time.time()-start:.1f}s", flush=True)

    return {
        'train_X': train_X,
        'test_X': test_X,
        'train_cell_type_ids': train_cell_ids,
        'test_cell_type_ids': test_cell_ids,
        'train_pert_ids': train_pert_ids,
        'test_pert_ids': test_pert_ids,
        'delta_lookup': delta_lookup,
        'control_baseline_lookup': control_baseline_lookup,
        'train_delta_ids': train_delta_ids,
        'test_delta_ids': test_delta_ids,
        'test_obs': test_obs,
        'var_names': var_names,
        'n_genes': n_genes,
        'n_cell_types': len(cell_categories),
        'n_perts': len(pert_categories),
        'cell_categories': cell_categories,
        'pert_categories': pert_categories,
        'cell_type_col': cell_type_col,
        'pert_col': pert_col,
        'control_label': control_label,
    }


# ============================================================================
# Metrics
# ============================================================================

def compute_mmd(source: torch.Tensor, target: torch.Tensor, kernel_mul=2.0, kernel_num=5) -> float:
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
    sc = source.size(0)
    xx = kernels[:sc, :sc]
    yy = kernels[sc:, sc:]
    xy = kernels[:sc, sc:]
    yx = kernels[sc:, :sc]
    return float(torch.mean(xx + yy - xy - yx).item())


def compute_r2(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_metrics(true_X: np.ndarray, pred_X: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((true_X - pred_X) ** 2))
    r2 = compute_r2(true_X, pred_X)
    return {'mse': mse, 'r2': r2}


# ============================================================================
# Model
# ============================================================================

def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_model(
    model_type: str,
    n_genes: int,
    latent_dim: int,
    device: torch.device,
    evo_config: Optional[dict] = None,
    vae_hidden_dims: Optional[list[int]] = None,
    vae_normalize_latent: bool = True,
    n_cell_types: int = 1,
    n_perts: int = 1,
    cond_embed_dim: int = 64,
    condition_use_cell_type: bool = True,
    condition_use_perturbation: bool = True,
    vq_num_codes: int = 512,
    vq_commitment_cost: float = 0.25,
    delta_loss_weight: float = 1.0,
) -> nn.Module:
    if model_type == "vae":
        return VAE(
            num_genes=n_genes,
            device=str(device),
            hidden_dim=latent_dim,
            hidden_dims=vae_hidden_dims,
            normalize_latent=vae_normalize_latent,
        ).to(device)
    if model_type == "cond_ae":
        return ConditionalAutoencoder(
            num_genes=n_genes,
            n_cell_types=n_cell_types,
            n_perts=n_perts,
            latent_dim=latent_dim,
            hidden_dims=vae_hidden_dims,
            cond_embed_dim=cond_embed_dim,
            normalize_latent=vae_normalize_latent,
            use_cell_type=condition_use_cell_type,
            use_perturbation=condition_use_perturbation,
        ).to(device)
    if model_type == "cond_vae":
        return ConditionalVAE(
            num_genes=n_genes,
            n_cell_types=n_cell_types,
            n_perts=n_perts,
            latent_dim=latent_dim,
            hidden_dims=vae_hidden_dims,
            cond_embed_dim=cond_embed_dim,
            normalize_latent=vae_normalize_latent,
            use_cell_type=condition_use_cell_type,
            use_perturbation=condition_use_perturbation,
        ).to(device)
    if model_type == "vqvae":
        return ConditionalVQVAE(
            num_genes=n_genes,
            n_cell_types=n_cell_types,
            n_perts=n_perts,
            latent_dim=latent_dim,
            hidden_dims=vae_hidden_dims,
            cond_embed_dim=cond_embed_dim,
            normalize_latent=vae_normalize_latent,
            use_cell_type=condition_use_cell_type,
            use_perturbation=condition_use_perturbation,
            num_codes=vq_num_codes,
            commitment_cost=vq_commitment_cost,
        ).to(device)
    if model_type == "cond_delta_vae":
        return ConditionalDeltaVAE(
            num_genes=n_genes,
            n_cell_types=n_cell_types,
            n_perts=n_perts,
            latent_dim=latent_dim,
            hidden_dims=vae_hidden_dims,
            cond_embed_dim=cond_embed_dim,
            normalize_latent=vae_normalize_latent,
            use_cell_type=condition_use_cell_type,
            use_perturbation=condition_use_perturbation,
            delta_loss_weight=delta_loss_weight,
        ).to(device)
    if model_type == "evoformer":
        config = evo_config or {}
        config['enabled'] = True
        config['latent_dim'] = latent_dim
        config['freeze'] = False
        return build_evoformer_ae_module(config, input_dim=n_genes, device=device)
    raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Training
# ============================================================================

def maybe_mask_evoformer_input(x: torch.Tensor) -> torch.Tensor:
    """Match the original v10_mse_xfy input corruption: mask shared genes in 50% of batches."""
    if torch.rand((), device=x.device) <= 0.5:
        return x

    n_genes = x.shape[1]
    mask_ratio = 0.1 + 0.15 * torch.rand((), device=x.device)
    n_mask = max(1, int(round(n_genes * float(mask_ratio.item()))))
    mask_idx = torch.randperm(n_genes, device=x.device)[:n_mask]

    masked = x.clone()
    masked[:, mask_idx] = 0.0
    return masked


def _move_batch(batch, device):
    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    return batch.to(device)


def _batch_x(batch):
    return batch["x"] if isinstance(batch, dict) else batch


def _conditional_reconstruction(model, batch, model_type: str):
    x = _batch_x(batch)
    m = model.module if isinstance(model, DDP) else model
    if model_type in {"cond_ae", "cond_vae", "vqvae", "cond_delta_vae"}:
        output = m(x, batch["cell_type_id"], batch["pert_id"])
        return output[0] if isinstance(output, tuple) else output
    if model_type == "vae":
        return m.decoder(m.encoder(x))
    return m(x, mode="pretrain")["pred"]


def _top_delta_weights(target_delta: torch.Tensor, alpha: float, max_weight: float) -> torch.Tensor:
    abs_delta = target_delta.abs()
    scale = abs_delta.mean(dim=1, keepdim=True).clamp_min(1e-6)
    weights = 1.0 + alpha * (abs_delta / scale)
    if max_weight > 0:
        weights = weights.clamp(max=max_weight)
    return weights


def _centroid_top_delta_loss(
    recon: torch.Tensor,
    batch: dict,
    top_k: int,
    min_group_size: int,
    delta_weight: float,
    cosine_weight: float,
    contrast_weight: float,
    contrast_genes: int,
    contrast_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Perturbation-level auxiliary objective focused on true top-response genes."""
    if (
        top_k <= 0
        or (delta_weight <= 0 and cosine_weight <= 0 and contrast_weight <= 0)
        or "target_delta" not in batch
        or "control_baseline" not in batch
        or "target_delta_id" not in batch
    ):
        return recon.new_zeros(()), {}

    target_delta = batch["target_delta"]
    control = batch["control_baseline"]
    pred_delta_cell = recon - control
    delta_ids = batch["target_delta_id"]
    n_genes = recon.shape[1]
    k = min(top_k, n_genes)
    min_group_size = max(int(min_group_size), 1)

    delta_losses = []
    cosine_losses = []
    contrast_losses = []
    used_groups = 0

    for delta_id in torch.unique(delta_ids):
        group_mask = delta_ids == delta_id
        if int(group_mask.sum().item()) < min_group_size:
            continue

        target = target_delta[group_mask][0]
        # Skip controls/default pairs with no perturbation response.
        if torch.linalg.vector_norm(target.detach()) < 1e-6:
            continue

        pred = pred_delta_cell[group_mask].mean(dim=0)
        top_idx = torch.topk(target.detach().abs(), k=k, largest=True).indices
        pred_top = pred[top_idx]
        target_top = target[top_idx]

        if delta_weight > 0:
            scale = target_top.detach().abs().mean().clamp_min(1e-4)
            delta_losses.append(F.mse_loss(pred_top / scale, target_top / scale))

        if cosine_weight > 0:
            cosine_losses.append(1.0 - F.cosine_similarity(pred_top, target_top, dim=0, eps=1e-8))

        if contrast_weight > 0 and k < n_genes:
            non_top_mask = torch.ones(n_genes, dtype=torch.bool, device=recon.device)
            non_top_mask[top_idx] = False
            non_top_idx = torch.nonzero(non_top_mask, as_tuple=False).flatten()
            if contrast_genes > 0 and non_top_idx.numel() > contrast_genes:
                perm = torch.randperm(non_top_idx.numel(), device=recon.device)[:contrast_genes]
                non_top_idx = non_top_idx[perm]
            top_abs = pred_top.abs().mean()
            bg_abs = pred[non_top_idx].abs().mean()
            contrast_losses.append(F.relu(bg_abs + contrast_margin - top_abs))

        used_groups += 1

    if used_groups == 0:
        return recon.new_zeros(()), {}

    total = recon.new_zeros(())
    aux = {"centroid_top_delta_groups": recon.new_tensor(float(used_groups))}
    if delta_losses:
        value = torch.stack(delta_losses).mean()
        total = total + delta_weight * value
        aux["centroid_top_delta_mse"] = value.detach()
    if cosine_losses:
        value = torch.stack(cosine_losses).mean()
        total = total + cosine_weight * value
        aux["centroid_top_delta_cosine"] = value.detach()
    if contrast_losses:
        value = torch.stack(contrast_losses).mean()
        total = total + contrast_weight * value
        aux["centroid_top_delta_contrast"] = value.detach()
    return total, aux


def train_epoch(model, loader, optimizer, device, model_type, epoch, rank, args):
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", ncols=100, leave=False, disable=not is_main_process(rank))

    for batch in pbar:
        batch = _move_batch(batch, device)
        x = _batch_x(batch)

        if model_type == "vae":
            m = model.module if isinstance(model, DDP) else model
            recon = m.decoder(m.encoder(x))
            loss = F.mse_loss(recon, x)
            if args.delta_recon_loss_weight > 0 and "target_delta" in batch and "control_baseline" in batch:
                pred_delta = recon - batch["control_baseline"]
                delta_loss = F.mse_loss(pred_delta, batch["target_delta"])
                loss = loss + args.delta_recon_loss_weight * delta_loss
        elif model_type in {"cond_ae", "cond_vae", "vqvae", "cond_delta_vae"}:
            m = model.module if isinstance(model, DDP) else model
            beta = min(args.vae_beta, args.vae_beta * float(epoch + 1) / max(args.kl_warmup_epochs, 1))
            loss, aux = m.compute_loss(
                x,
                batch["cell_type_id"],
                batch["pert_id"],
                target_delta=batch.get("target_delta"),
                beta=beta,
                delta_loss_weight=args.delta_loss_weight,
            )
            if (
                (args.top_delta_recon_loss_weight > 0 or args.top_delta_delta_loss_weight > 0)
                and "target_delta" in batch
                and "control_baseline" in batch
                and "recon" in aux
            ):
                recon = aux["recon"]
                weights = _top_delta_weights(
                    batch["target_delta"],
                    args.top_delta_weight_alpha,
                    args.top_delta_weight_max,
                )
                if args.top_delta_recon_loss_weight > 0:
                    weighted_recon = (weights * (recon - x).pow(2)).mean()
                    loss = loss + args.top_delta_recon_loss_weight * weighted_recon
                if args.top_delta_delta_loss_weight > 0:
                    pred_delta = recon - batch["control_baseline"]
                    weighted_delta = (weights * (pred_delta - batch["target_delta"]).pow(2)).mean()
                    loss = loss + args.top_delta_delta_loss_weight * weighted_delta
            if "recon" in aux:
                centroid_loss, _ = _centroid_top_delta_loss(
                    aux["recon"],
                    batch,
                    args.centroid_top_delta_k,
                    args.centroid_top_delta_min_group_size,
                    args.centroid_top_delta_loss_weight,
                    args.centroid_top_delta_cosine_weight,
                    args.centroid_top_delta_contrast_weight,
                    args.centroid_top_delta_contrast_genes,
                    args.centroid_top_delta_contrast_margin,
                )
                loss = loss + centroid_loss
        else:
            m = model.module if isinstance(model, DDP) else model
            model_input = maybe_mask_evoformer_input(x)
            loss, _ = m.compute_pretrain_loss(model_input, x)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if is_main_process(rank):
            pbar.set_postfix({'loss': f'{loss.item():.4e}'})

    return total_loss / max(n_batches, 1)


def validate(model, loader, device, model_type, rank, max_batches=10):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = _move_batch(batch, device)
            x = _batch_x(batch)
            recon = _conditional_reconstruction(model, batch, model_type)

            all_true.append(x.cpu())
            all_pred.append(recon.cpu())

    all_true = torch.cat(all_true).numpy()
    all_pred = torch.cat(all_pred).numpy()

    return compute_metrics(all_true, all_pred)


# ============================================================================
# Inference
# ============================================================================

def inference(model, loader, device, model_type, rank):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", ncols=100, disable=not is_main_process(rank)):
            batch = _move_batch(batch, device)
            recon = _conditional_reconstruction(model, batch, model_type)

            recon = torch.clamp(recon, min=0.0, max=10.0)
            results.append(recon.cpu().numpy())

    return np.concatenate(results, axis=0)


# ============================================================================
# Cell Eval
# ============================================================================

def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _distributed_barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def run_cell_eval(
    pred_adata,
    real_adata,
    output_dir,
    pert_col,
    control_label,
    celltype_col,
    profile,
    num_threads,
    batch_size,
    skip_metrics: list[str] | None = None,
):
    print("\nRunning cell-eval...", flush=True)

    try:
        from cell_eval import MetricsEvaluator
        from cell_eval.utils import split_anndata_on_celltype
    except ImportError:
        print("Warning: cell-eval not installed, skipping")
        return {}

    ce_dir = output_dir / "cell_eval"
    ce_dir.mkdir(exist_ok=True)

    kwargs = {
        "control_pert": control_label,
        "pert_col": pert_col,
        "de_method": "wilcoxon",
        "num_threads": num_threads,
        "batch_size": batch_size,
        "outdir": str(ce_dir),
        "skip_de": False,
    }

    if celltype_col and celltype_col in real_adata.obs.columns:
        real_split = split_anndata_on_celltype(real_adata, celltype_col)
        pred_split = split_anndata_on_celltype(pred_adata, celltype_col)

        for ct in sorted(real_split):
            print(f"  Evaluating {ct}...", flush=True)
            evaluator = MetricsEvaluator(
                adata_pred=pred_split[ct],
                adata_real=real_split[ct],
                prefix=ct,
                **kwargs,
            )
            evaluator.compute(profile=profile, skip_metrics=skip_metrics, basename="results.csv")
    else:
        evaluator = MetricsEvaluator(
            adata_pred=pred_adata,
            adata_real=real_adata,
            **kwargs,
        )
        evaluator.compute(profile=profile, skip_metrics=skip_metrics, basename="results.csv")

    return {"output_dir": str(ce_dir)}


def evaluate_checkpoint(
    model,
    ckpt_path: Path,
    eval_output_dir: Path,
    test_loader,
    data,
    device,
    model_type,
    rank,
    pert_col,
    control_label,
    celltype_col,
    args,
    epoch: int | None = None,
):
    if is_main_process(rank):
        print(f"\n{'='*70}")
        print("Inference")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Output: {eval_output_dir}")
        print(f"{'='*70}", flush=True)

    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    reconstructed = inference(model, test_loader, device, model_type, rank)

    if is_main_process(rank):
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        true_X = data['test_X']
        metrics = compute_metrics(true_X, reconstructed)

        print(f"\nReconstruction: MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}", flush=True)

        pred_adata = ad.AnnData(X=reconstructed)
        pred_adata.var_names = data['var_names']
        pred_adata.obs = data['test_obs'].copy()

        real_adata = ad.AnnData(X=true_X)
        real_adata.var_names = data['var_names']
        real_adata.obs = data['test_obs'].copy()

        import anndata
        anndata.settings.allow_write_nullable_strings = True
        pred_adata.write_h5ad(eval_output_dir / "reconstructed.h5ad")
        real_adata.write_h5ad(eval_output_dir / "real.h5ad")

        print(f"\n{'='*70}")
        print("Cell Evaluation")
        print(f"{'='*70}", flush=True)

        skip_metrics = _parse_csv_list(args.cell_eval_skip_metrics)
        ce_results = run_cell_eval(
            pred_adata,
            real_adata,
            eval_output_dir,
            pert_col,
            control_label,
            celltype_col,
            args.cell_eval_profile,
            args.num_threads,
            args.cell_eval_batch_size,
            skip_metrics=skip_metrics,
        )

        results = {
            'epoch': epoch,
            'checkpoint_path': str(ckpt_path),
            'metrics': metrics,
            'cell_eval': ce_results,
            'config': vars(args),
        }
        with open(eval_output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nDone! Results: {eval_output_dir}", flush=True)

    _distributed_barrier()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Autoencoder Pipeline")

    # Data
    parser.add_argument("--data-path", type=str,
                        default="/work/home/cryoem666/czx/dataset/STATE/arcinstitute-State-Replogle-Filtered-Dec-6-2025/replogle_concat.h5ad")
    parser.add_argument("--split", type=str, default="zeroshot_hepg2")
    parser.add_argument("--split-config", type=str, default=None)

    # Model
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vae", "evoformer"],
        required=True,
        help="Active encoder families. Conditional/VQ/delta auxiliary variants are disabled.",
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--vae-hidden-dims", type=str, default="1024,1024,1024",
                        help="Comma-separated hidden widths for the VAE encoder/decoder.")
    parser.add_argument("--vae-normalize-latent", dest="vae_normalize_latent", action="store_true", default=True)
    parser.add_argument("--no-vae-normalize-latent", dest="vae_normalize_latent", action="store_false")
    parser.add_argument("--cond-embed-dim", type=int, default=64)
    parser.add_argument("--condition-use-cell-type", dest="condition_use_cell_type", action="store_true", default=True)
    parser.add_argument("--no-condition-use-cell-type", dest="condition_use_cell_type", action="store_false")
    parser.add_argument("--condition-use-perturbation", dest="condition_use_perturbation", action="store_true", default=True)
    parser.add_argument("--no-condition-use-perturbation", dest="condition_use_perturbation", action="store_false")
    parser.add_argument("--vae-beta", type=float, default=1e-3)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--vq-num-codes", type=int, default=512)
    parser.add_argument("--vq-commitment-cost", type=float, default=0.25)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--delta-recon-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary train-only delta loss for plain VAE: MSE(recon-control_mean, train_pair_delta).",
    )
    parser.add_argument(
        "--top-delta-recon-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary conditional-model weighted reconstruction loss emphasizing high absolute train-pair deltas.",
    )
    parser.add_argument(
        "--top-delta-delta-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary conditional-model weighted delta loss emphasizing high absolute train-pair deltas.",
    )
    parser.add_argument(
        "--top-delta-weight-alpha",
        type=float,
        default=2.0,
        help="Strength for per-gene weights: 1 + alpha * abs(delta) / mean(abs(delta)).",
    )
    parser.add_argument(
        "--top-delta-weight-max",
        type=float,
        default=10.0,
        help="Maximum per-gene top-delta weight. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--centroid-top-delta-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary batch/group centroid loss on true top-k response genes.",
    )
    parser.add_argument(
        "--centroid-top-delta-cosine-weight",
        type=float,
        default=0.0,
        help="Auxiliary cosine direction loss on true top-k response genes.",
    )
    parser.add_argument(
        "--centroid-top-delta-contrast-weight",
        type=float,
        default=0.0,
        help="Auxiliary margin loss encouraging top response genes to have larger predicted deltas than background genes.",
    )
    parser.add_argument(
        "--centroid-top-delta-k",
        type=int,
        default=50,
        help="Number of true high-response genes used by centroid top-delta auxiliary losses.",
    )
    parser.add_argument(
        "--centroid-top-delta-min-group-size",
        type=int,
        default=1,
        help="Minimum batch samples with the same train cell/perturbation pair before applying centroid top-delta loss.",
    )
    parser.add_argument(
        "--centroid-top-delta-contrast-genes",
        type=int,
        default=256,
        help="Number of non-top genes sampled for centroid top-delta contrast loss. Set <=0 to use all.",
    )
    parser.add_argument(
        "--centroid-top-delta-contrast-margin",
        type=float,
        default=0.0,
        help="Margin for centroid top-delta contrast loss on absolute predicted delta.",
    )

    # Evoformer
    parser.add_argument("--evo-n-gene", type=int, default=10)
    parser.add_argument("--evo-n-gene-feat", type=int, default=32)
    parser.add_argument("--evo-n-pair-feat", type=int, default=16)
    parser.add_argument("--evo-n-embed", type=int, default=1280)
    parser.add_argument("--evo-num-blocks", type=int, default=6)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)

    # Cell eval
    parser.add_argument("--infer-batch-size", type=int, default=512)
    parser.add_argument("--pert-col", type=str, default=None)
    parser.add_argument("--control-label", type=str, default=None)
    parser.add_argument("--celltype-col", type=str, default=None)
    parser.add_argument("--cell-eval-profile", type=str, default="full")
    parser.add_argument("--cell-eval-skip-metrics", type=str, default="pearson_edistance",
                        help="Comma-separated cell-eval metrics to skip, e.g. pearson_edistance")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--cell-eval-batch-size", type=int, default=100)

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--ddp-timeout-minutes", type=int, default=60,
                        help="NCCL process-group timeout in minutes.")
    parser.add_argument("--eval-checkpoints-every", type=int, default=0,
                        help="After training, evaluate checkpoint_N.pt every N epochs.")
    parser.add_argument("--eval-output-subdir", type=str, default="checkpoint_eval")

    args = parser.parse_args()

    disabled_aux_weights = {
        "delta-recon-loss-weight": args.delta_recon_loss_weight,
        "top-delta-recon-loss-weight": args.top_delta_recon_loss_weight,
        "top-delta-delta-loss-weight": args.top_delta_delta_loss_weight,
        "centroid-top-delta-loss-weight": args.centroid_top_delta_loss_weight,
        "centroid-top-delta-cosine-weight": args.centroid_top_delta_cosine_weight,
        "centroid-top-delta-contrast-weight": args.centroid_top_delta_contrast_weight,
    }
    enabled_disabled_aux = [name for name, value in disabled_aux_weights.items() if value != 0]
    if enabled_disabled_aux:
        parser.error(
            "Disabled experimental auxiliary losses were requested: "
            + ", ".join(enabled_disabled_aux)
            + ". Use the plain vae or evoformer autoencoder settings for now."
        )

    # Setup
    rank, local_rank, world_size = setup_distributed(args.ddp_timeout_minutes, initialize=False)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    split_config = args.split_config or get_split_config_path(args.split)
    output_dir = Path(args.output_dir or f"./outputs/autoencoder_{args.model_type}_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process(rank):
        print(f"\n{'='*70}")
        print(f"Autoencoder Pipeline")
        print(f"  Model: {args.model_type}")
        print(f"  Split: {args.split}")
        print(f"  Output: {output_dir}")
        print(f"{'='*70}", flush=True)

    # Load data
    data = load_data(args.data_path, split_config, rank)
    n_genes = data['n_genes']

    pert_col = args.pert_col or data['pert_col']
    control_label = args.control_label or data['control_label']
    celltype_col = args.celltype_col or data['cell_type_col']

    # Create datasets
    conditional_model = args.model_type in {"cond_ae", "cond_vae", "vqvae", "cond_delta_vae"}
    use_plain_vae_delta_loss = args.model_type == "vae" and args.delta_recon_loss_weight > 0
    use_cond_top_delta_loss = conditional_model and (
        args.top_delta_recon_loss_weight > 0 or args.top_delta_delta_loss_weight > 0
    )
    use_cond_centroid_top_delta_loss = conditional_model and (
        args.centroid_top_delta_loss_weight > 0
        or args.centroid_top_delta_cosine_weight > 0
        or args.centroid_top_delta_contrast_weight > 0
    )
    train_dataset = SimpleDataset(
        data['train_X'],
        data['train_cell_type_ids'] if conditional_model or use_plain_vae_delta_loss else None,
        data['train_pert_ids'] if conditional_model or use_plain_vae_delta_loss else None,
        data['delta_lookup'] if args.model_type == "cond_delta_vae" or use_plain_vae_delta_loss or use_cond_top_delta_loss or use_cond_centroid_top_delta_loss else None,
        data['train_delta_ids'] if args.model_type == "cond_delta_vae" or use_plain_vae_delta_loss or use_cond_top_delta_loss or use_cond_centroid_top_delta_loss else None,
        data['control_baseline_lookup'] if use_plain_vae_delta_loss or use_cond_top_delta_loss or use_cond_centroid_top_delta_loss else None,
    )
    test_dataset = SimpleDataset(
        data['test_X'],
        data['test_cell_type_ids'] if conditional_model else None,
        data['test_pert_ids'] if conditional_model else None,
        data['delta_lookup'] if args.model_type == "cond_delta_vae" else None,
        data['test_delta_ids'] if args.model_type == "cond_delta_vae" else None,
    )

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.infer_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    if is_main_process(rank):
        print(f"\nDatasets: train={len(train_dataset):,}, test={len(test_dataset):,}, genes={n_genes}", flush=True)

    # Build model
    evo_config = {
        'n_gene': args.evo_n_gene,
        'n_gene_feat': args.evo_n_gene_feat,
        'n_pair_feat': args.evo_n_pair_feat,
        'n_embed': args.evo_n_embed,
        'num_evoformer_blocks': args.evo_num_blocks,
    }
    vae_hidden_dims = parse_int_list(args.vae_hidden_dims)
    model = build_model(
        args.model_type,
        n_genes,
        args.latent_dim,
        device,
        evo_config,
        vae_hidden_dims=vae_hidden_dims,
        vae_normalize_latent=args.vae_normalize_latent,
        n_cell_types=data['n_cell_types'],
        n_perts=data['n_perts'],
        cond_embed_dim=args.cond_embed_dim,
        condition_use_cell_type=args.condition_use_cell_type,
        condition_use_perturbation=args.condition_use_perturbation,
        vq_num_codes=args.vq_num_codes,
        vq_commitment_cost=args.vq_commitment_cost,
        delta_loss_weight=args.delta_loss_weight,
    )

    if is_main_process(rank):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {args.model_type}, params={n_params:,}", flush=True)

    if world_size > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=args.ddp_timeout_minutes),
            )
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Training
    if not args.skip_training:
        if is_main_process(rank):
            print(f"\n{'='*70}")
            print(f"Training: {args.epochs} epochs, lr={args.lr}")
            print(f"{'='*70}", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        best_r2 = -float('inf')
        history = {'train_loss': [], 'val_mse': [], 'val_r2': []}

        for epoch in range(args.epochs):
            if train_sampler:
                train_sampler.set_epoch(epoch)

            train_loss = train_epoch(model, train_loader, optimizer, device, args.model_type, epoch, rank, args)
            scheduler.step()

            if is_main_process(rank):
                history['train_loss'].append(train_loss)
                print(f"Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4e}", flush=True)

            if (epoch + 1) % args.validate_every == 0:
                val = validate(model, test_loader, device, args.model_type, rank)

                if is_main_process(rank):
                    history['val_mse'].append(val['mse'])
                    history['val_r2'].append(val['r2'])
                    print(f"  Val MSE={val['mse']:.4f}, R2={val['r2']:.4f}", flush=True)

                    if val['r2'] > best_r2:
                        best_r2 = val['r2']
                        torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                                   output_dir / "best_model.pt")
                        print(f"  ✓ Best model saved (R2={best_r2:.4f})", flush=True)

            if is_main_process(rank) and (epoch + 1) % args.save_every == 0:
                torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                           output_dir / f"checkpoint_{epoch+1}.pt")

        if is_main_process(rank):
            torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                       output_dir / "final_model.pt")
            with open(output_dir / "history.json", 'w') as f:
                json.dump(history, f, indent=2)

    _distributed_barrier()

    if args.eval_checkpoints_every > 0:
        for epoch in range(args.eval_checkpoints_every, args.epochs + 1, args.eval_checkpoints_every):
            ckpt_path = output_dir / f"checkpoint_{epoch}.pt"
            if not ckpt_path.exists():
                if is_main_process(rank):
                    print(f"Skipping missing checkpoint: {ckpt_path}", flush=True)
                continue
            eval_output_dir = output_dir / args.eval_output_subdir / f"epoch_{epoch:04d}"
            evaluate_checkpoint(
                model=model,
                ckpt_path=ckpt_path,
                eval_output_dir=eval_output_dir,
                test_loader=test_loader,
                data=data,
                device=device,
                model_type=args.model_type,
                rank=rank,
                pert_col=pert_col,
                control_label=control_label,
                celltype_col=celltype_col,
                args=args,
                epoch=epoch,
            )
        cleanup_distributed()
        return

    if args.skip_final_eval:
        cleanup_distributed()
        return

    # Inference
    ckpt_path = args.checkpoint_path or (output_dir / "best_model.pt")
    evaluate_checkpoint(
        model=model,
        ckpt_path=Path(ckpt_path),
        eval_output_dir=output_dir,
        test_loader=test_loader,
        data=data,
        device=device,
        model_type=args.model_type,
        rank=rank,
        pert_col=pert_col,
        control_label=control_label,
        celltype_col=celltype_col,
        args=args,
        epoch=None,
    )

    cleanup_distributed()


if __name__ == "__main__":
    main()
