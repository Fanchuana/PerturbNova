#!/usr/bin/env python3
"""Single-process checkpoint sweep for autoencoder experiments."""

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_autoencoder_pipeline import (
    SimpleDataset,
    build_model,
    compute_metrics,
    get_split_config_path,
    inference,
    load_data,
    parse_int_list,
)


def _parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _ensure_sparse_x(adata: ad.AnnData) -> ad.AnnData:
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    return adata


def _run_cell_eval_from_arrays(
    pred_x: np.ndarray,
    real_x: np.ndarray,
    obs: pd.DataFrame,
    var_names: list[str],
    output_dir: Path,
    prefix: str,
    pert_col: str,
    control_label: str,
    profile: str,
    num_threads: int,
    batch_size: int,
    skip_metrics: list[str] | None,
    real_de_path: Path | None,
) -> Path:
    from cell_eval import MetricsEvaluator

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_adata = ad.AnnData(X=sparse.csr_matrix(pred_x))
    pred_adata.obs = obs.copy()
    pred_adata.var_names = var_names
    real_adata = ad.AnnData(X=sparse.csr_matrix(real_x))
    real_adata.obs = obs.copy()
    real_adata.var_names = var_names

    evaluator = MetricsEvaluator(
        adata_pred=_ensure_sparse_x(pred_adata),
        adata_real=_ensure_sparse_x(real_adata),
        de_pred=None,
        de_real=str(real_de_path) if real_de_path and real_de_path.exists() else None,
        control_pert=control_label,
        pert_col=pert_col,
        de_method="wilcoxon",
        num_threads=num_threads,
        batch_size=batch_size,
        outdir=str(output_dir),
        prefix=prefix,
        skip_de=False,
    )
    evaluator.compute(profile=profile, skip_metrics=skip_metrics, basename="results.csv")
    return output_dir / f"{prefix}_real_de.csv"


def main():
    parser = argparse.ArgumentParser(description="Evaluate every Nth checkpoint in a single process.")
    parser.add_argument(
        "--model-type",
        choices=["vae", "evoformer"],
        required=True,
    )
    parser.add_argument("--split", default="zeroshot_hepg2")
    parser.add_argument("--split-config", default=None)
    parser.add_argument("--data-path", default="/work/home/cryoem666/czx/dataset/STATE/arcinstitute-State-Replogle-Filtered-Dec-6-2025/replogle_concat.h5ad")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-output-subdir", default="checkpoint_eval")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--infer-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--vae-hidden-dims", default="1024,1024,1024")
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
    parser.add_argument("--evo-n-gene", type=int, default=10)
    parser.add_argument("--evo-n-gene-feat", type=int, default=32)
    parser.add_argument("--evo-n-pair-feat", type=int, default=16)
    parser.add_argument("--evo-n-embed", type=int, default=1280)
    parser.add_argument("--evo-num-blocks", type=int, default=6)
    parser.add_argument("--cell-eval-profile", default="full")
    parser.add_argument("--cell-eval-skip-metrics", default="pearson_edistance")
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--cell-eval-batch-size", type=int, default=500)
    parser.add_argument("--pert-col", default=None)
    parser.add_argument("--control-label", default=None)
    parser.add_argument("--prefix", default="hepg2")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split_config = args.split_config or get_split_config_path(args.split)
    data = load_data(args.data_path, split_config, rank=0)
    conditional_model = args.model_type in {"cond_ae", "cond_vae", "vqvae", "cond_delta_vae"}
    test_loader = DataLoader(
        SimpleDataset(
            data["test_X"],
            data["test_cell_type_ids"] if conditional_model else None,
            data["test_pert_ids"] if conditional_model else None,
            data["delta_lookup"] if args.model_type == "cond_delta_vae" else None,
            data["test_delta_ids"] if args.model_type == "cond_delta_vae" else None,
        ),
        batch_size=args.infer_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    evo_config = {
        "n_gene": args.evo_n_gene,
        "n_gene_feat": args.evo_n_gene_feat,
        "n_pair_feat": args.evo_n_pair_feat,
        "n_embed": args.evo_n_embed,
        "num_evoformer_blocks": args.evo_num_blocks,
    }
    model = build_model(
        args.model_type,
        data["n_genes"],
        args.latent_dim,
        device,
        evo_config,
        vae_hidden_dims=parse_int_list(args.vae_hidden_dims),
        vae_normalize_latent=args.vae_normalize_latent,
        n_cell_types=data["n_cell_types"],
        n_perts=data["n_perts"],
        cond_embed_dim=args.cond_embed_dim,
        condition_use_cell_type=args.condition_use_cell_type,
        condition_use_perturbation=args.condition_use_perturbation,
        vq_num_codes=args.vq_num_codes,
        vq_commitment_cost=args.vq_commitment_cost,
        delta_loss_weight=args.delta_loss_weight,
    )

    pert_col = args.pert_col or data["pert_col"]
    control_label = args.control_label or data["control_label"]
    skip_metrics = _parse_csv_list(args.cell_eval_skip_metrics)
    rows = []
    real_de_path: Path | None = None

    for epoch in range(args.eval_every, args.epochs + 1, args.eval_every):
        ckpt_path = args.output_dir / f"checkpoint_{epoch}.pt"
        if not ckpt_path.exists():
            print(f"Skipping missing checkpoint: {ckpt_path}", flush=True)
            continue

        eval_dir = args.output_dir / args.eval_output_subdir / f"epoch_{epoch:04d}"
        cell_eval_dir = eval_dir / "cell_eval"
        print(f"\n=== Evaluating {args.model_type} epoch {epoch} ===", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        pred_x = inference(model, test_loader, device, args.model_type, rank=0)
        metrics = compute_metrics(data["test_X"], pred_x)
        print(f"Reconstruction: MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}", flush=True)

        real_de_path = _run_cell_eval_from_arrays(
            pred_x=pred_x,
            real_x=data["test_X"],
            obs=data["test_obs"],
            var_names=data["var_names"],
            output_dir=cell_eval_dir,
            prefix=args.prefix,
            pert_col=pert_col,
            control_label=control_label,
            profile=args.cell_eval_profile,
            num_threads=args.num_threads,
            batch_size=args.cell_eval_batch_size,
            skip_metrics=skip_metrics,
            real_de_path=real_de_path,
        )

        result = {
            "epoch": epoch,
            "checkpoint_path": str(ckpt_path),
            "metrics": metrics,
            "cell_eval": {"output_dir": str(cell_eval_dir)},
            "config": vars(args),
        }
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        rows.append({"epoch": epoch, **metrics})

        del pred_x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = args.output_dir / args.eval_output_subdir / "reconstruction_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Done. Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
