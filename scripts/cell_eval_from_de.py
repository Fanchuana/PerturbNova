#!/usr/bin/env python3
"""Compute cell-eval metrics from existing h5ad and DE CSV outputs."""

import argparse
from pathlib import Path

import anndata as ad
from scipy import sparse


def _parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def main():
    parser = argparse.ArgumentParser(description="Reuse existing cell-eval DE CSVs to compute metrics.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-eval-dir", type=Path, default=None)
    parser.add_argument("--pred-name", type=str, default="reconstructed.h5ad")
    parser.add_argument("--real-name", type=str, default="real.h5ad")
    parser.add_argument("--prefix", type=str, default="hepg2")
    parser.add_argument("--profile", type=str, default="full")
    parser.add_argument("--pert-col", type=str, default="gene")
    parser.add_argument("--control-label", type=str, default="non-targeting")
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--skip-metrics", type=str, default="pearson_edistance")
    args = parser.parse_args()

    from cell_eval import MetricsEvaluator

    output_dir = args.output_dir
    cell_eval_dir = args.cell_eval_dir or (output_dir / "cell_eval")
    pred_path = output_dir / args.pred_name
    real_path = output_dir / args.real_name
    de_real_path = cell_eval_dir / f"{args.prefix}_real_de.csv"
    de_pred_path = cell_eval_dir / f"{args.prefix}_pred_de.csv"

    for path in [pred_path, real_path, de_real_path, de_pred_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"Loading pred: {pred_path}", flush=True)
    pred_adata = ad.read_h5ad(pred_path)
    print(f"Loading real: {real_path}", flush=True)
    real_adata = ad.read_h5ad(real_path)
    if not sparse.issparse(pred_adata.X):
        pred_adata.X = sparse.csr_matrix(pred_adata.X)
    if not sparse.issparse(real_adata.X):
        real_adata.X = sparse.csr_matrix(real_adata.X)
    print(f"Using DE real: {de_real_path}", flush=True)
    print(f"Using DE pred: {de_pred_path}", flush=True)

    evaluator = MetricsEvaluator(
        adata_pred=pred_adata,
        adata_real=real_adata,
        de_pred=str(de_pred_path),
        de_real=str(de_real_path),
        control_pert=args.control_label,
        pert_col=args.pert_col,
        de_method="wilcoxon",
        num_threads=args.num_threads,
        batch_size=args.batch_size,
        outdir=str(cell_eval_dir),
        prefix=args.prefix,
        skip_de=False,
    )
    evaluator.compute(
        profile=args.profile,
        skip_metrics=_parse_csv_list(args.skip_metrics),
        basename="results.csv",
    )
    print(f"Done. Results written under {cell_eval_dir}", flush=True)


if __name__ == "__main__":
    main()
