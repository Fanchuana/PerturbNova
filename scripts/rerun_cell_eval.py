#!/usr/bin/env python3
"""Rerun cell-eval from saved reconstructed.h5ad and real.h5ad files."""

import argparse
import json
import sys
from pathlib import Path

import anndata as ad

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_autoencoder_pipeline import run_cell_eval


def main():
    parser = argparse.ArgumentParser(description="Rerun full cell-eval from existing h5ad outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=str, default="full")
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--pert-col", type=str, default="gene")
    parser.add_argument("--control-label", type=str, default="non-targeting")
    parser.add_argument("--celltype-col", type=str, default="cell_line")
    parser.add_argument("--pred-name", type=str, default="reconstructed.h5ad")
    parser.add_argument("--real-name", type=str, default="real.h5ad")
    args = parser.parse_args()

    pred_path = args.output_dir / args.pred_name
    real_path = args.output_dir / args.real_name
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not real_path.exists():
        raise FileNotFoundError(real_path)

    print(f"Loading pred: {pred_path}", flush=True)
    pred_adata = ad.read_h5ad(pred_path)
    print(f"Loading real: {real_path}", flush=True)
    real_adata = ad.read_h5ad(real_path)
    print(f"Shapes: pred={pred_adata.shape}, real={real_adata.shape}", flush=True)
    print(
        f"Cell-eval: profile={args.profile}, threads={args.num_threads}, batch_size={args.batch_size}",
        flush=True,
    )

    ce_results = run_cell_eval(
        pred_adata=pred_adata,
        real_adata=real_adata,
        output_dir=args.output_dir,
        pert_col=args.pert_col,
        control_label=args.control_label,
        celltype_col=args.celltype_col,
        profile=args.profile,
        num_threads=args.num_threads,
        batch_size=args.batch_size,
    )

    out_path = args.output_dir / "cell_eval_rerun_results.json"
    with open(out_path, "w") as f:
        json.dump({"cell_eval": ce_results, "config": vars(args)}, f, indent=2, default=str)
    print(f"Done. Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
