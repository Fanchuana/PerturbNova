#!/usr/bin/env python3
"""Post-hoc delta calibration for reconstructed h5ad predictions.

This rescales perturbation predictions away from the control baseline while
leaving control cells untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ad.settings.allow_write_nullable_strings = True


def _as_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _make_h5ad_compatible(adata: ad.AnnData) -> None:
    """Avoid nullable string dtypes that older anndata readers cannot write."""
    for frame in [adata.obs, adata.var]:
        frame.index = frame.index.astype(str)
        for col in frame.columns:
            if pd.api.types.is_string_dtype(frame[col].dtype):
                frame[col] = frame[col].astype(object)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--control-label", type=str, default="non-targeting")
    parser.add_argument("--pert-col", type=str, default="gene")
    parser.add_argument("--celltype-col", type=str, default="cell_line")
    parser.add_argument("--pred-name", type=str, default="reconstructed.h5ad")
    parser.add_argument("--real-name", type=str, default="real.h5ad")
    args = parser.parse_args()

    pred_path = args.input_dir / args.pred_name
    real_path = args.input_dir / args.real_name
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not real_path.exists():
        raise FileNotFoundError(real_path)

    pred = ad.read_h5ad(pred_path)
    real = ad.read_h5ad(real_path)
    if pred.shape != real.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, real={real.shape}")

    obs = pred.obs
    if args.pert_col not in obs.columns:
        raise KeyError(f"missing pert column {args.pert_col!r}")
    if args.celltype_col not in obs.columns:
        raise KeyError(f"missing celltype column {args.celltype_col!r}")

    X = _as_dense(pred.X).astype(np.float32, copy=True)
    pert = obs[args.pert_col].astype(str).to_numpy()
    celltypes = obs[args.celltype_col].astype(str).to_numpy()
    control_mask = pert == args.control_label

    control_baseline = {}
    global_control = X[control_mask].mean(axis=0)
    for cell in np.unique(celltypes):
        cell_mask = (celltypes == cell) & control_mask
        if np.any(cell_mask):
            control_baseline[cell] = X[cell_mask].mean(axis=0).astype(np.float32, copy=False)
        else:
            control_baseline[cell] = global_control.astype(np.float32, copy=False)

    out_X = X.copy()
    for cell in np.unique(celltypes):
        cell_mask = celltypes == cell
        pert_mask = cell_mask & ~control_mask
        if not np.any(pert_mask):
            continue
        base = control_baseline[cell]
        out_X[pert_mask] = base + args.gamma * (X[pert_mask] - base)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_pred = pred.copy()
    out_pred.X = out_X
    _make_h5ad_compatible(out_pred)
    _make_h5ad_compatible(real)
    out_pred.write_h5ad(args.output_dir / args.pred_name)
    real.write_h5ad(args.output_dir / args.real_name)

    meta = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "gamma": args.gamma,
        "control_label": args.control_label,
        "pert_col": args.pert_col,
        "celltype_col": args.celltype_col,
        "control_cells": int(control_mask.sum()),
        "perturbed_cells": int((~control_mask).sum()),
    }
    with open(args.output_dir / "delta_calibration.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
