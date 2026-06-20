from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import pandas as pd

from perturbnova.config import load_infer_config
from perturbnova.data import _extract_feature_matrix, _select_split_subset
from perturbnova.post_infer_eval import _infer_feature_names
from perturbnova.utils.checkpoint import export_json
from perturbnova.vae import build_vae_module, decode_array_with_vae, encode_with_vae


ROOT = Path("/work/home/cryoem666/xyf/temp/pycharm/PerturbNova")
OUTPUT_ROOT = ROOT / "outputs" / "experiments" / "replogle_stage2_ablation_500k_20260408"


def _stage1_dir(task: str, cell: str) -> Path:
    return OUTPUT_ROOT / task / cell / "stage1"


def _load_base_infer_config(task: str, cell: str) -> dict:
    snapshot = OUTPUT_ROOT / task / cell / "frozen_500k_infer" / "infer_config_snapshot.json"
    if snapshot.exists():
        with snapshot.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    path = ROOT / "configs" / f"replogle_{task}" / "inference" / cell / "infer.toml"
    return load_infer_config(path)


def _load_stage1_config(stage1_dir: Path) -> dict:
    path = stage1_dir / "config_snapshot.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_eval_subset(adata_: ad.AnnData, dataset_config: dict) -> ad.AnnData:
    return _select_split_subset(adata_, dataset_config, "test")


def _make_prediction_adata(real_subset: ad.AnnData, recon: np.ndarray, feature_names: list[str]) -> ad.AnnData:
    pred = ad.AnnData(X=np.asarray(recon, dtype=np.float32))
    pred.obs = real_subset.obs.copy()
    for column in pred.obs.columns:
        if pd.api.types.is_string_dtype(pred.obs[column].dtype) or pred.obs[column].dtype == object:
            pred.obs[column] = pred.obs[column].astype(str)
    pred.var_names = feature_names
    return pred


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value)!r}")


def _write_toml(path: Path, payload: dict) -> None:
    lines: list[str] = []

    def emit_table(prefix: list[str], data: dict) -> None:
        scalars: list[tuple[str, object]] = []
        nested: list[tuple[str, dict]] = []
        for key, value in data.items():
            if isinstance(value, dict):
                nested.append((key, value))
            else:
                scalars.append((key, value))
        if prefix:
            lines.append(f"[{'.'.join(prefix)}]")
        for key, value in scalars:
            lines.append(f"{key} = {_toml_value(value)}")
        if prefix or scalars:
            lines.append("")
        for key, value in nested:
            emit_table(prefix + [key], value)

    emit_table([], payload)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_infer_like_config(
    base_infer_config: dict,
    out_dir: Path,
    pred_path: Path,
    real_path: Path,
    reference_path: Path,
    control_label: str,
) -> Path:
    cfg = dict(base_infer_config)
    cfg["experiment"] = dict(cfg["experiment"])
    cfg["experiment"]["output_dir"] = str(out_dir)
    cfg["checkpoint"] = dict(cfg["checkpoint"])
    cfg["output"] = dict(cfg["output"])
    cfg["output"]["write_to"] = "X"
    cfg["output"]["prediction_path"] = str(pred_path)
    cfg["output"]["real_copy_path"] = str(real_path)
    cfg["input"] = dict(cfg["input"])
    cfg["input"]["data_path"] = str(reference_path)
    cfg["input"]["reference_data_path"] = str(reference_path)
    cfg["cell_eval"] = dict(cfg["cell_eval"])
    cfg["cell_eval"]["enabled"] = True
    cfg["cell_eval"]["outdir"] = str(out_dir / "cell_eval")
    cfg["cell_eval"]["control_pert"] = control_label
    out_path = out_dir / "vae_ceiling_infer_config.toml"
    _write_toml(out_path, cfg)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PerturbNova VAE reconstruction ceiling on Replogle splits.")
    parser.add_argument("--task", choices=["fewshot", "zeroshot"], required=True)
    parser.add_argument("--cell", choices=["hepg2", "jurkat", "k562", "rpe1"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=10.0)
    args = parser.parse_args()
    ad.settings.allow_write_nullable_strings = True

    stage1_dir = _stage1_dir(args.task, args.cell)
    train_cfg = _load_stage1_config(stage1_dir)
    dataset_cfg = train_cfg["dataset"]
    data_path = Path(dataset_cfg["data_path"])
    out_dir = stage1_dir / f"vae_ceiling_{args.task}_{args.cell}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    vae_cfg = dict(train_cfg["vae"])
    vae_cfg["checkpoint_path"] = str(stage1_dir / "vae_checkpoints" / "latest.pt")

    adata = ad.read_h5ad(data_path)
    real_subset = _build_eval_subset(adata, dataset_cfg)
    features = _extract_feature_matrix(real_subset, dataset_cfg).astype(np.float32, copy=False)

    vae = build_vae_module(vae_cfg, input_dim=features.shape[1], device=device)
    if vae is None:
        raise RuntimeError("VAE is disabled in the selected stage1 config.")
    vae.eval()

    all_latent = []
    all_recon = []
    with torch.no_grad():
        for start in range(0, features.shape[0], args.batch_size):
            end = min(start + args.batch_size, features.shape[0])
            batch = torch.from_numpy(features[start:end]).to(device)
            latent = encode_with_vae(vae, batch)
            recon = decode_array_with_vae(vae, latent.detach().cpu().numpy(), batch_size=args.batch_size, device=device)
            all_latent.append(latent.detach().cpu().numpy())
            all_recon.append(recon)

    latent_np = np.concatenate(all_latent, axis=0).astype(np.float32, copy=False)
    recon_np = np.concatenate(all_recon, axis=0).astype(np.float32, copy=False)
    recon_np = np.clip(recon_np, args.clamp_min, args.clamp_max)

    feature_names = _infer_feature_names(adata, dataset_cfg, features.shape[1])
    pred_adata = _make_prediction_adata(real_subset, recon_np, feature_names)
    pred_adata.obsm["X_vae_latent"] = latent_np

    pred_path = out_dir / "predictions.h5ad"
    real_path = out_dir / "cell_eval_real.h5ad"
    for column in real_subset.obs.columns:
        if pd.api.types.is_string_dtype(real_subset.obs[column].dtype) or real_subset.obs[column].dtype == object:
            real_subset.obs[column] = real_subset.obs[column].astype(str)
    pred_adata.write_h5ad(pred_path)
    real_subset.write_h5ad(real_path)

    infer_cfg = _load_base_infer_config(args.task, args.cell)
    infer_cfg_path = _write_infer_like_config(
        base_infer_config=infer_cfg,
        out_dir=out_dir,
        pred_path=pred_path,
        real_path=real_path,
        reference_path=data_path,
        control_label=str(dataset_cfg["control"]["label"]),
    )

    mse = float(np.mean((recon_np - features) ** 2))
    export_json(
        out_dir / "summary.json",
        {
            "task": args.task,
            "cell": args.cell,
            "n_obs": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "reconstruction_mse": mse,
            "prediction_path": str(pred_path),
            "real_path": str(real_path),
            "infer_config_json": str(infer_cfg_path),
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "reconstruction_mse": mse}, ensure_ascii=False))


if __name__ == "__main__":
    main()
