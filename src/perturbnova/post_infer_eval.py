from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from anndata import concat as ad_concat

from .config import load_infer_config
from .data import _extract_feature_matrix
from .utils import ExperimentLogger, export_json
from .utils.checkpoint import load_checkpoint_payload


def _resolve_prediction_path(config: dict[str, Any]) -> Path:
    output_dir = Path(config["experiment"]["output_dir"])
    return Path(config["output"]["prediction_path"] or (output_dir / "predictions.h5ad"))


def _resolve_real_path(config: dict[str, Any]) -> Path:
    output_dir = Path(config["experiment"]["output_dir"])
    real_copy_path = config["output"]["real_copy_path"]
    if real_copy_path:
        return Path(real_copy_path)
    return output_dir / "cell_eval_real.h5ad"


def _resolve_obs_key(config: dict[str, Any], train_config: dict[str, Any], field: str) -> str:
    infer_key = config["input"]["obs_keys"].get(field, "")
    if infer_key:
        return infer_key
    return str(train_config["dataset"]["obs_keys"][field])


def _resolve_control_label(config: dict[str, Any], train_config: dict[str, Any], payload: dict[str, Any]) -> str:
    if config["cell_eval"]["control_pert"]:
        return str(config["cell_eval"]["control_pert"])
    if config["control"]["label"]:
        return str(config["control"]["label"])
    artifacts = payload.get("dataset_artifacts", {})
    if artifacts.get("control_label"):
        return str(artifacts["control_label"])
    return str(train_config["dataset"]["control"]["label"])


def _extract_prediction_feature_matrix(pred_adata: ad.AnnData, config: dict[str, Any]) -> np.ndarray:
    output_config = config["output"]
    if output_config["write_to"] == "X":
        matrix = pred_adata.X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)
    if output_config["write_to"] != "obsm":
        raise ValueError("Unsupported `output.write_to` value for cell_eval preparation.")
    key = output_config["key"]
    if key not in pred_adata.obsm:
        raise ValueError(f"Prediction AnnData is missing output.obsm key: {key}")
    return np.asarray(pred_adata.obsm[key], dtype=np.float32)


def _resolve_reference_path(config: dict[str, Any], train_config: dict[str, Any]) -> Path:
    input_config = config["input"]
    reference_data_path = input_config["reference_data_path"]
    if reference_data_path:
        return Path(reference_data_path)
    if input_config["data_path"]:
        return Path(input_config["data_path"])
    return Path(train_config["dataset"]["data_path"])


def _append_control_cells_if_missing(
    pred_adata: ad.AnnData,
    real_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    pert_col: str,
    control_label: str,
    celltype_col: str,
) -> tuple[ad.AnnData, ad.AnnData]:
    pred = pred_adata.copy()
    real = real_adata.copy()

    pred_has_control = control_label in set(pred.obs[pert_col].astype(str))
    real_has_control = control_label in set(real.obs[pert_col].astype(str))
    if pred_has_control and real_has_control:
        return pred, real

    reference_mask = reference_adata.obs[pert_col].astype(str) == control_label
    if celltype_col and celltype_col in real.obs and celltype_col in reference_adata.obs:
        target_cell_types = set(real.obs[celltype_col].astype(str))
        reference_mask = reference_mask & reference_adata.obs[celltype_col].astype(str).isin(target_cell_types)

    control_adata = reference_adata[reference_mask].copy()
    if control_adata.n_obs == 0:
        raise ValueError(
            "cell_eval requires control cells, but none were found in the configured reference dataset."
        )

    if not pred_has_control:
        pred = ad_concat([pred, control_adata.copy()], join="inner", merge="same")
    if not real_has_control:
        real = ad_concat([real, control_adata.copy()], join="inner", merge="same")
    return pred, real


def _build_metric_kwargs(embed_key: str, num_threads: int) -> dict[str, dict[str, Any]]:
    metric_kwargs: dict[str, dict[str, Any]] = {
        "pearson_edistance": {"n_jobs": num_threads},
    }
    if embed_key:
        metric_kwargs["discrimination_score_l2"] = {"embed_key": embed_key}
        metric_kwargs["discrimination_score_cosine"] = {"embed_key": embed_key}
    return metric_kwargs


def _infer_feature_names(
    adata_: ad.AnnData,
    dataset_config: dict[str, Any],
    feature_dim: int,
) -> list[str]:
    feature_space = dataset_config["feature_space"]
    source = feature_space["source"]
    key = feature_space["key"]

    if source in {"X", "layer"}:
        names = adata_.var_names.astype(str).tolist()
        if dataset_config["use_hvg"]:
            mask = adata_.var[dataset_config["hvg_key"]].astype(bool).to_numpy()
            names = [name for name, keep in zip(names, mask, strict=False) if keep]
        if len(names) == feature_dim:
            return names

    if source == "obsm" and dataset_config["hvg_key"] in adata_.var:
        mask = adata_.var[dataset_config["hvg_key"]].astype(bool).to_numpy()
        names = adata_.var_names[mask].astype(str).tolist()
        if len(names) == feature_dim:
            return names

    prefix = key or source or "feature"
    return [f"{prefix}_{index:05d}" for index in range(feature_dim)]


def _build_eval_anndata(
    template_adata: ad.AnnData,
    feature_matrix: np.ndarray,
    feature_names: list[str],
) -> ad.AnnData:
    eval_adata = ad.AnnData(X=np.asarray(feature_matrix, dtype=np.float32))
    eval_adata.obs = template_adata.obs.copy()
    eval_adata.var = pd.DataFrame(index=pd.Index(feature_names, dtype=str))
    return eval_adata


def run_cell_eval_from_config(
    config_path: str | Path,
    output_dir_override: str = "",
) -> tuple[Path, Path]:
    config = load_infer_config(config_path)
    if output_dir_override:
        config["experiment"]["output_dir"] = output_dir_override
    if not config["cell_eval"]["enabled"]:
        raise ValueError("cell_eval is disabled in this inference config.")

    output_dir = Path(config["experiment"]["output_dir"])
    cell_eval_outdir = Path(config["cell_eval"]["outdir"] or (output_dir / "cell_eval"))
    cell_eval_outdir.mkdir(parents=True, exist_ok=True)
    ad.settings.allow_write_nullable_strings = True
    logger = ExperimentLogger(
        output_dir,
        enabled=True,
        render=config["experiment"]["log_render"],
        progress_width=int(config["experiment"]["log_progress_width"]),
        run_name=config["experiment"]["name"],
    )

    payload = load_checkpoint_payload(config["checkpoint"]["path"], map_location="cpu")
    train_config = payload["config"]
    pert_col = config["cell_eval"]["pert_col"] or _resolve_obs_key(config, train_config, "perturbation")
    celltype_col = config["cell_eval"]["celltype_col"] or _resolve_obs_key(config, train_config, "cell_type")
    control_label = _resolve_control_label(config, train_config, payload)

    prediction_path = _resolve_prediction_path(config)
    real_path = _resolve_real_path(config)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found for cell_eval: {prediction_path}")
    if not real_path.exists():
        raise FileNotFoundError(
            f"Real AnnData copy not found for cell_eval: {real_path}. "
            "Enable `output.real_copy_path` or `cell_eval.enabled` before running inference."
        )

    raw_pred_adata = ad.read_h5ad(prediction_path)
    raw_real_adata = ad.read_h5ad(real_path)
    raw_reference_adata = ad.read_h5ad(_resolve_reference_path(config, train_config))

    dataset_config = train_config["dataset"]
    pred_features = _extract_prediction_feature_matrix(raw_pred_adata, config)
    real_features = _extract_feature_matrix(raw_real_adata, dataset_config)
    reference_features = _extract_feature_matrix(raw_reference_adata, dataset_config)
    if pred_features.shape[1] != real_features.shape[1]:
        raise ValueError(
            "Predicted and real feature spaces do not match for cell_eval preparation: "
            f"{pred_features.shape[1]} != {real_features.shape[1]}"
        )

    feature_names = _infer_feature_names(raw_reference_adata, dataset_config, int(real_features.shape[1]))
    pred_adata = _build_eval_anndata(raw_pred_adata, pred_features, feature_names)
    real_adata = _build_eval_anndata(raw_real_adata, real_features, feature_names)
    reference_adata = _build_eval_anndata(raw_reference_adata, reference_features, feature_names)

    pred_adata, real_adata = _append_control_cells_if_missing(
        pred_adata=pred_adata,
        real_adata=real_adata,
        reference_adata=reference_adata,
        pert_col=pert_col,
        control_label=control_label,
        celltype_col=celltype_col,
    )

    prepared_pred_path = cell_eval_outdir / "pred_for_cell_eval.h5ad"
    prepared_real_path = cell_eval_outdir / "real_for_cell_eval.h5ad"
    pred_adata.write_h5ad(prepared_pred_path)
    real_adata.write_h5ad(prepared_real_path)

    export_json(
        cell_eval_outdir / "cell_eval_context.json",
        {
            "prediction_path": str(prediction_path),
            "real_path": str(real_path),
            "prepared_prediction_path": str(prepared_pred_path),
            "prepared_real_path": str(prepared_real_path),
            "pert_col": pert_col,
            "celltype_col": celltype_col,
            "control_pert": control_label,
            "profile": config["cell_eval"]["profile"],
            "feature_dim": int(real_features.shape[1]),
        },
    )
    logger.info(f"prepared_eval_dir={cell_eval_outdir}", tag="CEVAL")

    try:
        from cell_eval import MetricsEvaluator
        from cell_eval.utils import split_anndata_on_celltype
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cell_eval is not installed in the active Python environment; run inference and cell_eval in the same env."
        ) from exc

    metric_kwargs = _build_metric_kwargs(
        embed_key=str(config["cell_eval"]["embed_key"]),
        num_threads=int(config["cell_eval"]["num_threads"]),
    )
    skip_metrics = (
        [metric.strip() for metric in str(config["cell_eval"]["skip_metrics"]).split(",") if metric.strip()]
        or None
    )
    profile = str(config["cell_eval"]["profile"])
    common_kwargs = {
        "de_pred": None,
        "de_real": None,
        "control_pert": control_label,
        "pert_col": pert_col,
        "de_method": str(config["cell_eval"]["de_method"]),
        "num_threads": int(config["cell_eval"]["num_threads"]),
        "batch_size": int(config["cell_eval"]["batch_size"]),
        "outdir": str(cell_eval_outdir),
        "allow_discrete": bool(config["cell_eval"]["allow_discrete"]),
        "skip_de": profile in {"anndata", "pds"},
    }

    if celltype_col and celltype_col in real_adata.obs and celltype_col in pred_adata.obs:
        real_split = split_anndata_on_celltype(real_adata, celltype_col)
        pred_split = split_anndata_on_celltype(pred_adata, celltype_col)
        if set(real_split) != set(pred_split):
            raise ValueError("Real and predicted AnnData have different cell types for cell_eval.")
        for cell_type in sorted(real_split):
            evaluator = MetricsEvaluator(
                adata_pred=pred_split[cell_type],
                adata_real=real_split[cell_type],
                prefix=cell_type,
                **common_kwargs,
            )
            evaluator.compute(
                profile=profile,
                metric_configs=metric_kwargs,
                skip_metrics=skip_metrics,
                basename="results.csv",
            )
    else:
        evaluator = MetricsEvaluator(
            adata_pred=pred_adata,
            adata_real=real_adata,
            prefix=None,
            **common_kwargs,
        )
        evaluator.compute(
            profile=profile,
            metric_configs=metric_kwargs,
            skip_metrics=skip_metrics,
            basename="results.csv",
        )

    logger.info(f"saved_results={cell_eval_outdir}", tag="CEVAL")

    return prepared_pred_path, prepared_real_path
