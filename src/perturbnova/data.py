from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .config import load_dataset_config
from .utils.distributed import DistributedContext


def _to_numpy(matrix) -> np.ndarray:
    if isinstance(matrix, np.ndarray):
        return matrix.astype(np.float32, copy=False)
    if hasattr(matrix, "toarray"):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def _extract_feature_matrix(adata_: ad.AnnData, dataset_config: dict[str, Any]) -> np.ndarray:
    source = dataset_config["feature_space"]["source"]
    key = dataset_config["feature_space"]["key"]
    use_hvg = dataset_config["use_hvg"]
    hvg_key = dataset_config["hvg_key"]

    if source == "X":
        matrix = adata_.X
        if use_hvg:
            matrix = adata_[:, adata_.var[hvg_key].astype(bool)].X
    elif source == "layer":
        if not key:
            raise ValueError("`dataset.feature_space.key` is required when source=layer.")
        matrix = adata_.layers[key]
        if use_hvg:
            matrix = matrix[:, adata_.var[hvg_key].astype(bool)]
    elif source == "obsm":
        if not key:
            raise ValueError("`dataset.feature_space.key` is required when source=obsm.")
        matrix = adata_.obsm[key]
    else:
        raise ValueError(f"Unsupported feature source: {source}")
    return _to_numpy(matrix)


def _read_h5ad(path: str | Path) -> ad.AnnData:
    return ad.read_h5ad(path)


def _build_split_masks(
    adata_: ad.AnnData,
    dataset_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_config = dataset_config["split"]
    if split_config["mode"] == "none":
        train_mask = np.ones(adata_.n_obs, dtype=bool)
        empty_mask = np.zeros(adata_.n_obs, dtype=bool)
        return train_mask, empty_mask, empty_mask

    obs_keys = dataset_config["obs_keys"]
    target_cell_type = split_config["target_cell_type"]
    cell_type_names = adata_.obs[obs_keys["cell_type"]].astype(str).to_numpy()
    is_target = cell_type_names == target_cell_type
    perturbation_names = adata_.obs[obs_keys["perturbation"]].astype(str).to_numpy()
    control_label = str(dataset_config["control"]["label"])
    is_control = perturbation_names == control_label
    target_controls = is_target & is_control

    if split_config["mode"] == "fewshot_holdout":
        val_perturbations = set(split_config["val_perturbations"])
        test_perturbations = set(split_config["test_perturbations"])
        is_val_perturbed = is_target & ~is_control & np.isin(perturbation_names, list(val_perturbations))
        is_test_perturbed = is_target & ~is_control & np.isin(perturbation_names, list(test_perturbations))
        is_val = is_val_perturbed | target_controls
        is_test = is_test_perturbed | target_controls
        train_mask = ~(is_val_perturbed | is_test_perturbed)
        return train_mask, is_val, is_test

    if split_config["mode"] == "zeroshot_holdout":
        role = str(split_config.get("zeroshot_role", "test")).lower()
        if role not in {"train", "val", "test"}:
            raise ValueError("`dataset.split.zeroshot_role` must be `train`, `val`, or `test`.")
        # Zero-shot keeps target-cell-line controls in training and only holds out perturbed cells.
        is_target_perturbed = is_target & ~is_control
        is_val = (is_target_perturbed | target_controls) if role == "val" else np.zeros(adata_.n_obs, dtype=bool)
        is_test = (is_target_perturbed | target_controls) if role == "test" else np.zeros(adata_.n_obs, dtype=bool)
        train_mask = ~(is_val | is_test)
        train_mask = train_mask | target_controls
        return train_mask, is_val, is_test

    raise ValueError(f"Unsupported split mode: {split_config['mode']}")


def _select_split_subset(
    adata_: ad.AnnData,
    dataset_config: dict[str, Any],
    subset: str,
) -> ad.AnnData:
    if subset not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split subset: {subset}")

    split_mode = dataset_config["split"]["mode"]
    if split_mode == "none" and subset in {"val", "test"}:
        raise ValueError(
            f"Cannot select `{subset}` split for inference because `dataset.split.mode` is `none`."
        )

    train_mask, val_mask, test_mask = _build_split_masks(adata_, dataset_config)
    subset_mask = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }[subset]
    if int(np.count_nonzero(subset_mask)) == 0:
        raise ValueError(f"Inference split subset `{subset}` selected 0 observations.")
    return adata_[subset_mask].copy()


def _resolve_inference_dataset_config(
    infer_config: dict[str, Any],
    train_dataset_config: dict[str, Any],
) -> dict[str, Any]:
    split_config = infer_config["input"]["split"]
    if split_config["dataset_config_path"]:
        dataset_config = load_dataset_config(split_config["dataset_config_path"])
    else:
        dataset_config = deepcopy(train_dataset_config)

    input_data_path = infer_config["input"]["data_path"]
    if input_data_path:
        dataset_config["data_path"] = input_data_path
    return dataset_config


def _build_vocab(values) -> dict[str, int]:
    unique_values = sorted({str(value) for value in values})
    return {value: idx for idx, value in enumerate(unique_values)}


def _encode_values(
    values,
    vocab: dict[str, int],
    field_name: str,
    default_label: str = "",
    fallback_label: str = "",
) -> np.ndarray:
    encoded = []
    default_value = default_label or next(iter(vocab))
    for value in values:
        value = str(value)
        if value in vocab:
            encoded.append(vocab[value])
        elif fallback_label and fallback_label in vocab:
            encoded.append(vocab[fallback_label])
        elif default_value in vocab:
            encoded.append(vocab[default_value])
        else:
            raise KeyError(f"Unknown {field_name} label: {value}")
    return np.asarray(encoded, dtype=np.int64)


@dataclass
class StateDataArtifacts:
    feature_dim: int
    raw_feature_dim: int
    output_dim: int
    obs_keys: dict[str, str]
    feature_space: dict[str, Any]
    use_hvg: bool
    hvg_key: str
    condition_sizes: dict[str, int]
    vocab: dict[str, dict[str, int]]
    default_labels: dict[str, str]
    control_label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StateDataArtifacts":
        if "raw_feature_dim" not in payload:
            payload = dict(payload)
            payload["raw_feature_dim"] = payload["feature_dim"]
        return cls(**payload)


class StatePerturbationTrainingDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        perturbation_ids: np.ndarray,
        batch_ids: np.ndarray,
        cell_type_ids: np.ndarray,
        cell_type_names: np.ndarray,
        perturbation_names: np.ndarray,
        control_enabled: bool,
        control_label: str,
        samples_per_query: int,
    ) -> None:
        self.features = features
        self.perturbation_ids = perturbation_ids
        self.batch_ids = batch_ids
        self.cell_type_ids = cell_type_ids
        self.control_enabled = control_enabled
        self.samples_per_query = samples_per_query
        self.control_indices: dict[str, np.ndarray] = {}

        if control_enabled:
            for cell_type_name in np.unique(cell_type_names):
                mask = (cell_type_names == cell_type_name) & (perturbation_names == control_label)
                self.control_indices[str(cell_type_name)] = np.flatnonzero(mask)
        self.cell_type_names = cell_type_names.astype(str)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "features": torch.from_numpy(self.features[index]),
            "perturbation": torch.tensor(self.perturbation_ids[index], dtype=torch.long),
            "batch": torch.tensor(self.batch_ids[index], dtype=torch.long),
            "cell_type": torch.tensor(self.cell_type_ids[index], dtype=torch.long),
        }
        if self.control_enabled:
            pool = self.control_indices.get(self.cell_type_names[index], np.array([], dtype=np.int64))
            if pool.size == 0:
                chosen = np.repeat(index, self.samples_per_query)
            else:
                chosen = np.random.choice(pool, size=self.samples_per_query, replace=True)
            item["control_set"] = torch.from_numpy(self.features[chosen])
        return item


class StatePerturbationInferenceDataset(Dataset):
    def __init__(
        self,
        features_shape: tuple[int, int],
        perturbation_ids: np.ndarray,
        batch_ids: np.ndarray,
        cell_type_ids: np.ndarray,
        cell_type_names: np.ndarray,
        reference_features: np.ndarray,
        reference_cell_type_names: np.ndarray,
        reference_perturbation_names: np.ndarray,
        control_enabled: bool,
        control_label: str,
        samples_per_query: int,
    ) -> None:
        self.features_shape = features_shape
        self.perturbation_ids = perturbation_ids
        self.batch_ids = batch_ids
        self.cell_type_ids = cell_type_ids
        self.cell_type_names = cell_type_names.astype(str)
        self.reference_features = reference_features
        self.control_enabled = control_enabled
        self.samples_per_query = samples_per_query
        self.control_indices: dict[str, np.ndarray] = {}

        if control_enabled:
            for cell_type_name in np.unique(reference_cell_type_names):
                mask = (reference_cell_type_names == cell_type_name) & (
                    reference_perturbation_names == control_label
                )
                self.control_indices[str(cell_type_name)] = np.flatnonzero(mask)

    def __len__(self) -> int:
        return self.features_shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "index": torch.tensor(index, dtype=torch.long),
            "perturbation": torch.tensor(self.perturbation_ids[index], dtype=torch.long),
            "batch": torch.tensor(self.batch_ids[index], dtype=torch.long),
            "cell_type": torch.tensor(self.cell_type_ids[index], dtype=torch.long),
        }
        if self.control_enabled:
            pool = self.control_indices.get(self.cell_type_names[index], np.array([], dtype=np.int64))
            if pool.size == 0:
                chosen = np.zeros(self.samples_per_query, dtype=np.int64)
            else:
                chosen = np.random.choice(pool, size=self.samples_per_query, replace=True)
            item["control_set"] = torch.from_numpy(self.reference_features[chosen])
        return item


@dataclass
class StatePerturbationDataModule:
    train_loader: DataLoader
    train_sampler: DistributedSampler | None
    val_loader: DataLoader | None
    val_sampler: DistributedSampler | None
    artifacts: StateDataArtifacts
    split_summary: dict[str, dict[str, int]]


def _split_anndata(adata_: ad.AnnData, dataset_config: dict[str, Any]) -> tuple[ad.AnnData, ad.AnnData | None]:
    train_mask, val_mask, _ = _build_split_masks(adata_, dataset_config)
    train_adata = adata_[train_mask].copy()
    val_adata = adata_[val_mask].copy() if int(np.count_nonzero(val_mask)) > 0 else None
    return train_adata, val_adata


def _build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    distributed_context: DistributedContext,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if distributed_context.is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )
    return loader, sampler


def build_training_data_module(
    train_config: dict[str, Any],
    distributed_context: DistributedContext,
) -> StatePerturbationDataModule:
    dataset_config = train_config["dataset"]
    adata_ = _read_h5ad(dataset_config["data_path"])
    train_mask, val_mask, test_mask = _build_split_masks(adata_, dataset_config)
    train_adata, val_adata = _split_anndata(adata_, dataset_config)

    obs_keys = dataset_config["obs_keys"]
    full_obs = adata_.obs
    perturbation_names = full_obs[obs_keys["perturbation"]].astype(str).to_numpy()
    control_mask = perturbation_names == str(dataset_config["control"]["label"])
    perturbation_vocab = _build_vocab(full_obs[obs_keys["perturbation"]].astype(str).values)
    batch_vocab = _build_vocab(full_obs[obs_keys["batch"]].astype(str).values)
    cell_type_vocab = _build_vocab(full_obs[obs_keys["cell_type"]].astype(str).values)

    train_features = _extract_feature_matrix(train_adata, dataset_config)
    train_perturbation_names = train_adata.obs[obs_keys["perturbation"]].astype(str).values
    train_batch_names = train_adata.obs[obs_keys["batch"]].astype(str).values
    train_cell_type_names = train_adata.obs[obs_keys["cell_type"]].astype(str).values

    train_dataset = StatePerturbationTrainingDataset(
        features=train_features,
        perturbation_ids=_encode_values(
            train_perturbation_names,
            perturbation_vocab,
            field_name="perturbation",
        ),
        batch_ids=_encode_values(
            train_batch_names,
            batch_vocab,
            field_name="batch",
        ),
        cell_type_ids=_encode_values(
            train_cell_type_names,
            cell_type_vocab,
            field_name="cell_type",
        ),
        cell_type_names=train_cell_type_names,
        perturbation_names=train_perturbation_names,
        control_enabled=dataset_config["control"]["enabled"],
        control_label=dataset_config["control"]["label"],
        samples_per_query=dataset_config["control"]["samples_per_query"],
    )

    val_loader = None
    val_sampler = None
    if val_adata is not None:
        val_features = _extract_feature_matrix(val_adata, dataset_config)
        val_perturbation_names = val_adata.obs[obs_keys["perturbation"]].astype(str).values
        val_batch_names = val_adata.obs[obs_keys["batch"]].astype(str).values
        val_cell_type_names = val_adata.obs[obs_keys["cell_type"]].astype(str).values
        val_dataset = StatePerturbationTrainingDataset(
            features=val_features,
            perturbation_ids=_encode_values(val_perturbation_names, perturbation_vocab, "perturbation"),
            batch_ids=_encode_values(val_batch_names, batch_vocab, "batch"),
            cell_type_ids=_encode_values(val_cell_type_names, cell_type_vocab, "cell_type"),
            cell_type_names=val_cell_type_names,
            perturbation_names=val_perturbation_names,
            control_enabled=dataset_config["control"]["enabled"],
            control_label=dataset_config["control"]["label"],
            samples_per_query=dataset_config["control"]["samples_per_query"],
        )
        val_loader, val_sampler = _build_dataloader(
            dataset=val_dataset,
            batch_size=train_config["optimization"]["batch_size"],
            num_workers=train_config["optimization"]["num_workers"],
            pin_memory=train_config["optimization"]["pin_memory"],
            persistent_workers=train_config["optimization"]["persistent_workers"],
            distributed_context=distributed_context,
            shuffle=False,
        )

    train_loader, train_sampler = _build_dataloader(
        dataset=train_dataset,
        batch_size=train_config["optimization"]["batch_size"],
        num_workers=train_config["optimization"]["num_workers"],
        pin_memory=train_config["optimization"]["pin_memory"],
        persistent_workers=train_config["optimization"]["persistent_workers"],
        distributed_context=distributed_context,
        shuffle=True,
    )

    artifacts = StateDataArtifacts(
        feature_dim=int(train_config["vae"]["latent_dim"] if train_config["vae"]["enabled"] else train_features.shape[1]),
        raw_feature_dim=int(train_features.shape[1]),
        output_dim=int(train_config["vae"]["latent_dim"] if train_config["vae"]["enabled"] else train_features.shape[1]),
        obs_keys=obs_keys,
        feature_space=dataset_config["feature_space"],
        use_hvg=dataset_config["use_hvg"],
        hvg_key=dataset_config["hvg_key"],
        condition_sizes={
            "perturbation": len(perturbation_vocab),
            "batch": len(batch_vocab),
            "cell_type": len(cell_type_vocab),
        },
        vocab={
            "perturbation": perturbation_vocab,
            "batch": batch_vocab,
            "cell_type": cell_type_vocab,
        },
        default_labels={
            "perturbation": next(iter(perturbation_vocab)),
            "batch": next(iter(batch_vocab)),
            "cell_type": next(iter(cell_type_vocab)),
        },
        control_label=dataset_config["control"]["label"],
    )
    split_summary = {}
    for split_name, split_mask in {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }.items():
        total = int(np.count_nonzero(split_mask))
        control = int(np.count_nonzero(split_mask & control_mask))
        split_summary[split_name] = {
            "samples": total,
            "control": control,
            "perturbed": total - control,
        }
    return StatePerturbationDataModule(
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        val_sampler=val_sampler,
        artifacts=artifacts,
        split_summary=split_summary,
    )


def build_inference_loader(
    infer_config: dict[str, Any],
    train_dataset_config: dict[str, Any],
    artifacts: StateDataArtifacts,
    distributed_context: DistributedContext,
) -> tuple[ad.AnnData, DataLoader, DistributedSampler | None]:
    input_config = infer_config["input"]
    inference_dataset_config = _resolve_inference_dataset_config(infer_config, train_dataset_config)
    data_path = input_config["data_path"] or inference_dataset_config["data_path"]
    if not data_path:
        raise ValueError("Inference requires `input.data_path` or `input.split.dataset_config_path` with `dataset.data_path`.")

    split_subset = input_config["split"]["subset"]
    split_obs_keys = inference_dataset_config.get("obs_keys", {})
    obs_keys = {
        field: input_config["obs_keys"][field] or split_obs_keys.get(field, "") or artifacts.obs_keys[field]
        for field in ("perturbation", "batch", "cell_type")
    }
    defaults = input_config["defaults"]
    full_adata = _read_h5ad(data_path)
    target_adata = (
        _select_split_subset(full_adata, inference_dataset_config, split_subset)
        if split_subset
        else full_adata
    )
    if input_config["reference_data_path"]:
        reference_adata = _read_h5ad(input_config["reference_data_path"])
    elif split_subset:
        reference_adata = full_adata
    else:
        reference_adata = target_adata

    target_perturbation_names = (
        target_adata.obs[obs_keys["perturbation"]].astype(str).values
        if obs_keys["perturbation"] in target_adata.obs
        else np.asarray([defaults["perturbation"] or artifacts.control_label] * target_adata.n_obs)
    )
    target_batch_names = (
        target_adata.obs[obs_keys["batch"]].astype(str).values
        if obs_keys["batch"] in target_adata.obs
        else np.asarray([defaults["batch"] or artifacts.default_labels["batch"]] * target_adata.n_obs)
    )
    target_cell_type_names = (
        target_adata.obs[obs_keys["cell_type"]].astype(str).values
        if obs_keys["cell_type"] in target_adata.obs
        else np.asarray([defaults["cell_type"] or artifacts.default_labels["cell_type"]] * target_adata.n_obs)
    )

    reference_perturbation_names = (
        reference_adata.obs[obs_keys["perturbation"]].astype(str).values
    )
    reference_cell_type_names = reference_adata.obs[obs_keys["cell_type"]].astype(str).values
    reference_features = _extract_feature_matrix(reference_adata, train_dataset_config)

    inference_dataset = StatePerturbationInferenceDataset(
        features_shape=(target_adata.n_obs, artifacts.feature_dim),
        perturbation_ids=_encode_values(
            target_perturbation_names,
            artifacts.vocab["perturbation"],
            field_name="perturbation",
            default_label=defaults["perturbation"],
            fallback_label=infer_config["control"]["label"] or artifacts.control_label,
        ),
        batch_ids=_encode_values(
            target_batch_names,
            artifacts.vocab["batch"],
            field_name="batch",
            default_label=defaults["batch"] or artifacts.default_labels["batch"],
        ),
        cell_type_ids=_encode_values(
            target_cell_type_names,
            artifacts.vocab["cell_type"],
            field_name="cell_type",
            default_label=defaults["cell_type"] or artifacts.default_labels["cell_type"],
        ),
        cell_type_names=target_cell_type_names,
        reference_features=reference_features,
        reference_cell_type_names=reference_cell_type_names.astype(str),
        reference_perturbation_names=reference_perturbation_names.astype(str),
        control_enabled=infer_config["control"]["enabled"],
        control_label=infer_config["control"]["label"],
        samples_per_query=infer_config["control"]["samples_per_query"],
    )
    loader, sampler = _build_dataloader(
        dataset=inference_dataset,
        batch_size=infer_config["sampling"]["batch_size"],
        num_workers=infer_config["sampling"]["num_workers"],
        pin_memory=infer_config["sampling"]["pin_memory"],
        persistent_workers=infer_config["sampling"]["persistent_workers"],
        distributed_context=distributed_context,
        shuffle=False,
    )
    return target_adata, loader, sampler
