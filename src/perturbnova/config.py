from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "perturbnova_run",
        "output_dir": "./outputs/perturbnova_run",
        "seed": 42,
        "log_render": "compact",
        "log_progress_width": 24,
    },
    "distributed": {
        "backend": "",
        "find_unused_parameters": False,
    },
    "training": {
        "stage": "",
        "unfreeze_vae_in_stage2": False,
        "mode": "diffusion_only",
    },
    "dataset": {
        "name": "state_perturbation",
        "data_path": "",
        "use_hvg": True,
        "hvg_key": "highly_variable",
        "feature_space": {
            "source": "X",
            "key": "",
        },
        "obs_keys": {
            "perturbation": "gene",
            "batch": "gem_group",
            "cell_type": "cell_line",
        },
        "split": {
            "mode": "none",
            "target_cell_type": "",
            "val_perturbations": [],
            "test_perturbations": [],
        },
        "control": {
            "enabled": False,
            "label": "non-targeting",
            "samples_per_query": 32,
        },
    },
    "model": {
        "name": "film_mlp",
        "hidden_dim": 2048,
        "num_layers": 3,
        "time_embed_dim": 2048,
        "dropout": 0.0,
        "use_encoder": True,
        "use_batch_embeddings": False,
        "control_attention_heads": 4,
        "perturbation_dropout": 0.15,
    },
    "diffusion": {
        "steps": 1000,
        "learn_sigma": False,
        "noise_schedule": "linear",
        "timestep_respacing": "",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
    },
    "optimization": {
        "batch_size": 128,
        "microbatch_size": -1,
        "max_steps": 100_000,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "lr_anneal_steps": 100_000,
        "ema_rates": [0.9999],
        "schedule_sampler": "uniform",
        "amp": False,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "gradient_clip_norm": 1.0,
        "loss_logging_mode": "step",
        "log_every_steps": 100,
        "save_every_steps": 1000,
    },
    "objective": {
        "control_usage": "input_only",
        "loss_weights": {
            "diffusion_mse": 1.0,
            "xstart_mse": 0.0,
            "effect_batch_mse": 0.0,
            "effect_cosine": 0.0,
        },
    },
    "checkpoint": {
        "resume_path": "",
    },
    "vae": {
        "enabled": False,
        "checkpoint_path": "",
        "pretrained_state_dir": "",
        "latent_dim": 128,
        "freeze": True,
        "seed": 0,
        "loss_ae": "mse",
        "decoder_activation": "ReLU",
        "reconstruction_loss_weight": 0.0,
        "decode_predictions": True,
        "batch_size": 512,
    },
    "evaluation": {
        "enabled": True,
        "every_steps": 1000,
        "num_batches": 1,
        "sampling_method": "ddim",
        "cfg_scale": 1.0,
    },
}


DEFAULT_INFER_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "perturbnova_infer",
        "output_dir": "./outputs/perturbnova_infer",
        "seed": 42,
        "log_render": "compact",
        "log_progress_width": 24,
    },
    "distributed": {
        "backend": "",
    },
    "checkpoint": {
        "path": "",
        "ema_rate": 0.9999,
        "use_ema": True,
    },
    "input": {
        "data_path": "",
        "reference_data_path": "",
        "split": {
            "subset": "",
            "dataset_config_path": "",
        },
        "obs_keys": {
            "perturbation": "",
            "batch": "",
            "cell_type": "",
        },
        "defaults": {
            "perturbation": "",
            "batch": "",
            "cell_type": "",
        },
    },
    "control": {
        "enabled": False,
        "label": "non-targeting",
        "samples_per_query": 32,
    },
    "sampling": {
        "batch_size": 128,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "method": "ddim",
        "cfg_scale": 1.0,
        "clip_denoised": False,
        "eta": 0.0,
        "progress": True,
    },
    "diffusion": {},
    "output": {
        "write_to": "obsm",
        "key": "X_perturbnova",
        "prediction_path": "",
        "real_copy_path": "",
        "store_latent_key": "",
        "clamp_min": 0.0,
        "clamp_max": 10.0,
    },
    "decode": {
        "enabled": False,
        "callable": "",
        "checkpoint_path": "",
        "state_dict_key": "",
        "call_method": "__call__",
        "kwargs": {},
        "call_kwargs": {},
        "batch_size": 512,
    },
    "cell_eval": {
        "enabled": False,
        "outdir": "",
        "profile": "full",
        "pert_col": "",
        "celltype_col": "",
        "control_pert": "",
        "embed_key": "",
        "num_threads": 1,
        "batch_size": 100,
        "de_method": "wilcoxon",
        "skip_metrics": "",
        "allow_discrete": False,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _normalize_relative_dataset_config_path(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    config = deepcopy(config)
    split_config = config.get("input", {}).get("split", {})
    dataset_config_path = split_config.get("dataset_config_path", "")
    if dataset_config_path:
        candidate = Path(dataset_config_path)
        if not candidate.is_absolute():
            split_config["dataset_config_path"] = str((base_dir / candidate).resolve())
    return config


def load_train_config(path: str | Path) -> dict[str, Any]:
    config = _load_toml_with_bases(path)
    config = _normalize_train_dataset_aliases(config)
    config = _normalize_state_style_dataset_config(config)
    config = _deep_merge(DEFAULT_TRAIN_CONFIG, config)
    config = _normalize_train_config(config)
    _validate_train_config(config)
    return config


def load_infer_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = _deep_merge(DEFAULT_INFER_CONFIG, _load_toml_with_bases(path))
    config = _normalize_infer_paths(config, path.parent)
    config = _normalize_infer_config(config)
    _validate_infer_config(config)
    return config


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    config = _load_toml_with_bases(path)
    config = _normalize_train_dataset_aliases(config)
    config = _normalize_state_style_dataset_config(config)
    config = _deep_merge({"dataset": deepcopy(DEFAULT_TRAIN_CONFIG["dataset"])}, config)
    return config["dataset"]


def _load_toml_with_bases(
    path: str | Path,
    visited: set[Path] | None = None,
) -> dict[str, Any]:
    path = Path(path).resolve()
    visited = visited or set()
    if path in visited:
        raise ValueError(f"Cyclic base config reference detected at: {path}")
    visited.add(path)

    current = _load_toml(path)
    base_configs = current.pop("base_configs", [])
    current = _normalize_state_style_dataset_config(current)
    current = _normalize_relative_dataset_config_path(current, path.parent)
    if isinstance(base_configs, (str, Path)):
        base_configs = [base_configs]

    merged: dict[str, Any] = {}
    for base_config in base_configs:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        merged = _deep_merge(merged, _load_toml_with_bases(base_path, visited))

    visited.remove(path)
    return _deep_merge(merged, current)


def _validate_train_config(config: dict[str, Any]) -> None:
    if config["experiment"]["log_render"] not in {"compact", "rich"}:
        raise ValueError("`experiment.log_render` must be `compact` or `rich`.")
    if config["objective"]["control_usage"] not in {"none", "input_only", "loss_only", "input_and_loss"}:
        raise ValueError(
            "`objective.control_usage` must be one of `none`, `input_only`, `loss_only`, or `input_and_loss`."
        )
    loss_weights = config["objective"]["loss_weights"]
    if float(loss_weights.get("diffusion_mse", 1.0)) <= 0:
        raise ValueError("`objective.loss_weights.diffusion_mse` must be positive.")
    for key, value in loss_weights.items():
        if float(value) < 0:
            raise ValueError(f"`objective.loss_weights.{key}` must be non-negative.")
    if not config["dataset"]["data_path"]:
        raise ValueError("`dataset.data_path` is required.")
    if config["dataset"]["name"] != "state_perturbation":
        raise ValueError("Only `state_perturbation` is supported in this refactor.")
    if config["model"]["name"] not in {"film_mlp", "additive_mlp"}:
        raise ValueError("`model.name` must be `film_mlp` or `additive_mlp`.")
    if config["training"]["mode"] not in {"vae_only", "diffusion_only", "joint"}:
        raise ValueError("`training.mode` must be `vae_only`, `diffusion_only`, or `joint`.")
    if config["training"]["stage"] not in {"", "stage1", "stage2"}:
        raise ValueError("`training.stage` must be empty, `stage1`, or `stage2`.")
    if config["dataset"]["split"]["mode"] not in {"none", "fewshot_holdout", "zeroshot_holdout"}:
        raise ValueError("`dataset.split.mode` must be `none`, `fewshot_holdout`, or `zeroshot_holdout`.")


def _validate_infer_config(config: dict[str, Any]) -> None:
    if config["experiment"]["log_render"] not in {"compact", "rich"}:
        raise ValueError("`experiment.log_render` must be `compact` or `rich`.")
    if not config["checkpoint"]["path"]:
        raise ValueError("`checkpoint.path` is required.")
    split_config = config["input"]["split"]
    if split_config["subset"] not in {"", "train", "val", "test"}:
        raise ValueError("`input.split.subset` must be empty, `train`, `val`, or `test`.")
    if not config["input"]["data_path"] and not split_config["subset"]:
        raise ValueError("`input.data_path` is required unless `input.split.subset` is set.")
    if config["cell_eval"]["profile"] not in {"full", "minimal", "vcc", "de", "anndata", "pds"}:
        raise ValueError(
            "`cell_eval.profile` must be one of `full`, `minimal`, `vcc`, `de`, `anndata`, or `pds`."
        )


def _normalize_infer_config(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    cell_eval = config.setdefault("cell_eval", {})
    profile = str(cell_eval.get("profile", "")).strip().lower()
    if profile == "squidiff":
        cell_eval["profile"] = "full"
    return config


def _normalize_train_config(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)

    training = config.setdefault("training", {})
    vae = config.setdefault("vae", {})
    objective = config.setdefault("objective", {})
    loss_weights = objective.setdefault("loss_weights", {})

    legacy_loss_keys = {
        "xstart_mse_weight": "xstart_mse",
        "effect_mse_weight": "effect_batch_mse",
        "effect_cosine_weight": "effect_cosine",
    }
    for legacy_key, target_key in legacy_loss_keys.items():
        if legacy_key in objective:
            loss_weights[target_key] = objective.pop(legacy_key)

    stage = training.get("stage", "")
    if stage == "stage1":
        training["mode"] = "vae_only"
        vae["freeze"] = False
    elif stage == "stage2":
        training["mode"] = "joint" if training.get("unfreeze_vae_in_stage2", False) else "diffusion_only"
        vae["freeze"] = not training.get("unfreeze_vae_in_stage2", False)

    return config


def _normalize_infer_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    config = deepcopy(config)
    split_config = config.setdefault("input", {}).setdefault("split", {})
    dataset_config_path = split_config.get("dataset_config_path", "")
    if dataset_config_path:
        candidate = Path(dataset_config_path)
        if not candidate.is_absolute():
            split_config["dataset_config_path"] = str((base_dir / candidate).resolve())
    return config


def _normalize_train_dataset_aliases(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    if "data" in config:
        config["dataset"] = _deep_merge(config.pop("data"), config.get("dataset", {}))
    dataset = config.get("dataset", {})
    if isinstance(dataset, dict):
        task = dataset.get("task") or dataset.get("task_type")
        if task and not dataset.get("name"):
            dataset["name"] = task
    return config


def _normalize_state_style_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    if "datasets" not in config:
        return config

    datasets = config.pop("datasets")
    raw_training = config.pop("training", {})
    raw_zeroshot = config.pop("zeroshot", {})
    raw_fewshot = config.pop("fewshot", {})

    if len(datasets) != 1:
        raise ValueError("State-style dataset configs must define exactly one dataset in `[datasets]`.")

    source_name, data_path = next(iter(datasets.items()))
    dataset = deepcopy(config.get("dataset", {}))
    dataset.setdefault("source_name", source_name)
    dataset.setdefault("dataset_name", source_name)
    dataset.setdefault("name", config.get("task") or config.get("task_type") or "state_perturbation")
    dataset.setdefault("data_path", data_path)

    split = deepcopy(dataset.get("split", {}))
    split.setdefault("mode", "none")
    split.setdefault("target_cell_type", "")
    split.setdefault("zeroshot_role", "test")
    split.setdefault("val_perturbations", [])
    split.setdefault("test_perturbations", [])

    if raw_fewshot:
        if len(raw_fewshot) != 1:
            raise ValueError("State-style fewshot configs must define exactly one entry in `[fewshot]`.")
        key, spec = next(iter(raw_fewshot.items()))
        split["mode"] = "fewshot_holdout"
        split["target_cell_type"] = _extract_target_cell_type(key, source_name)
        split["val_perturbations"] = list(spec.get("val", []))
        split["test_perturbations"] = list(spec.get("test", []))
    elif raw_zeroshot:
        if len(raw_zeroshot) != 1:
            raise ValueError("State-style zeroshot configs must define exactly one entry in `[zeroshot]`.")
        key, role = next(iter(raw_zeroshot.items()))
        split["mode"] = "zeroshot_holdout"
        split["target_cell_type"] = _extract_target_cell_type(key, source_name)
        split["zeroshot_role"] = str(role)

    dataset["split"] = split
    config["dataset"] = dataset
    config["raw_state_training"] = raw_training
    return config


def _extract_target_cell_type(key: str, source_name: str) -> str:
    prefix = f"{source_name}."
    if key.startswith(prefix):
        return key[len(prefix):]
    if "." in key:
        return key.split(".", 1)[1]
    return key
