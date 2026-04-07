from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def export_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _ema_key(rate: float | str) -> str:
    return f"{float(rate):.10g}"


def resolve_checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        nested = candidate / "checkpoints" / "latest.pt"
        if nested.exists():
            return nested
        latest = candidate / "latest.pt"
        if latest.exists():
            return latest
    return candidate


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model,
    optimizer,
    scaler,
    ema_state_dicts: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
    dataset_artifacts: dict[str, Any],
    enabled: bool,
    vae=None,
) -> None:
    if not enabled:
        return

    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_state = {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()}
    ema_state = {
        key: {name: tensor.detach().cpu() for name, tensor in value.items()}
        for key, value in ema_state_dicts.items()
    }
    payload = {
        "step": step,
        "model": model_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema_state,
        "config": config,
        "dataset_artifacts": dataset_artifacts,
        "vae": ({k: v.detach().cpu() for k, v in vae.state_dict().items()} if vae is not None else None),
    }
    step_path = checkpoint_dir / f"step_{step:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, step_path)
    shutil.copy2(step_path, latest_path)


def load_checkpoint_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    resolved = resolve_checkpoint_path(path)
    return torch.load(resolved, map_location=map_location, weights_only=False)


def load_training_state(
    path: str | Path,
    model,
    optimizer=None,
    scaler=None,
    vae=None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    unwrap_model(model).load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if vae is not None and payload.get("vae") is not None:
        vae.load_state_dict(payload["vae"])
    return payload


def select_state_dict_for_inference(payload: dict[str, Any], ema_rate: float | None) -> dict[str, torch.Tensor]:
    ema_states = payload.get("ema") or {}
    if ema_rate is not None:
        key = _ema_key(ema_rate)
        if key in ema_states:
            return ema_states[key]
    if ema_states:
        sorted_keys = sorted(ema_states.keys(), key=lambda item: float(item))
        return ema_states[sorted_keys[-1]]
    return payload["model"]


def extract_state_dict(raw_state: Any, state_dict_key: str = "") -> dict[str, torch.Tensor]:
    if state_dict_key:
        return raw_state[state_dict_key]
    if isinstance(raw_state, dict) and "vae" in raw_state:
        return raw_state["vae"]
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        return raw_state["state_dict"]
    if isinstance(raw_state, dict) and "model" in raw_state:
        return raw_state["model"]
    return raw_state
