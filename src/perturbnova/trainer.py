from __future__ import annotations

from contextlib import ExitStack, nullcontext
import re
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from .core import LossAwareSampler, build_diffusion, create_named_schedule_sampler
from .data import StatePerturbationDataModule, build_training_data_module
from .models import build_model
from .utils import ExperimentLogger, barrier, export_json, unwrap_model
from .utils.checkpoint import _ema_key, extract_state_dict, load_training_state, save_checkpoint
from .utils.distributed import DistributedContext
from .vae import build_vae_module, decode_with_vae, encode_with_vae


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _safe_r2(true_mean: np.ndarray, pred_mean: np.ndarray) -> float:
    denominator = np.sum((true_mean - true_mean.mean()) ** 2)
    if denominator <= 0:
        return 0.0
    numerator = np.sum((true_mean - pred_mean) ** 2)
    return float(1.0 - numerator / denominator)


def _safe_pearson(true_mean: np.ndarray, pred_mean: np.ndarray) -> float:
    true_std = float(np.std(true_mean))
    pred_std = float(np.std(pred_mean))
    if true_std == 0.0 or pred_std == 0.0:
        return 0.0
    return float(np.corrcoef(true_mean, pred_mean)[0, 1])


def _mmd_rbf(source: torch.Tensor, target: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> float:
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
    source_count = source.size(0)
    xx = kernels[:source_count, :source_count]
    yy = kernels[source_count:, source_count:]
    xy = kernels[:source_count, source_count:]
    yx = kernels[source_count:, :source_count]
    return float(torch.mean(xx + yy - xy - yx).item())


def _quartile_name(timestep: int, num_timesteps: int) -> str:
    quartile = int(4 * timestep / max(num_timesteps, 1))
    quartile = min(max(quartile, 0), 3)
    return f"q{quartile}"


def _accumulate_metric_buffer(
    sums: dict[str, float],
    counts: dict[str, int],
    metrics: dict[str, float],
) -> None:
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1


def _merge_metric_buffers(
    target_sums: dict[str, float],
    target_counts: dict[str, int],
    source_sums: dict[str, float],
    source_counts: dict[str, int],
) -> None:
    for key, total in source_sums.items():
        target_sums[key] = target_sums.get(key, 0.0) + float(total)
        target_counts[key] = target_counts.get(key, 0) + int(source_counts.get(key, 0))


def _average_metric_buffer(
    sums: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float]:
    averaged: dict[str, float] = {}
    for key, total in sums.items():
        count = counts.get(key, 0)
        if count > 0:
            averaged[key] = total / count
    return averaged


class DiffusionTrainer:
    def __init__(self, config: dict, distributed_context: DistributedContext) -> None:
        self.config = config
        self.context = distributed_context
        self.training_mode = config["training"]["mode"]
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExperimentLogger(
            self.output_dir,
            enabled=distributed_context.is_main_process,
            render=config["experiment"]["log_render"],
            progress_total=int(config["optimization"]["max_steps"]),
            progress_width=int(config["experiment"]["log_progress_width"]),
            run_name=config["experiment"]["name"],
            training_stage=config["training"]["stage"],
            training_mode=config["training"]["mode"],
            world_size=distributed_context.world_size,
        )
        self.data_module: StatePerturbationDataModule = build_training_data_module(config, distributed_context)
        self.vae = build_vae_module(
            config["vae"],
            input_dim=self.data_module.artifacts.raw_feature_dim,
            device=self.context.device,
        )
        self.use_vae = self.vae is not None
        self.vae_frozen = (not self.use_vae) or bool(config["vae"]["freeze"])
        self.reconstruction_loss_weight = float(config["vae"]["reconstruction_loss_weight"])
        self.ddp_vae: nn.Module | None = self.vae
        self.control_usage = str(config["objective"]["control_usage"])
        self.loss_weights = {
            key: float(value)
            for key, value in config["objective"]["loss_weights"].items()
        }
        self.use_control_input = self.control_usage in {"input_only", "input_and_loss"}
        self.use_control_loss = self.control_usage in {"loss_only", "input_and_loss"}

        if self.training_mode == "vae_only" and not self.use_vae:
            raise ValueError("`training.mode = vae_only` requires `vae.enabled = true`.")
        if self.training_mode == "vae_only" and self.vae_frozen:
            raise ValueError("`training.mode = vae_only` requires `vae.freeze = false`.")
        if self.training_mode == "diffusion_only" and self.use_vae and not self.vae_frozen:
            raise ValueError("`training.mode = diffusion_only` requires `vae.freeze = true`.")

        if self.use_vae and (self.training_mode == "vae_only" or not self.vae_frozen) and self.context.is_distributed:
            self.ddp_vae = DDP(
                self.vae,
                device_ids=[self.context.local_rank] if self.context.device.type == "cuda" else None,
                output_device=self.context.local_rank if self.context.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        self.diffusion = None
        self.model = None
        self.ddp_model = None
        self.null_perturbation_index = None
        if self.training_mode != "vae_only":
            self.diffusion = build_diffusion(config["diffusion"])
            self.model = build_model(config["model"], self.data_module.artifacts.to_dict()).to(self.context.device)
            self.ddp_model = self.model
            if self.context.is_distributed:
                ddp_find_unused = bool(config["distributed"]["find_unused_parameters"])
                # Control attention parameters are conditionally unused when control sets are disabled.
                if not bool(config["dataset"]["control"]["enabled"]) or not self.use_control_input:
                    ddp_find_unused = True
                self.ddp_model = DDP(
                    self.model,
                    device_ids=[self.context.local_rank] if self.context.device.type == "cuda" else None,
                    output_device=self.context.local_rank if self.context.device.type == "cuda" else None,
                    broadcast_buffers=False,
                    find_unused_parameters=ddp_find_unused,
                )
            self.null_perturbation_index = unwrap_model(self.ddp_model).null_perturbation_index

        trainable_parameters = []
        if self.ddp_model is not None:
            trainable_parameters.extend(list(unwrap_model(self.ddp_model).parameters()))
        if self.use_vae and (self.training_mode == "vae_only" or not self.vae_frozen):
            trainable_parameters.extend(parameter for parameter in self.vae.parameters() if parameter.requires_grad)
        self.optimizer = AdamW(
            trainable_parameters,
            lr=config["optimization"]["lr"],
            weight_decay=config["optimization"]["weight_decay"],
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=config["optimization"]["amp"] and self.context.device.type == "cuda"
        )
        self.schedule_sampler = (
            create_named_schedule_sampler(config["optimization"]["schedule_sampler"], self.diffusion)
            if self.diffusion is not None
            else None
        )
        self.ema_rates = [float(rate) for rate in config["optimization"]["ema_rates"]] if self.ddp_model is not None else []
        self.ema_state_dicts = self._initialize_ema_state() if self.ddp_model is not None else {}
        self.start_step = 0
        self.microbatch_size = config["optimization"]["microbatch_size"]
        if self.microbatch_size <= 0:
            self.microbatch_size = config["optimization"]["batch_size"]
        self.max_steps = int(config["optimization"]["max_steps"])
        self.lr_anneal_steps = int(config["optimization"]["lr_anneal_steps"])
        self.grad_clip = float(config["optimization"]["gradient_clip_norm"])
        self.loss_logging_mode = str(config["optimization"].get("loss_logging_mode", "step")).strip().lower()

        if self.context.is_main_process:
            export_json(self.output_dir / "config_snapshot.json", self.config)
            export_json(self.output_dir / "dataset_artifacts.json", self.data_module.artifacts.to_dict())
            self.logger.log_run_header()
            for split_name in ("train", "val", "test"):
                summary = self.data_module.split_summary.get(split_name, {})
                self.logger.info(
                    (
                        f"{split_name}: samples={summary.get('samples', 0):,} | "
                        f"control={summary.get('control', 0):,} | "
                        f"perturbed={summary.get('perturbed', 0):,}"
                    ),
                    tag="DATA",
                )
        self._maybe_resume()
        if self.ddp_model is not None:
            parameter_count = sum(parameter.numel() for parameter in unwrap_model(self.ddp_model).parameters())
            self.logger.info(f"Model parameters: {parameter_count:,}")
        if self.use_vae:
            vae_parameter_count = sum(parameter.numel() for parameter in self.vae.parameters())
            self.logger.info(f"VAE parameters: {vae_parameter_count:,}")
        if self.context.is_main_process:
            self._log_training_context()

    def _initialize_ema_state(self) -> dict[str, dict[str, torch.Tensor]]:
        current_state = unwrap_model(self.ddp_model).state_dict()
        return {
            _ema_key(rate): {name: tensor.detach().clone() for name, tensor in current_state.items()}
            for rate in self.ema_rates
        }

    def _maybe_resume(self) -> None:
        resume_path = self.config["checkpoint"]["resume_path"]
        if not resume_path:
            return
        if self.training_mode == "vae_only":
            payload = self._load_vae_only_state(resume_path)
            step = int(payload.get("step", self._parse_step_from_name(str(resume_path))))
            self.start_step = step + 1
            self.logger.info(f"Resumed VAE from step {self.start_step}")
            self.logger.log_resume(self.start_step, self._resolve_vae_resume_path(resume_path), kind="resume")
            return
        payload = load_training_state(
            resume_path,
            model=self.ddp_model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            vae=self.vae,
            map_location=self.context.device,
        )
        loaded_ema = payload.get("ema") or {}
        if loaded_ema:
            self.ema_state_dicts = {
                key: {name: tensor.to(self.context.device) for name, tensor in state.items()}
                for key, state in loaded_ema.items()
            }
        self.start_step = int(payload.get("step", 0)) + 1
        self.logger.info(f"Resumed from step {self.start_step}")
        self.logger.log_resume(self.start_step, resume_path, kind="resume")

    def _resolve_vae_resume_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_dir():
            nested = candidate / "vae_checkpoints" / "latest.pt"
            if nested.exists():
                return nested
            latest = candidate / "latest.pt"
            if latest.exists():
                return latest
        return candidate

    def _parse_step_from_name(self, path: str) -> int:
        match = re.search(r"step[=_-](\d+)", path)
        return int(match.group(1)) if match else 0

    def _load_vae_only_state(self, path: str | Path) -> dict:
        resolved = self._resolve_vae_resume_path(path)
        payload = torch.load(resolved, map_location=self.context.device, weights_only=False)
        state_dict = extract_state_dict(payload)
        self.vae.load_state_dict(state_dict, strict=True)
        return payload if isinstance(payload, dict) else {"step": self._parse_step_from_name(str(resolved))}

    def _batch_iterator(self, loader, sampler) -> Iterator[dict[str, torch.Tensor]]:
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                yield batch
            epoch += 1

    def _autocast_context(self):
        if self.scaler.is_enabled():
            return torch.autocast(device_type=self.context.device.type, enabled=True)
        return nullcontext()

    def _vae_module(self):
        return self.ddp_vae if self.ddp_vae is not None else self.vae

    def _prepare_micro_condition(self, batch: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
        condition = {
            "perturbation": batch["perturbation"][start:end].clone(),
            "batch": batch["batch"][start:end],
            "cell_type": batch["cell_type"][start:end],
            "control_set": batch.get("control_set", None),
        }
        if condition["control_set"] is not None:
            condition["control_set"] = condition["control_set"][start:end]
        drop_prob = float(self.config["model"]["perturbation_dropout"])
        if drop_prob > 0.0:
            drop_mask = torch.rand(condition["perturbation"].shape[0], device=self.context.device) < drop_prob
            condition["perturbation"][drop_mask] = self.null_perturbation_index
        return condition

    def _encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.use_vae:
            return tensor
        if self.vae_frozen:
            with torch.no_grad():
                return encode_with_vae(self._vae_module(), tensor)
        return encode_with_vae(self._vae_module(), tensor)

    def _encode_control_set(self, control_set: torch.Tensor | None) -> torch.Tensor | None:
        if control_set is None:
            return None
        if not self.use_vae:
            return control_set
        batch_size, control_count, feature_dim = control_set.shape
        encoded = self._encode_tensor(control_set.reshape(-1, feature_dim))
        return encoded.reshape(batch_size, control_count, -1)

    def _decode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return decode_with_vae(self._vae_module(), tensor)

    def _update_ema(self) -> None:
        current_state = unwrap_model(self.ddp_model).state_dict()
        for rate in self.ema_rates:
            key = _ema_key(rate)
            ema_state = self.ema_state_dicts[key]
            for name, tensor in current_state.items():
                ema_tensor = ema_state[name]
                if torch.is_floating_point(tensor):
                    ema_tensor.mul_(rate).add_(tensor.detach(), alpha=1.0 - rate)
                else:
                    ema_tensor.copy_(tensor)

    def _anneal_lr(self, step: int) -> None:
        if self.lr_anneal_steps <= 0:
            return
        fraction_done = min(step / float(self.lr_anneal_steps), 1.0)
        lr = self.config["optimization"]["lr"] * (1.0 - fraction_done)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _log_training_context(self) -> None:
        dataset_config = self.config["dataset"]
        self.logger.log_mapping(
            "env",
            {
                "device": self.context.device,
                "backend": self.context.backend,
                "world": self.context.world_size,
                "seed": self.config["experiment"]["seed"],
            },
            tag="ENV",
        )
        self.logger.log_mapping(
            "dataset",
            {
                "path": dataset_config["data_path"],
                "split_mode": dataset_config["split"]["mode"],
                "target_cell_type": dataset_config["split"].get("target_cell_type", ""),
                "feature_source": dataset_config["feature_space"]["source"],
                "feature_key": dataset_config["feature_space"]["key"],
                "raw_dim": self.data_module.artifacts.raw_feature_dim,
                "latent_dim": self.data_module.artifacts.feature_dim,
            },
            tag="DATA",
        )
        if self.ddp_model is not None:
            self.logger.log_mapping(
                "model",
                {
                    "name": self.config["model"]["name"],
                    "hidden_dim": self.config["model"]["hidden_dim"],
                    "layers": self.config["model"]["num_layers"],
                    "time_embed_dim": self.config["model"]["time_embed_dim"],
                    "dropout": self.config["model"]["dropout"],
                    "perturb_dropout": self.config["model"]["perturbation_dropout"],
                },
                tag="MODEL",
            )
            self.logger.log_mapping(
                "diffusion",
                {
                    "steps": self.config["diffusion"]["steps"],
                    "noise_schedule": self.config["diffusion"]["noise_schedule"],
                    "timestep_respacing": self.config["diffusion"]["timestep_respacing"] or "",
                    "predict_xstart": self.config["diffusion"]["predict_xstart"],
                },
                tag="DIFF",
            )
        self.logger.log_mapping(
            "optimization",
            {
                "batch_size": self.config["optimization"]["batch_size"],
                "microbatch_size": self.microbatch_size,
                "max_steps": self.max_steps,
                "lr": self.config["optimization"]["lr"],
                "weight_decay": self.config["optimization"]["weight_decay"],
                "schedule_sampler": self.config["optimization"]["schedule_sampler"],
                "ema_rates": ",".join(str(x) for x in self.ema_rates) if self.ema_rates else "",
                "grad_clip": self.grad_clip,
            },
            tag="OPT",
        )
        self.logger.log_mapping(
            "objective",
            {
                "training_mode": self.training_mode,
                "control_usage": self.control_usage,
                **self.loss_weights,
            },
            tag="LOSS",
        )
        self.logger.log_mapping(
            "vae",
            {
                "enabled": self.use_vae,
                "freeze": self.vae_frozen,
                "checkpoint": self.config["vae"].get("checkpoint_path", ""),
                "pretrained_state_dir": self.config["vae"].get("pretrained_state_dir", ""),
                "latent_dim": self.config["vae"].get("latent_dim", ""),
                "recon_weight": self.reconstruction_loss_weight,
                "decode_predictions": self.config["vae"].get("decode_predictions", True),
            },
            tag="VAE",
        )
        if self.config["checkpoint"]["resume_path"]:
            self.logger.log_mapping(
                "resume",
                {
                    "path": self.config["checkpoint"]["resume_path"],
                    "start_step": self.start_step,
                },
                tag="CKPT",
            )

    def _accumulate_bucketed_losses(
        self,
        bucket_sums: dict[str, float],
        bucket_counts: dict[str, int],
        timesteps: torch.Tensor,
        values: torch.Tensor,
        key: str,
    ) -> None:
        if self.diffusion is None:
            return
        ts = timesteps.detach().cpu().tolist()
        vals = values.detach().cpu().tolist()
        for sub_t, sub_value in zip(ts, vals):
            bucket_name = _quartile_name(int(sub_t), self.diffusion.num_timesteps)
            metric_key = f"{key}_{bucket_name}"
            bucket_sums[metric_key] = bucket_sums.get(metric_key, 0.0) + float(sub_value)
            bucket_counts[metric_key] = bucket_counts.get(metric_key, 0) + 1

    def _accumulate_squidiff_style_losses(
        self,
        log_sums: dict[str, float],
        log_counts: dict[str, int],
        timesteps: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> None:
        for key, value in loss_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim != 1 or value.shape[0] != timesteps.shape[0]:
                continue
            weighted = value * weights
            log_sums[key] = log_sums.get(key, 0.0) + float(weighted.mean().detach().item())
            log_counts[key] = log_counts.get(key, 0) + 1
            self._accumulate_bucketed_losses(
                bucket_sums=log_sums,
                bucket_counts=log_counts,
                timesteps=timesteps,
                values=weighted,
                key=key,
            )

    def _run_train_step(self, batch: dict[str, torch.Tensor], step: int) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
        self.optimizer.zero_grad(set_to_none=True)
        batch = _move_batch_to_device(batch, self.context.device)
        batch_size = batch["features"].shape[0]
        accumulated = {"loss": 0.0, "mse": 0.0}
        bucket_sums: dict[str, float] = {}
        bucket_counts: dict[str, int] = {}
        log_sums: dict[str, float] = {}
        log_counts: dict[str, int] = {}
        if self.loss_weights.get("xstart_mse", 0.0) > 0:
            accumulated["xstart_mse"] = 0.0
        if self.loss_weights.get("effect_batch_mse", 0.0) > 0:
            accumulated["effect_batch_mse"] = 0.0
        if self.loss_weights.get("effect_cosine", 0.0) > 0:
            accumulated["effect_cosine"] = 0.0
        if self.use_vae and not self.vae_frozen and self.reconstruction_loss_weight > 0:
            accumulated["vae_reconstruction"] = 0.0

        for start in range(0, batch_size, self.microbatch_size):
            end = min(start + self.microbatch_size, batch_size)
            micro_raw_features = batch["features"][start:end]
            micro_condition = self._prepare_micro_condition(batch, start, end)
            micro_features = self._encode_tensor(micro_raw_features)
            encoded_control_set = self._encode_control_set(micro_condition["control_set"])
            control_anchor = encoded_control_set.mean(dim=1) if encoded_control_set is not None else None
            micro_condition["control_set"] = encoded_control_set if self.use_control_input else None
            timesteps, weights = self.schedule_sampler.sample(micro_features.shape[0], self.context.device)
            with ExitStack() as stack:
                if self.context.is_distributed and end < batch_size:
                    stack.enter_context(self.ddp_model.no_sync())
                    if self.use_vae and not self.vae_frozen and hasattr(self.ddp_vae, "no_sync"):
                        stack.enter_context(self.ddp_vae.no_sync())
                with self._autocast_context():
                    loss_dict = self.diffusion.training_losses(
                        self.ddp_model,
                        micro_features,
                        timesteps,
                        model_kwargs=micro_condition,
                    )
                    diffusion_loss = (loss_dict["loss"] * weights).mean()
                    raw_loss = self.loss_weights.get("diffusion_mse", 1.0) * diffusion_loss
                    xstart_mse = None
                    effect_batch_mse = None
                    effect_cosine = None
                    pred_xstart = loss_dict.get("pred_xstart")
                    if self.loss_weights.get("xstart_mse", 0.0) > 0 and "xstart_mse" in loss_dict:
                        xstart_mse = (loss_dict["xstart_mse"] * weights).mean()
                        raw_loss = raw_loss + self.loss_weights["xstart_mse"] * xstart_mse
                    if self.use_control_loss and control_anchor is not None and pred_xstart is not None:
                        control_anchor_mean = control_anchor.mean(dim=0, keepdim=True)
                        pred_effect_mean = pred_xstart.mean(dim=0, keepdim=True) - control_anchor_mean
                        true_effect_mean = micro_features.mean(dim=0, keepdim=True) - control_anchor_mean
                        if self.loss_weights.get("effect_batch_mse", 0.0) > 0:
                            effect_batch_mse = F.mse_loss(pred_effect_mean, true_effect_mean)
                            raw_loss = raw_loss + self.loss_weights["effect_batch_mse"] * effect_batch_mse
                        if self.loss_weights.get("effect_cosine", 0.0) > 0:
                            pred_effect = pred_xstart - control_anchor
                            true_effect = micro_features - control_anchor
                            effect_cosine = 1.0 - F.cosine_similarity(
                                pred_effect,
                                true_effect,
                                dim=1,
                                eps=1e-8,
                            ).mean()
                            raw_loss = raw_loss + self.loss_weights["effect_cosine"] * effect_cosine
                    if self.use_vae and not self.vae_frozen and self.reconstruction_loss_weight > 0:
                        reconstruction = self._decode_tensor(micro_features)
                        reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, micro_raw_features)
                        raw_loss = raw_loss + self.reconstruction_loss_weight * reconstruction_loss
                    scaled_loss = raw_loss * (micro_features.shape[0] / batch_size)
                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(timesteps, loss_dict["loss"].detach())
                self.scaler.scale(scaled_loss).backward()
                self._accumulate_squidiff_style_losses(
                    log_sums=log_sums,
                    log_counts=log_counts,
                    timesteps=timesteps,
                    loss_dict=loss_dict,
                    weights=weights,
                )

            fraction = micro_features.shape[0] / batch_size
            accumulated["loss"] += raw_loss.detach().item() * fraction
            if "mse" in loss_dict:
                accumulated["mse"] += ((loss_dict["mse"] * weights).mean().detach().item()) * fraction
            if xstart_mse is not None:
                accumulated["xstart_mse"] += xstart_mse.detach().item() * fraction
            if effect_batch_mse is not None:
                accumulated["effect_batch_mse"] += effect_batch_mse.detach().item() * fraction
            if effect_cosine is not None:
                accumulated["effect_cosine"] += effect_cosine.detach().item() * fraction
            if self.use_vae and not self.vae_frozen and self.reconstruction_loss_weight > 0:
                accumulated["vae_reconstruction"] += reconstruction_loss.detach().item() * fraction

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            if self.training_mode == "joint":
                parameters = list(unwrap_model(self.ddp_model).parameters()) + list(self.vae.parameters())
                torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(unwrap_model(self.ddp_model).parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._update_ema()
        self._anneal_lr(step + 1)
        for key, total in log_sums.items():
            count = log_counts.get(key, 0)
            if count > 0:
                accumulated[key] = total / count
        accumulated["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return accumulated, log_sums, log_counts

    def _run_vae_only_step(self, batch: dict[str, torch.Tensor], step: int) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
        self.optimizer.zero_grad(set_to_none=True)
        batch = _move_batch_to_device(batch, self.context.device)
        batch_size = batch["features"].shape[0]
        accumulated = {"vae_reconstruction": 0.0}
        log_sums: dict[str, float] = {}
        log_counts: dict[str, int] = {}

        for start in range(0, batch_size, self.microbatch_size):
            end = min(start + self.microbatch_size, batch_size)
            micro_features = batch["features"][start:end]
            sync_context = (
                self.ddp_vae.no_sync()
                if self.context.is_distributed and hasattr(self.ddp_vae, "no_sync") and end < batch_size
                else nullcontext()
            )
            with sync_context:
                with self._autocast_context():
                    reconstruction = self._vae_module()(micro_features)
                    raw_loss = torch.nn.functional.mse_loss(reconstruction, micro_features)
                    scaled_loss = raw_loss * (micro_features.shape[0] / batch_size)
                self.scaler.scale(scaled_loss).backward()
            accumulated["vae_reconstruction"] += raw_loss.detach().item() * (micro_features.shape[0] / batch_size)
            log_sums["vae_reconstruction"] = log_sums.get("vae_reconstruction", 0.0) + raw_loss.detach().item()
            log_counts["vae_reconstruction"] = log_counts.get("vae_reconstruction", 0) + 1

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._anneal_lr(step + 1)
        accumulated["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return accumulated, log_sums, log_counts

    def _sample_predictions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        method = self.config["evaluation"]["sampling_method"]
        sample_fn = (
            self.diffusion.ddim_sample_loop
            if method == "ddim"
            else self.diffusion.p_sample_loop
        )
        latent_control_set = self._encode_control_set(batch.get("control_set")) if self.use_control_input else None
        return sample_fn(
            unwrap_model(self.ddp_model),
            shape=(batch["features"].shape[0], self.data_module.artifacts.feature_dim),
            model_kwargs={
                "perturbation": batch["perturbation"],
                "batch": batch["batch"],
                "cell_type": batch["cell_type"],
                "control_set": latent_control_set,
            },
            device=self.context.device,
            progress=False,
            cfg_scale=self.config["evaluation"]["cfg_scale"],
        )

    def _evaluate_vae(self, step: int) -> None:
        if not self.context.is_main_process:
            barrier(self.context)
            barrier(self.context)
            return
        barrier(self.context)
        loader = self.data_module.val_loader or self.data_module.train_loader
        losses = []
        self.vae.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                batch = _move_batch_to_device(batch, self.context.device)
                reconstruction = self.vae(batch["features"])
                losses.append(torch.nn.functional.mse_loss(reconstruction, batch["features"]).item())
                if batch_index + 1 >= int(self.config["evaluation"]["num_batches"]):
                    break
        if self.training_mode == "vae_only":
            self.vae.train()
        self.logger.log_metrics(step=step, split="eval", metrics={"vae_reconstruction": float(np.mean(losses))})
        barrier(self.context)

    def evaluate(self, step: int) -> None:
        if not self.config["evaluation"]["enabled"]:
            return
        if not self.context.is_main_process:
            barrier(self.context)
            barrier(self.context)
            return

        barrier(self.context)
        loader = self.data_module.val_loader or self.data_module.train_loader
        batches = []
        for batch_index, batch in enumerate(loader):
            batches.append(batch)
            if batch_index + 1 >= int(self.config["evaluation"]["num_batches"]):
                break

        unwrap_model(self.ddp_model).eval()
        if self.use_vae:
            self.vae.eval()
        metric_buffer = {"r2_mean": [], "pearson_mean": [], "mmd": []}
        with torch.no_grad():
            for batch in batches:
                batch = _move_batch_to_device(batch, self.context.device)
                latent_truth = self._encode_tensor(batch["features"])
                predictions = self._sample_predictions(batch)
                truth_np = latent_truth.detach().cpu().numpy()
                pred_np = predictions.detach().cpu().numpy()
                metric_buffer["r2_mean"].append(_safe_r2(truth_np.mean(axis=0), pred_np.mean(axis=0)))
                metric_buffer["pearson_mean"].append(
                    _safe_pearson(truth_np.mean(axis=0), pred_np.mean(axis=0))
                )
                metric_buffer["mmd"].append(
                    _mmd_rbf(
                        latent_truth.detach().cpu(),
                        predictions.detach().cpu(),
                    )
                )
        unwrap_model(self.ddp_model).train()
        if self.use_vae and not self.vae_frozen:
            self.vae.train()
        metrics = {key: float(np.mean(values)) for key, values in metric_buffer.items()}
        self.logger.log_metrics(step=step, split="eval", metrics=metrics)
        barrier(self.context)

    def train(self) -> None:
        train_iterator = self._batch_iterator(self.data_module.train_loader, self.data_module.train_sampler)
        if self.ddp_model is not None:
            unwrap_model(self.ddp_model).train()
        if self.use_vae:
            if self.training_mode == "diffusion_only" and self.vae_frozen:
                self.vae.eval()
            else:
                self.vae.train()
        train_metric_sums: dict[str, float] = {}
        train_metric_counts: dict[str, int] = {}
        for step in range(self.start_step, self.max_steps):
            batch = next(train_iterator)
            if self.training_mode == "vae_only":
                train_metrics, log_sums, log_counts = self._run_vae_only_step(batch, step)
            else:
                train_metrics, log_sums, log_counts = self._run_train_step(batch, step)
            if self.loss_logging_mode == "squidiff":
                _merge_metric_buffers(train_metric_sums, train_metric_counts, log_sums, log_counts)
            else:
                _accumulate_metric_buffer(train_metric_sums, train_metric_counts, train_metrics)
            if self.context.is_main_process and (
                step == self.start_step or (step + 1) % self.config["optimization"]["log_every_steps"] == 0
            ):
                averaged_metrics = _average_metric_buffer(train_metric_sums, train_metric_counts)
                self.logger.log_metrics(step=step + 1, split="train", metrics=averaged_metrics)
                train_metric_sums.clear()
                train_metric_counts.clear()

            if (step + 1) % self.config["optimization"]["save_every_steps"] == 0:
                if self.training_mode == "vae_only":
                    self._save_vae_checkpoint(step + 1)
                else:
                    save_checkpoint(
                        output_dir=self.output_dir,
                        step=step + 1,
                        model=self.ddp_model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        ema_state_dicts=self.ema_state_dicts,
                        config=self.config,
                        dataset_artifacts=self.data_module.artifacts.to_dict(),
                        vae=self.vae,
                        enabled=self.context.is_main_process,
                    )
                    if self.context.is_main_process:
                        self.logger.log_checkpoint(step + 1, self.output_dir / "checkpoints" / "latest.pt")
                barrier(self.context)

            if self.config["evaluation"]["enabled"] and (step + 1) % self.config["evaluation"]["every_steps"] == 0:
                if self.training_mode == "vae_only":
                    self._evaluate_vae(step + 1)
                else:
                    self.evaluate(step + 1)

        if self.training_mode == "vae_only":
            self._save_vae_checkpoint(self.max_steps)
        else:
            save_checkpoint(
                output_dir=self.output_dir,
                step=self.max_steps,
                model=self.ddp_model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                ema_state_dicts=self.ema_state_dicts,
                config=self.config,
                dataset_artifacts=self.data_module.artifacts.to_dict(),
                vae=self.vae,
                enabled=self.context.is_main_process,
            )
            if self.context.is_main_process:
                self.logger.log_checkpoint(self.max_steps, self.output_dir / "checkpoints" / "latest.pt")
        barrier(self.context)

    def _save_vae_checkpoint(self, step: int) -> None:
        if not self.context.is_main_process:
            return
        checkpoint_dir = self.output_dir / "vae_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": step,
            "vae": {name: tensor.detach().cpu() for name, tensor in self.vae.state_dict().items()},
            "config": self.config,
            "dataset_artifacts": self.data_module.artifacts.to_dict(),
        }
        seed = int(self.config["experiment"]["seed"])
        step_path = checkpoint_dir / f"model_seed={seed}_step={step}.pt"
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(payload, step_path)
        shutil.copy2(step_path, latest_path)
        self.logger.log_checkpoint(step, latest_path, kind="vae")
