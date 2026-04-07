from __future__ import annotations

import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from tqdm.auto import tqdm

from .core import build_diffusion
from .data import StateDataArtifacts, build_inference_loader
from .models import build_model
from .utils import ExperimentLogger, barrier, export_json, import_string
from .utils.checkpoint import (
    extract_state_dict,
    load_checkpoint_payload,
    select_state_dict_for_inference,
)
from .utils.distributed import DistributedContext
from .vae import build_vae_module, decode_array_with_vae, encode_with_vae


class DecoderAdapter:
    def __init__(self, decode_config: dict, device: torch.device) -> None:
        self.enabled = decode_config["enabled"]
        self.device = device
        self.call_method = decode_config["call_method"]
        self.call_kwargs = decode_config["call_kwargs"]
        self.module = None

        if not self.enabled:
            return

        constructor = import_string(decode_config["callable"])
        self.module = constructor(**decode_config["kwargs"])
        if decode_config["checkpoint_path"]:
            raw_state = torch.load(
                decode_config["checkpoint_path"],
                map_location="cpu",
                weights_only=False,
            )
            state_dict = extract_state_dict(raw_state, decode_config["state_dict_key"])
            self.module.load_state_dict(state_dict)
        self.module.to(self.device)
        self.module.eval()

    def decode(self, values: np.ndarray, batch_size: int) -> np.ndarray:
        if not self.enabled or self.module is None:
            return values

        outputs = []
        with torch.no_grad():
            for start in range(0, values.shape[0], batch_size):
                end = min(start + batch_size, values.shape[0])
                batch_tensor = torch.from_numpy(values[start:end]).to(self.device)
                method = self.module if self.call_method == "__call__" else getattr(self.module, self.call_method)
                decoded = method(batch_tensor, **self.call_kwargs)
                if isinstance(decoded, tuple):
                    decoded = decoded[0]
                if not isinstance(decoded, torch.Tensor):
                    raise TypeError("Decoder output must be a torch.Tensor.")
                outputs.append(decoded.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


class DiffusionInferenceRunner:
    def __init__(self, config: dict, distributed_context: DistributedContext) -> None:
        self.config = config
        self.context = distributed_context
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExperimentLogger(
            self.output_dir,
            enabled=self.context.is_main_process,
            render=config["experiment"]["log_render"],
            progress_width=int(config["experiment"]["log_progress_width"]),
            run_name=config["experiment"]["name"],
            world_size=distributed_context.world_size,
        )
        self.payload = load_checkpoint_payload(config["checkpoint"]["path"], map_location="cpu")
        self.train_config = self.payload["config"]
        self.artifacts = StateDataArtifacts.from_dict(self.payload["dataset_artifacts"])
        self.vae = build_vae_module(
            self.train_config.get("vae", {"enabled": False}),
            input_dim=self.artifacts.raw_feature_dim,
            device=self.context.device,
        )
        if self.vae is not None and self.payload.get("vae") is not None:
            self.vae.load_state_dict(self.payload["vae"])
            self.vae.eval()
        self.model = build_model(self.train_config["model"], self.artifacts.to_dict()).to(self.context.device)
        self.model.load_state_dict(
            select_state_dict_for_inference(self.payload, config["checkpoint"]["ema_rate"])
        )
        self.model.eval()
        self.diffusion = build_diffusion(self.train_config["diffusion"])
        self.decoder = DecoderAdapter(config["decode"], self.context.device)

        if self.context.is_main_process:
            export_json(self.output_dir / "infer_config_snapshot.json", self.config)
            self.logger.log_run_header()

    def _resolve_prediction_path(self) -> Path:
        output_config = self.config["output"]
        return Path(output_config["prediction_path"] or (self.output_dir / "predictions.h5ad"))

    def _resolve_real_copy_path(self) -> Path | None:
        output_config = self.config["output"]
        if output_config["real_copy_path"]:
            return Path(output_config["real_copy_path"])
        if self.config.get("cell_eval", {}).get("enabled", False):
            return self.output_dir / "cell_eval_real.h5ad"
        return None

    def _sample_fn(self):
        return (
            self.diffusion.ddim_sample_loop
            if self.config["sampling"]["method"] == "ddim"
            else self.diffusion.p_sample_loop
        )

    def _prepare_shard_dir(self) -> Path:
        shard_dir = self.output_dir / ".prediction_shards"
        if self.context.is_main_process and shard_dir.exists():
            shutil.rmtree(shard_dir)
        barrier(self.context)
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir

    def _write_predictions(self, target_adata, latent_predictions: np.ndarray) -> None:
        output_config = self.config["output"]
        ad.settings.allow_write_nullable_strings = True
        latent_predictions = latent_predictions.astype(np.float32)
        vae_predictions = latent_predictions
        if self.vae is not None and self.train_config.get("vae", {}).get("decode_predictions", True):
            vae_predictions = decode_array_with_vae(
                self.vae,
                latent_predictions,
                device=self.context.device,
                batch_size=int(self.train_config.get("vae", {}).get("batch_size", 512)),
            )
        vae_predictions = np.clip(
            vae_predictions,
            output_config["clamp_min"],
            output_config["clamp_max"],
        ).astype(np.float32)
        final_predictions = self.decoder.decode(
            vae_predictions,
            batch_size=int(self.config["decode"]["batch_size"]),
        )

        prediction_path = self._resolve_prediction_path()
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_adata = target_adata.copy()

        if output_config["write_to"] == "X":
            if final_predictions.shape[1] != prediction_adata.n_vars:
                raise ValueError(
                    "Cannot write predictions to `X` because feature dimension does not match `n_vars`."
                )
            prediction_adata.X = final_predictions
            if output_config["store_latent_key"]:
                prediction_adata.obsm[output_config["store_latent_key"]] = latent_predictions
        elif output_config["write_to"] == "obsm":
            prediction_adata.obsm[output_config["key"]] = final_predictions
            if output_config["store_latent_key"]:
                prediction_adata.obsm[output_config["store_latent_key"]] = latent_predictions
        else:
            raise ValueError("`output.write_to` must be `X` or `obsm`.")

        prediction_adata.write_h5ad(prediction_path)
        real_copy_path = self._resolve_real_copy_path()
        if real_copy_path is not None:
            real_copy_path.parent.mkdir(parents=True, exist_ok=True)
            target_adata.write_h5ad(real_copy_path)
        self.logger.info(f"saved_predictions={prediction_path}", tag="INFER")

    def run(self) -> None:
        target_adata, loader, sampler = build_inference_loader(
            infer_config=self.config,
            train_dataset_config=self.train_config["dataset"],
            artifacts=self.artifacts,
            distributed_context=self.context,
        )
        if sampler is not None:
            sampler.set_epoch(0)

        shard_dir = self._prepare_shard_dir()
        sample_fn = self._sample_fn()
        all_indices = []
        all_predictions = []
        iterator = tqdm(
            loader,
            disable=not (self.context.is_main_process and self.config["sampling"]["progress"]),
        )
        with torch.no_grad():
            for batch in iterator:
                indices = batch["index"].numpy()
                model_kwargs = {
                    "perturbation": batch["perturbation"].to(self.context.device, non_blocking=True),
                    "batch": batch["batch"].to(self.context.device, non_blocking=True),
                    "cell_type": batch["cell_type"].to(self.context.device, non_blocking=True),
                    "control_set": batch.get("control_set"),
                }
                if model_kwargs["control_set"] is not None:
                    model_kwargs["control_set"] = model_kwargs["control_set"].to(
                        self.context.device,
                        non_blocking=True,
                    )
                    if self.vae is not None:
                        batch_size, control_count, feature_dim = model_kwargs["control_set"].shape
                        model_kwargs["control_set"] = encode_with_vae(
                            self.vae,
                            model_kwargs["control_set"].reshape(-1, feature_dim),
                        ).reshape(batch_size, control_count, -1)

                sample_kwargs = {
                    "model": self.model,
                    "shape": (indices.shape[0], self.artifacts.feature_dim),
                    "model_kwargs": model_kwargs,
                    "device": self.context.device,
                    "progress": False,
                    "clip_denoised": self.config["sampling"]["clip_denoised"],
                    "cfg_scale": self.config["sampling"]["cfg_scale"],
                }
                if self.config["sampling"]["method"] == "ddim":
                    sample_kwargs["eta"] = self.config["sampling"]["eta"]

                predictions = sample_fn(
                    self.model,
                    shape=sample_kwargs["shape"],
                    model_kwargs=sample_kwargs["model_kwargs"],
                    device=sample_kwargs["device"],
                    progress=sample_kwargs["progress"],
                    clip_denoised=sample_kwargs["clip_denoised"],
                    cfg_scale=sample_kwargs["cfg_scale"],
                    **({"eta": sample_kwargs["eta"]} if "eta" in sample_kwargs else {}),
                )
                all_indices.append(indices)
                all_predictions.append(predictions.detach().cpu().numpy().astype(np.float32))

        shard_path = shard_dir / f"rank_{self.context.rank:04d}.pt"
        torch.save(
            {
                "indices": np.concatenate(all_indices, axis=0) if all_indices else np.zeros(0, dtype=np.int64),
                "predictions": (
                    np.concatenate(all_predictions, axis=0) if all_predictions else np.zeros((0, self.artifacts.feature_dim), dtype=np.float32)
                ),
            },
            shard_path,
        )
        barrier(self.context)

        if self.context.is_main_process:
            merged = np.zeros((target_adata.n_obs, self.artifacts.feature_dim), dtype=np.float32)
            filled = np.zeros(target_adata.n_obs, dtype=bool)
            for path in sorted(shard_dir.glob("rank_*.pt")):
                shard = torch.load(path, map_location="cpu", weights_only=False)
                shard_indices = shard["indices"]
                shard_predictions = shard["predictions"]
                merged[shard_indices] = shard_predictions
                filled[shard_indices] = True
            if not filled.all():
                missing = int((~filled).sum())
                raise RuntimeError(f"Inference merge failed, {missing} samples are missing.")
            self._write_predictions(target_adata, merged)
            shutil.rmtree(shard_dir, ignore_errors=True)

        barrier(self.context)
