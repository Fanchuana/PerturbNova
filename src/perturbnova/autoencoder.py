from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .evoformer_ae import (
    EvoformerAutoencoder,
    build_evoformer_ae_module,
    encode_with_evoformer_ae,
)
from .utils import unwrap_model
from .vae import build_vae_module, decode_with_vae, encode_with_vae


class AutoencoderAdapter:
    def __init__(self, module: nn.Module | None, config: dict) -> None:
        self.module = module
        self.config = config
        self.kind = str(config.get("type", "mlp"))
        self.representation = str(config.get("representation", "latent"))

    @property
    def enabled(self) -> bool:
        return self.module is not None

    @property
    def uses_evoformer_pretrain_embedding(self) -> bool:
        return self.kind == "evoformer" and self.representation == "pretrain_embedding"

    def encode(self, tensor: torch.Tensor, module: nn.Module | None = None) -> torch.Tensor:
        if self.module is None:
            return tensor
        active_module = module or self.module
        if self.uses_evoformer_pretrain_embedding:
            return active_module(tensor, mode="pretrain")["embedding"]
        if self.kind == "evoformer":
            return encode_with_evoformer_ae(active_module, tensor)
        return encode_with_vae(active_module, tensor)

    def decode(self, tensor: torch.Tensor, module: nn.Module | None = None) -> torch.Tensor:
        if self.module is None:
            return tensor
        active_module = module or self.module
        if self.uses_evoformer_pretrain_embedding:
            return unwrap_model(active_module).pred_head.output(tensor)
        if self.kind == "evoformer":
            return unwrap_model(active_module).decode(tensor)
        return decode_with_vae(active_module, tensor)

    def decode_array(
        self,
        values: np.ndarray,
        device: torch.device,
        batch_size: int = 512,
        module: nn.Module | None = None,
    ) -> np.ndarray:
        if self.module is None:
            return values.astype(np.float32, copy=False)
        active_module = module or self.module
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                end = min(start + batch_size, len(values))
                batch = torch.as_tensor(values[start:end], dtype=torch.float32, device=device)
                outputs.append(self.decode(batch, module=active_module).detach().cpu())
        return torch.cat(outputs, dim=0).numpy().astype("float32")


def build_autoencoder_adapter(
    vae_config: dict,
    input_dim: int,
    device: torch.device,
) -> AutoencoderAdapter:
    vae_type = vae_config.get("type", "mlp")
    if vae_type == "evoformer":
        module = build_evoformer_ae_module(
            vae_config,
            input_dim=input_dim,
            device=device,
        )
    else:
        module = build_vae_module(
            vae_config,
            input_dim=input_dim,
            device=device,
        )
    return AutoencoderAdapter(module=module, config=vae_config)


def autoencoder_feature_dim(vae_config: dict, raw_feature_dim: int) -> int:
    if not vae_config.get("enabled", False):
        return int(raw_feature_dim)
    if (
        vae_config.get("type", "mlp") == "evoformer"
        and vae_config.get("representation", "latent") == "pretrain_embedding"
    ):
        return int(vae_config.get("n_embed", 1280))
    return int(vae_config["latent_dim"])


def is_autoencoder_module(value: nn.Module | None) -> bool:
    return value is not None
