from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .utils.checkpoint import extract_state_dict


class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        residual: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def load_scimilarity_state(self, filename: str, use_gpu: bool = False):
        if not use_gpu:
            checkpoint = torch.load(filename, map_location=torch.device("cpu"), weights_only=False)
        else:
            checkpoint = torch.load(filename, weights_only=False)
        state_dict = extract_state_dict(checkpoint)
        first_layer_keys = [
            "network.0.1.weight",
            "network.0.1.bias",
            "network.0.2.weight",
            "network.0.2.bias",
            "network.0.2.running_mean",
            "network.0.2.running_var",
            "network.0.2.num_batches_tracked",
            "network.0.3.weight",
        ]
        for key in first_layer_keys:
            state_dict.pop(key, None)
        self.load_state_dict(state_dict, strict=False)


class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        residual: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def load_scimilarity_state(self, filename: str, use_gpu: bool = False):
        if not use_gpu:
            checkpoint = torch.load(filename, map_location=torch.device("cpu"), weights_only=False)
        else:
            checkpoint = torch.load(filename, weights_only=False)
        state_dict = extract_state_dict(checkpoint)
        last_layer_keys = ["network.3.weight", "network.3.bias"]
        for key in last_layer_keys:
            state_dict.pop(key, None)
        self.load_state_dict(state_dict, strict=False)


class VAE(nn.Module):
    def __init__(
        self,
        num_genes: int,
        device: str = "cuda",
        seed: int = 0,
        loss_ae: str = "gauss",
        decoder_activation: str = "linear",
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.device_name = device
        self.seed = seed
        self.loss_ae = loss_ae
        self.best_score = -1e3
        self.patience_trials = 0
        self.hidden_dim = [1024, 1024, 1024]
        self.dropout = 0.0
        self.input_dropout = 0.0
        self.residual = False
        self.hparams = self.set_hparams_(hidden_dim)
        self.encoder = Encoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            residual=self.residual,
        )
        self.decoder = Decoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=list(reversed(self.hidden_dim)),
            dropout=self.dropout,
            residual=self.residual,
        )
        self.loss_autoencoder = nn.MSELoss(reduction="mean")
        self.iteration = 0

    def forward(self, genes, return_latent: bool = False, return_decoded: bool = False):
        if return_decoded:
            return nn.ReLU()(self.decoder(genes))
        latent_basal = self.encoder(genes)
        if return_latent:
            return latent_basal
        return self.decoder(latent_basal)

    def set_hparams_(self, hidden_dim: int):
        return {
            "dim": hidden_dim,
            "autoencoder_width": 5000,
            "autoencoder_depth": 3,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 0.01,
            "autoencoder_lr": 5e-4,
        }


@dataclass
class VAESpec:
    enabled: bool
    checkpoint_path: str
    latent_dim: int
    freeze: bool
    reconstruction_loss_weight: float
    decode_predictions: bool


def build_vae_module(vae_config: dict, input_dim: int, device: torch.device) -> VAE | None:
    if not vae_config.get("enabled", False):
        return None
    module = VAE(
        num_genes=input_dim,
        device=str(device),
        seed=int(vae_config.get("seed", 0)),
        loss_ae=vae_config.get("loss_ae", "mse"),
        decoder_activation=vae_config.get("decoder_activation", "ReLU"),
        hidden_dim=int(vae_config.get("latent_dim", 128)),
    )
    module.to(device)
    checkpoint_path = vae_config.get("checkpoint_path", "")
    if checkpoint_path:
        raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(raw_state)
        module.load_state_dict(state_dict, strict=True)
    elif vae_config.get("pretrained_state_dir", ""):
        state_dir = vae_config["pretrained_state_dir"]
        use_gpu = device.type == "cuda"
        module.encoder.load_scimilarity_state(f"{state_dir}/encoder.ckpt", use_gpu=use_gpu)
        module.decoder.load_scimilarity_state(f"{state_dir}/decoder.ckpt", use_gpu=use_gpu)
    if vae_config.get("freeze", True):
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval()
    else:
        module.train()
    return module


def encode_with_vae(vae: VAE, tensor: torch.Tensor) -> torch.Tensor:
    return vae(tensor, return_latent=True)


def decode_with_vae(vae: VAE, tensor: torch.Tensor) -> torch.Tensor:
    return vae(tensor, return_decoded=True)


def decode_array_with_vae(
    vae: VAE,
    values,
    device: torch.device,
    batch_size: int = 512,
):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            batch = torch.as_tensor(values[start:end], dtype=torch.float32, device=device)
            outputs.append(decode_with_vae(vae, batch).detach().cpu())
    return torch.cat(outputs, dim=0).numpy().astype("float32")
