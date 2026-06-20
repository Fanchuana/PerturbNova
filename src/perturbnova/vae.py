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
        normalize_latent: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        self.normalize_latent = normalize_latent
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
        if self.normalize_latent:
            x = F.normalize(x, p=2, dim=1)
        return x

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
        hidden_dims: List[int] | None = None,
        normalize_latent: bool = True,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.device_name = device
        self.seed = seed
        self.loss_ae = loss_ae
        self.best_score = -1e3
        self.patience_trials = 0
        self.hidden_dim = hidden_dims or [1024, 1024, 1024]
        self.dropout = 0.0
        self.input_dropout = 0.0
        self.residual = False
        self.normalize_latent = normalize_latent
        self.hparams = self.set_hparams_(hidden_dim)
        self.encoder = Encoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            residual=self.residual,
            normalize_latent=self.normalize_latent,
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


class ConditionEmbedding(nn.Module):
    def __init__(
        self,
        n_cell_types: int,
        n_perts: int,
        embed_dim: int = 64,
        use_cell_type: bool = True,
        use_perturbation: bool = True,
    ):
        super().__init__()
        self.use_cell_type = use_cell_type
        self.use_perturbation = use_perturbation
        self.cell_embedding = nn.Embedding(max(n_cell_types, 1), embed_dim) if use_cell_type else None
        self.pert_embedding = nn.Embedding(max(n_perts, 1), embed_dim) if use_perturbation else None
        self.output_dim = embed_dim * int(use_cell_type) + embed_dim * int(use_perturbation)

    def forward(self, cell_type_id: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        pieces = []
        if self.cell_embedding is not None:
            pieces.append(self.cell_embedding(cell_type_id.long()))
        if self.pert_embedding is not None:
            pieces.append(self.pert_embedding(pert_id.long()))
        if not pieces:
            return torch.empty((cell_type_id.shape[0], 0), device=cell_type_id.device)
        return torch.cat(pieces, dim=-1)


class ConditionalAutoencoder(nn.Module):
    def __init__(
        self,
        num_genes: int,
        n_cell_types: int,
        n_perts: int,
        latent_dim: int = 128,
        hidden_dims: List[int] | None = None,
        cond_embed_dim: int = 64,
        normalize_latent: bool = True,
        use_cell_type: bool = True,
        use_perturbation: bool = True,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [1024, 1024, 1024]
        self.condition = ConditionEmbedding(
            n_cell_types,
            n_perts,
            embed_dim=cond_embed_dim,
            use_cell_type=use_cell_type,
            use_perturbation=use_perturbation,
        )
        self.encoder = Encoder(
            num_genes + self.condition.output_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dims,
            dropout=0.0,
            input_dropout=0.0,
            normalize_latent=normalize_latent,
        )
        self.decoder = Decoder(
            num_genes,
            latent_dim=latent_dim + self.condition.output_dim,
            hidden_dim=list(reversed(hidden_dims)),
            dropout=0.0,
        )

    def _condition(self, cell_type_id: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        return self.condition(cell_type_id, pert_id)

    def encode(self, x: torch.Tensor, cell_type_id: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        cond = self._condition(cell_type_id, pert_id)
        return self.encoder(torch.cat([x, cond], dim=-1))

    def decode(self, z: torch.Tensor, cell_type_id: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        cond = self._condition(cell_type_id, pert_id)
        return self.decoder(torch.cat([z, cond], dim=-1))

    def forward(self, x: torch.Tensor, cell_type_id: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x, cell_type_id, pert_id), cell_type_id, pert_id)

    def compute_loss(
        self,
        x: torch.Tensor,
        cell_type_id: torch.Tensor,
        pert_id: torch.Tensor,
        **_: torch.Tensor,
    ):
        recon = self.forward(x, cell_type_id, pert_id)
        loss = F.mse_loss(recon, x)
        return loss, {"recon_loss": loss.detach(), "recon": recon}


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        num_genes: int,
        n_cell_types: int,
        n_perts: int,
        latent_dim: int = 128,
        hidden_dims: List[int] | None = None,
        cond_embed_dim: int = 64,
        normalize_latent: bool = False,
        use_cell_type: bool = True,
        use_perturbation: bool = True,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [1024, 1024, 1024]
        self.condition = ConditionEmbedding(
            n_cell_types,
            n_perts,
            embed_dim=cond_embed_dim,
            use_cell_type=use_cell_type,
            use_perturbation=use_perturbation,
        )
        self.encoder_backbone = Encoder(
            num_genes + self.condition.output_dim,
            latent_dim=hidden_dims[-1],
            hidden_dim=hidden_dims,
            dropout=0.0,
            input_dropout=0.0,
            normalize_latent=False,
        )
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.normalize_latent = normalize_latent
        self.decoder = Decoder(
            num_genes,
            latent_dim=latent_dim + self.condition.output_dim,
            hidden_dim=list(reversed(hidden_dims)),
            dropout=0.0,
        )

    def _condition(self, cell_type_id: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        return self.condition(cell_type_id, pert_id)

    def encode(self, x: torch.Tensor, cell_type_id: torch.Tensor, pert_id: torch.Tensor):
        cond = self._condition(cell_type_id, pert_id)
        h = self.encoder_backbone(torch.cat([x, cond], dim=-1))
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-12.0, max=12.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        if self.normalize_latent:
            z = F.normalize(z, p=2, dim=1)
        return z

    def decode(self, z: torch.Tensor, cell_type_id: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        cond = self._condition(cell_type_id, pert_id)
        return self.decoder(torch.cat([z, cond], dim=-1))

    def forward(self, x: torch.Tensor, cell_type_id: torch.Tensor, pert_id: torch.Tensor):
        mu, logvar = self.encode(x, cell_type_id, pert_id)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cell_type_id, pert_id)
        return recon, mu, logvar, z

    def compute_loss(
        self,
        x: torch.Tensor,
        cell_type_id: torch.Tensor,
        pert_id: torch.Tensor,
        beta: float = 1e-3,
        **_: torch.Tensor,
    ):
        recon, mu, logvar, _ = self.forward(x, cell_type_id, pert_id)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon_loss + beta * kl_loss
        return loss, {
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "beta": torch.as_tensor(beta, device=x.device),
            "recon": recon,
        }


class ConditionalDeltaVAE(ConditionalVAE):
    def __init__(self, *args, delta_loss_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_loss_weight = delta_loss_weight
        latent_dim = self.mu.out_features
        num_genes = self.decoder.network[-1].out_features
        self.delta_head = nn.Sequential(
            nn.Linear(latent_dim + self.condition.output_dim, latent_dim),
            nn.PReLU(),
            nn.Linear(latent_dim, num_genes),
        )

    def predict_delta(self, z: torch.Tensor, cell_type_id: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        cond = self._condition(cell_type_id, pert_id)
        return self.delta_head(torch.cat([z, cond], dim=-1))

    def compute_loss(
        self,
        x: torch.Tensor,
        cell_type_id: torch.Tensor,
        pert_id: torch.Tensor,
        target_delta: torch.Tensor | None = None,
        beta: float = 1e-3,
        delta_loss_weight: float | None = None,
    ):
        recon, mu, logvar, z = self.forward(x, cell_type_id, pert_id)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        if target_delta is None:
            delta_loss = torch.zeros((), device=x.device)
        else:
            delta_loss = F.mse_loss(self.predict_delta(z, cell_type_id, pert_id), target_delta)
        weight = self.delta_loss_weight if delta_loss_weight is None else delta_loss_weight
        loss = recon_loss + beta * kl_loss + weight * delta_loss
        return loss, {
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "delta_loss": delta_loss.detach(),
            "beta": torch.as_tensor(beta, device=x.device),
            "recon": recon,
        }


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.commitment_cost = commitment_cost

    def forward(self, z: torch.Tensor):
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        loss = codebook_loss + self.commitment_cost * commitment_loss
        z_q = z + (z_q - z).detach()
        return z_q, loss, indices


class ConditionalVQVAE(ConditionalAutoencoder):
    def __init__(
        self,
        *args,
        num_codes: int = 512,
        commitment_cost: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.quantizer = VectorQuantizer(num_codes, self.encoder.latent_dim, commitment_cost)

    def forward(self, x: torch.Tensor, cell_type_id: torch.Tensor, pert_id: torch.Tensor):
        z = self.encode(x, cell_type_id, pert_id)
        z_q, vq_loss, indices = self.quantizer(z)
        recon = self.decode(z_q, cell_type_id, pert_id)
        return recon, vq_loss, indices

    def compute_loss(
        self,
        x: torch.Tensor,
        cell_type_id: torch.Tensor,
        pert_id: torch.Tensor,
        **_: torch.Tensor,
    ):
        recon, vq_loss, indices = self.forward(x, cell_type_id, pert_id)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        return loss, {
            "recon_loss": recon_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "code_usage": indices.unique().numel(),
            "recon": recon,
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
    hidden_dims = vae_config.get("hidden_dims", [1024, 1024, 1024])
    if isinstance(hidden_dims, str):
        hidden_dims = [int(value) for value in hidden_dims.split(",") if value.strip()]
    module = VAE(
        num_genes=input_dim,
        device=str(device),
        seed=int(vae_config.get("seed", 0)),
        loss_ae=vae_config.get("loss_ae", "mse"),
        decoder_activation=vae_config.get("decoder_activation", "ReLU"),
        hidden_dim=int(vae_config.get("latent_dim", 128)),
        hidden_dims=hidden_dims,
        normalize_latent=bool(vae_config.get("normalize_latent", True)),
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
