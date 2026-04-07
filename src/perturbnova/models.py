from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import timestep_embedding

MODEL_REGISTRY: dict[str, Callable[[dict, dict], nn.Module]] = {}


def register_model(name: str):
    def decorator(builder: Callable[[dict, dict], nn.Module]):
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def build_model(model_config: dict, dataset_artifacts: dict) -> nn.Module:
    name = model_config["name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](model_config, dataset_artifacts)


class ControlSetAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, hidden: torch.Tensor, control_set: torch.Tensor | None) -> torch.Tensor:
        if control_set is None:
            return hidden
        query = hidden.unsqueeze(1)
        attended, _ = self.attention(query, control_set, control_set)
        return hidden + attended.squeeze(1)


class AdditiveConditionEncoder(nn.Module):
    def __init__(
        self,
        perturbation_count: int,
        cell_type_count: int,
        batch_count: int,
        hidden_dim: int,
        use_batch_embeddings: bool,
    ) -> None:
        super().__init__()
        self.perturb_len = perturbation_count
        self.use_batch_embeddings = use_batch_embeddings
        self.perturbation_embedding = nn.Embedding(perturbation_count + 1, hidden_dim)
        self.cell_type_embedding = nn.Embedding(cell_type_count, hidden_dim)
        self.batch_embedding = nn.Embedding(batch_count, hidden_dim) if use_batch_embeddings else None
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        with torch.no_grad():
            self.perturbation_embedding.weight[perturbation_count].zero_()

    def forward(
        self,
        perturbation: torch.Tensor,
        cell_type: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.perturbation_embedding(perturbation) + self.cell_type_embedding(cell_type)
        if self.batch_embedding is not None and batch is not None:
            hidden = hidden + self.batch_embedding(batch)
        return self.mlp(hidden)


class FiLMConditionEncoder(nn.Module):
    def __init__(
        self,
        perturbation_count: int,
        cell_type_count: int,
        batch_count: int,
        hidden_dim: int,
        use_batch_embeddings: bool,
    ) -> None:
        super().__init__()
        self.perturb_len = perturbation_count
        self.use_batch_embeddings = use_batch_embeddings
        self.perturbation_embedding = nn.Embedding(perturbation_count + 1, hidden_dim)
        self.cell_type_embedding = nn.Embedding(cell_type_count, hidden_dim)
        self.batch_embedding = nn.Embedding(batch_count, hidden_dim) if use_batch_embeddings else None
        self.cell_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.base_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.interaction_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.perturbation_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.delta_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        with torch.no_grad():
            self.perturbation_embedding.weight[perturbation_count].zero_()
            self.delta_generator[-1].weight.mul_(0.01)
            self.delta_generator[-1].bias.zero_()

    def forward(
        self,
        perturbation: torch.Tensor,
        cell_type: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        perturbation_hidden = self.perturbation_embedding(perturbation)
        context = self.cell_type_embedding(cell_type)
        if self.batch_embedding is not None and batch is not None:
            context = context + self.batch_embedding(batch)

        perturbation_features = self.perturbation_encoder(perturbation_hidden)
        context_features = self.cell_encoder(context)
        base_gamma, base_beta = torch.chunk(self.base_generator(context_features), 2, dim=-1)

        gate = self.interaction_gate(context_features)
        perturbation_effect = self.perturbation_projection(perturbation_features)
        delta_gamma, delta_beta = torch.chunk(
            self.delta_generator(perturbation_effect * gate),
            2,
            dim=-1,
        )
        return {
            "gamma": base_gamma + delta_gamma,
            "beta": base_beta + delta_beta,
        }


class AdditiveMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.time_projection = nn.Linear(time_embed_dim, hidden_dim)
        self.condition_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        time_embedding_: torch.Tensor | None,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.fc1(hidden)))
        if time_embedding_ is not None:
            hidden = hidden + self.time_projection(time_embedding_)
        if condition is not None:
            hidden = hidden + self.condition_projection(condition)
        hidden = self.dropout(hidden)
        hidden = F.silu(self.norm2(self.fc2(hidden)))
        return hidden


class FiLMMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_embed_dim: int, dropout: float, control_heads: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.time_projection = nn.Linear(time_embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.control_attention = ControlSetAttention(hidden_dim, control_heads)

    def forward(
        self,
        hidden: torch.Tensor,
        time_embedding_: torch.Tensor | None,
        film_parameters: tuple[torch.Tensor, torch.Tensor] | None,
        control_set: torch.Tensor | None,
    ) -> torch.Tensor:
        gamma: torch.Tensor
        beta: torch.Tensor
        if film_parameters is None:
            gamma = torch.zeros_like(hidden)
            beta = torch.zeros_like(hidden)
        else:
            gamma, beta = film_parameters
        hidden = self.fc1(hidden)
        hidden = self.norm1(hidden)
        hidden = hidden * (1 + gamma) + beta
        hidden = F.silu(hidden)
        if time_embedding_ is not None:
            hidden = hidden + self.time_projection(time_embedding_)
        hidden = self.control_attention(hidden, control_set)
        hidden = self.dropout(hidden)
        hidden = F.silu(self.norm2(self.fc2(hidden)))
        return hidden


class BaseConditionedMLP(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_layer = nn.Linear(feature_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, feature_dim)
        self.num_layers = num_layers

    def _time_embedding(self, timesteps: torch.Tensor | None) -> torch.Tensor | None:
        if timesteps is None:
            return None
        return self.time_embed(timestep_embedding(timesteps, self.hidden_dim))


class AdditiveConditionedMLP(BaseConditionedMLP):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_embed_dim: int,
        dropout: float,
        perturbation_count: int,
        cell_type_count: int,
        batch_count: int,
        use_batch_embeddings: bool,
    ) -> None:
        super().__init__(feature_dim, hidden_dim, num_layers, time_embed_dim)
        self.encoder = AdditiveConditionEncoder(
            perturbation_count=perturbation_count,
            cell_type_count=cell_type_count,
            batch_count=batch_count,
            hidden_dim=hidden_dim,
            use_batch_embeddings=use_batch_embeddings,
        )
        self.blocks = nn.ModuleList(
            [AdditiveMLPBlock(hidden_dim, time_embed_dim, dropout) for _ in range(num_layers)]
        )
        self.null_perturbation_index = self.encoder.perturb_len

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        perturbation: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        cell_type: torch.Tensor | None = None,
        z_mod: torch.Tensor | None = None,
        **_: dict,
    ) -> torch.Tensor:
        time_embedding_ = self._time_embedding(timesteps)
        condition = z_mod
        if condition is None and perturbation is not None and cell_type is not None:
            condition = self.encoder(perturbation=perturbation, cell_type=cell_type, batch=batch)
        hidden = self.input_layer(x)
        for block in self.blocks:
            hidden = block(hidden, time_embedding_, condition)
        return self.output_layer(hidden)


class FiLMConditionedMLP(BaseConditionedMLP):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_embed_dim: int,
        dropout: float,
        perturbation_count: int,
        cell_type_count: int,
        batch_count: int,
        use_batch_embeddings: bool,
        control_attention_heads: int,
    ) -> None:
        super().__init__(feature_dim, hidden_dim, num_layers, time_embed_dim)
        self.encoder = FiLMConditionEncoder(
            perturbation_count=perturbation_count,
            cell_type_count=cell_type_count,
            batch_count=batch_count,
            hidden_dim=hidden_dim,
            use_batch_embeddings=use_batch_embeddings,
        )
        self.blocks = nn.ModuleList(
            [
                FiLMMLPBlock(hidden_dim, time_embed_dim, dropout, control_attention_heads)
                for _ in range(num_layers)
            ]
        )
        self.layer_gamma = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.layer_beta = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.null_perturbation_index = self.encoder.perturb_len

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        perturbation: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        cell_type: torch.Tensor | None = None,
        control_set: torch.Tensor | None = None,
        z_mod: dict[str, torch.Tensor] | None = None,
        **_: dict,
    ) -> torch.Tensor:
        time_embedding_ = self._time_embedding(timesteps)
        film_parameters = z_mod
        if film_parameters is None and perturbation is not None and cell_type is not None:
            film_parameters = self.encoder(perturbation=perturbation, cell_type=cell_type, batch=batch)

        hidden = self.input_layer(x)
        projected_control = None
        if control_set is not None:
            batch_size, control_count, feature_dim = control_set.shape
            projected_control = self.input_layer(control_set.reshape(-1, feature_dim)).reshape(
                batch_size,
                control_count,
                self.hidden_dim,
            )

        per_layer_film: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        if film_parameters is None:
            per_layer_film = [None for _ in self.blocks]
        else:
            gamma = film_parameters["gamma"]
            beta = film_parameters["beta"]
            for gamma_projection, beta_projection in zip(self.layer_gamma, self.layer_beta):
                per_layer_film.append((gamma_projection(gamma), beta_projection(beta)))

        for block, layer_film in zip(self.blocks, per_layer_film):
            hidden = block(hidden, time_embedding_, layer_film, projected_control)
        return self.output_layer(hidden)


@register_model("additive_mlp")
def _build_additive(model_config: dict, dataset_artifacts: dict) -> nn.Module:
    return AdditiveConditionedMLP(
        feature_dim=dataset_artifacts["feature_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        time_embed_dim=model_config["time_embed_dim"],
        dropout=model_config["dropout"],
        perturbation_count=dataset_artifacts["condition_sizes"]["perturbation"],
        cell_type_count=dataset_artifacts["condition_sizes"]["cell_type"],
        batch_count=dataset_artifacts["condition_sizes"]["batch"],
        use_batch_embeddings=model_config["use_batch_embeddings"],
    )


@register_model("film_mlp")
def _build_film(model_config: dict, dataset_artifacts: dict) -> nn.Module:
    return FiLMConditionedMLP(
        feature_dim=dataset_artifacts["feature_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        time_embed_dim=model_config["time_embed_dim"],
        dropout=model_config["dropout"],
        perturbation_count=dataset_artifacts["condition_sizes"]["perturbation"],
        cell_type_count=dataset_artifacts["condition_sizes"]["cell_type"],
        batch_count=dataset_artifacts["condition_sizes"]["batch"],
        use_batch_embeddings=model_config["use_batch_embeddings"],
        control_attention_heads=model_config["control_attention_heads"],
    )
