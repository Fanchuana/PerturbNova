from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist

from .diffusion import LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from .nn import timestep_embedding
from .respace import SpacedDiffusion, space_timesteps

__all__ = [
    "LossAwareSampler",
    "LossSecondMomentResampler",
    "ScheduleSampler",
    "UniformSampler",
    "build_diffusion",
    "create_named_schedule_sampler",
    "timestep_embedding",
]


def build_diffusion(config: dict) -> SpacedDiffusion:
    betas = get_named_beta_schedule(config["noise_schedule"], config["steps"])
    if config["use_kl"]:
        loss_type = LossType.RESCALED_KL
    elif config["rescale_learned_sigmas"]:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    timestep_respacing = config["timestep_respacing"] or [config["steps"]]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(config["steps"], timestep_respacing),
        betas=betas,
        model_mean_type=(
            ModelMeanType.START_X if config["predict_xstart"] else ModelMeanType.EPSILON
        ),
        model_var_type=(
            ModelVarType.LEARNED_RANGE
            if config["learn_sigma"]
            else ModelVarType.FIXED_LARGE
        ),
        loss_type=loss_type,
        rescale_timesteps=config["rescale_timesteps"],
        use_encoder=True,
    )


def create_named_schedule_sampler(name: str, diffusion: SpacedDiffusion) -> "ScheduleSampler":
    if name == "uniform":
        return UniformSampler(diffusion)
    if name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    raise NotImplementedError(f"Unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self) -> np.ndarray:
        raise NotImplementedError

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights()
        probs = weights / np.sum(weights)
        indices_np = np.random.choice(len(probs), size=(batch_size,), p=probs)
        indices = torch.from_numpy(indices_np).long().to(device)
        sample_weights = torch.from_numpy(1.0 / (len(probs) * probs[indices_np])).float().to(device)
        return indices, sample_weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion: SpacedDiffusion) -> None:
        self._weights = np.ones([diffusion.num_timesteps], dtype=np.float64)

    def weights(self) -> np.ndarray:
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(
        self,
        local_timesteps: torch.Tensor,
        local_losses: torch.Tensor,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            self.update_with_all_losses(local_timesteps.tolist(), local_losses.tolist())
            return

        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_timesteps.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            torch.tensor([len(local_timesteps)], dtype=torch.int32, device=local_timesteps.device),
        )
        batch_sizes = [size.item() for size in batch_sizes]
        max_batch = max(batch_sizes)
        timestep_batches = [torch.zeros(max_batch, device=local_timesteps.device) for _ in batch_sizes]
        loss_batches = [torch.zeros(max_batch, device=local_losses.device) for _ in batch_sizes]
        dist.all_gather(timestep_batches, local_timesteps)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [int(x.item()) for batch, size in zip(timestep_batches, batch_sizes) for x in batch[:size]]
        losses = [float(x.item()) for batch, size in zip(loss_batches, batch_sizes) for x in batch[:size]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, timesteps: list[int], losses: list[float]) -> None:
        raise NotImplementedError


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion: SpacedDiffusion, history_per_term: int = 10, uniform_prob: float = 0.001) -> None:
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros((diffusion.num_timesteps, history_per_term), dtype=np.float64)
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int64)

    def weights(self) -> np.ndarray:
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, timesteps: list[int], losses: list[float]) -> None:
        for timestep, loss in zip(timesteps, losses):
            count = self._loss_counts[timestep]
            if count == self.history_per_term:
                self._loss_history[timestep, :-1] = self._loss_history[timestep, 1:]
                self._loss_history[timestep, -1] = loss
            else:
                self._loss_history[timestep, count] = loss
                self._loss_counts[timestep] += 1

    def _warmed_up(self) -> bool:
        return bool((self._loss_counts == self.history_per_term).all())
