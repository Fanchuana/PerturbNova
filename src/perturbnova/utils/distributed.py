from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed(backend: str | None = None) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        resolved_backend = backend or "nccl"
    else:
        device = torch.device("cpu")
        resolved_backend = backend or "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=resolved_backend, init_method="env://")

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=resolved_backend,
    )


def barrier(context: DistributedContext) -> None:
    if context.is_distributed and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)
