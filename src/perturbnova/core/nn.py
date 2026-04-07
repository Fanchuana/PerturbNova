from __future__ import annotations

import math

import torch as th
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def update_ema(target_params, source_params, rate: float = 0.99):
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source, alpha=1 - rate)


def zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def scale_module(module, scale: float):
    for parameter in module.parameters():
        parameter.detach().mul_(scale)
    return module


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels: int):
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / max(half, 1)
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag: bool):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
