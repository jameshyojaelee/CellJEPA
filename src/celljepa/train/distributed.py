"""Distributed training utilities for multi-GPU pretraining (P4).

Provides DDP setup/teardown, distributed sampler creation, and a
configuration dataclass for distributed training. Designed to work
with SLURM environments (auto-detects rank/world_size from env vars).

Usage::

    from celljepa.train.distributed import (
        DistributedTrainConfig, setup_distributed, cleanup_distributed,
        get_ddp_model, get_distributed_sampler, is_main_process,
    )

    cfg = DistributedTrainConfig()
    setup_distributed()
    model = get_ddp_model(model, cfg.local_rank)
    sampler = get_distributed_sampler(dataset)
    # ... training loop ...
    cleanup_distributed()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass
class DistributedTrainConfig:
    """Configuration for distributed training."""

    # DDP settings (auto-detected from env if not specified)
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"

    # Training
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    seed: int = 42

    # Checkpointing
    save_every_n_epochs: int = 1
    resume_from: Optional[str] = None

    # Encoder
    encoder_type: str = "transformer"
    mask_type: str = "random"
    mask_ratio: float = 0.25

    @classmethod
    def from_env(cls, **overrides) -> "DistributedTrainConfig":
        """Create config auto-detecting DDP env vars from SLURM/torchrun."""
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return cls(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            **overrides,
        )


def setup_distributed(backend: str = "nccl") -> None:
    """Initialize the distributed process group.

    Auto-detects SLURM/torchrun environment variables.
    No-op if WORLD_SIZE is not set or equals 1.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process() -> bool:
    """Return True if this is the main (rank 0) process."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def get_ddp_model(
    model: nn.Module,
    local_rank: int = 0,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap a model in DistributedDataParallel.

    No-op if distributed is not initialized (single GPU).
    """
    if not torch.distributed.is_initialized():
        return model

    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )


def get_distributed_sampler(
    dataset: Dataset,
    shuffle: bool = True,
    seed: int = 42,
) -> Optional[torch.utils.data.distributed.DistributedSampler]:
    """Create a DistributedSampler if distributed is active.

    Returns None for single-GPU training.
    """
    if not torch.distributed.is_initialized():
        return None

    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=shuffle,
        seed=seed,
    )


def effective_batch_size(
    per_device_batch: int,
    gradient_accumulation: int = 1,
    world_size: int = 1,
) -> int:
    """Compute effective batch size accounting for DDP and gradient accumulation."""
    return per_device_batch * gradient_accumulation * world_size


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay schedule."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
