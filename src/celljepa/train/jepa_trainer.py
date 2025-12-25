"""Training utilities for JEPA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn

from celljepa.models.jepa import JEPA, JepaConfig, variance_covariance_loss


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    steps_per_epoch: int | None = None
    device: str = "cpu"
    seed: int = 0
    fast_dev: bool = False


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: JEPA,
    optimizer: torch.optim.Optimizer,
    data_iter: Iterable[torch.Tensor],
    cfg: JepaConfig,
    train_cfg: TrainConfig,
) -> Dict[str, float]:
    model.train()
    losses = []
    var_losses = []
    cov_losses = []
    mse_losses = []

    for step, x in enumerate(data_iter):
        if train_cfg.steps_per_epoch and step >= train_cfg.steps_per_epoch:
            break
        x = x.to(train_cfg.device)
        mask = (torch.rand_like(x) < cfg.mask_ratio).float()
        z_pred, z_tgt, z_ctx = model(x, mask)

        mse = nn.functional.mse_loss(z_pred, z_tgt)
        reg = variance_covariance_loss(z_ctx, target_var=cfg.variance_target)
        loss = mse + cfg.variance_weight * reg["var_loss"] + cfg.covariance_weight * reg["cov_loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_teacher()

        losses.append(loss.item())
        mse_losses.append(mse.item())
        var_losses.append(reg["var_loss"].item())
        cov_losses.append(reg["cov_loss"].item())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mse": float(np.mean(mse_losses)) if mse_losses else float("nan"),
        "var": float(np.mean(var_losses)) if var_losses else float("nan"),
        "cov": float(np.mean(cov_losses)) if cov_losses else float("nan"),
    }

