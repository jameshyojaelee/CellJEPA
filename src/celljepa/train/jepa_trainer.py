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
    mask_type: str = "random"
    module_mask_path: str | None = None


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
    mask_type: str = "random",
    module_indices: list[np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    model.train()
    losses = []
    var_losses = []
    cov_losses = []
    mse_losses = []
    if rng is None:
        rng = np.random.default_rng(train_cfg.seed)

    def build_module_mask(batch_size: int, num_genes: int) -> torch.Tensor:
        if not module_indices:
            raise ValueError("module_indices required for module masking.")
        mask = np.zeros((batch_size, num_genes), dtype=np.float32)
        target = max(1, int(cfg.mask_ratio * num_genes))
        for i in range(batch_size):
            masked = 0
            tries = 0
            while masked < target and tries < len(module_indices) * 5:
                module = module_indices[rng.integers(len(module_indices))]
                mask[i, module] = 1.0
                masked = int(mask[i].sum())
                tries += 1
            if masked == 0:
                idx = rng.choice(num_genes, size=target, replace=False)
                mask[i, idx] = 1.0
        return torch.from_numpy(mask)

    for step, x in enumerate(data_iter):
        if train_cfg.steps_per_epoch and step >= train_cfg.steps_per_epoch:
            break
        x = x.to(train_cfg.device)
        if mask_type == "random":
            mask = (torch.rand_like(x) < cfg.mask_ratio).float()
        elif mask_type == "module":
            mask = build_module_mask(x.shape[0], x.shape[1]).to(train_cfg.device)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
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
