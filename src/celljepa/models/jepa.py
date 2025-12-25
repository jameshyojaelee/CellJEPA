"""Minimal JEPA model for M2 (student/teacher + predictor)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class JepaConfig:
    input_dim: int
    embed_dim: int = 256
    hidden_dim: int = 512
    predictor_hidden: int = 512
    ema_decay: float = 0.99
    mask_ratio: float = 0.25
    variance_target: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 1.0


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z_ctx: torch.Tensor, mask_ratio: torch.Tensor) -> torch.Tensor:
        if mask_ratio.ndim == 1:
            mask_ratio = mask_ratio[:, None]
        x = torch.cat([z_ctx, mask_ratio], dim=1)
        return self.net(x)


class JEPA(nn.Module):
    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg
        self.student = MLPEncoder(cfg.input_dim, cfg.embed_dim, cfg.hidden_dim)
        self.teacher = MLPEncoder(cfg.input_dim, cfg.embed_dim, cfg.hidden_dim)
        self.predictor = Predictor(cfg.embed_dim, cfg.predictor_hidden)
        self._init_teacher()

    def _init_teacher(self) -> None:
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.copy_(s.data)
            t.requires_grad = False

    @torch.no_grad()
    def update_teacher(self) -> None:
        m = self.cfg.ema_decay
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # mask: 1 for target, 0 for context
        x_ctx = x * (1.0 - mask)
        x_tgt = x * mask

        z_ctx = self.student(x_ctx)
        with torch.no_grad():
            z_tgt = self.teacher(x_tgt)
        mask_ratio = mask.mean(dim=1)
        z_pred = self.predictor(z_ctx, mask_ratio)
        return z_pred, z_tgt, z_ctx


def variance_covariance_loss(z: torch.Tensor, target_var: float = 1.0) -> Dict[str, torch.Tensor]:
    """Simple variance/covariance regularizer (VICReg-style)."""
    z = z - z.mean(dim=0)
    var = z.var(dim=0) + 1e-4
    var_loss = torch.mean(torch.relu(target_var - var))

    cov = (z.T @ z) / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).mean()

    return {"var_loss": var_loss, "cov_loss": cov_loss}

