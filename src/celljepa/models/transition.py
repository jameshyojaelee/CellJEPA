"""Transition predictors for M3."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class TransitionConfig:
    embed_dim: int
    perturbation_vocab: int
    hidden_dim: int = 512
    unk_index: int = 0
    residual: bool = False
    alpha: float = 1.0
    max_delta_norm: float | None = None


class PerturbationEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, unk_index: int = 0):
        super().__init__()
        self.unk_index = unk_index
        self.emb = nn.Embedding(vocab_size, embed_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


class PrototypePredictor(nn.Module):
    def __init__(self, cfg: TransitionConfig):
        super().__init__()
        self.pert_emb = PerturbationEmbedding(cfg.perturbation_vocab, cfg.embed_dim, cfg.unk_index)
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(self, control_proto: torch.Tensor, perturbation_idx: torch.Tensor) -> torch.Tensor:
        p = self.pert_emb(perturbation_idx)
        x = torch.cat([control_proto, p], dim=1)
        return self.net(x)


class SetPredictor(nn.Module):
    def __init__(self, cfg: TransitionConfig):
        super().__init__()
        self.pert_emb = PerturbationEmbedding(cfg.perturbation_vocab, cfg.embed_dim, cfg.unk_index)
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(self, control_emb: torch.Tensor, perturbation_idx: torch.Tensor) -> torch.Tensor:
        p = self.pert_emb(perturbation_idx)
        if p.ndim == 2 and control_emb.ndim == 2:
            p = p.expand(control_emb.shape[0], -1)
        x = torch.cat([control_emb, p], dim=1)
        return self.net(x)


def _clamp_by_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return torch.zeros_like(x)
    norms = torch.linalg.norm(x, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
    return x * scale


class ResidualSetPredictor(nn.Module):
    """Safe-by-construction set predictor: z_hat = z + alpha * clamp(Δ)."""

    def __init__(self, cfg: TransitionConfig):
        super().__init__()
        self.pert_emb = PerturbationEmbedding(cfg.perturbation_vocab, cfg.embed_dim, cfg.unk_index)
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )
        self.residual = cfg.residual
        self.alpha = float(cfg.alpha)
        self.max_delta_norm = cfg.max_delta_norm

    def forward(self, control_emb: torch.Tensor, perturbation_idx: torch.Tensor) -> torch.Tensor:
        p = self.pert_emb(perturbation_idx)
        if p.ndim == 2 and control_emb.ndim == 2:
            p = p.expand(control_emb.shape[0], -1)
        x = torch.cat([control_emb, p], dim=1)
        delta = self.net(x)
        if self.max_delta_norm is not None:
            delta = _clamp_by_norm(delta, float(self.max_delta_norm))
        if self.residual:
            return control_emb + (self.alpha * delta)
        return self.alpha * delta
