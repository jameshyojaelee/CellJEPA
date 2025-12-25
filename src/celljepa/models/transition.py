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

