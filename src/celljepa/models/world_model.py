"""Action-conditioned set-to-set world model for M3C."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class WorldModelConfig:
    embed_dim: int
    action_vocab: int
    hidden_dim: int = 512
    context_dim: int | None = None
    residual: bool = True
    action_dim: int | None = None


class ActionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


class WorldModel(nn.Module):
    """Minimal set-to-set predictor conditioned on action embeddings."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        context_dim = cfg.context_dim or cfg.embed_dim
        self.cfg = cfg
        action_dim = cfg.action_dim or cfg.embed_dim
        self.action_emb = ActionEmbedding(cfg.action_vocab, action_dim)
        self.context_net = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, context_dim),
        )
        self.cell_net = nn.Sequential(
            nn.Linear(cfg.embed_dim + context_dim + action_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(self, control_emb: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """Predict perturbed embeddings for a control set.

        Args:
            control_emb: (n, d) tensor of control embeddings.
            action_idx: (1,) tensor with action index.
        """
        context = control_emb.mean(dim=0, keepdim=True)
        context = self.context_net(context)
        action = self.action_emb(action_idx)
        context = context.expand(control_emb.shape[0], -1)
        action = action.expand(control_emb.shape[0], -1)
        x = torch.cat([control_emb, context, action], dim=1)
        delta = self.cell_net(x)
        if self.cfg.residual:
            return control_emb + delta
        return delta
