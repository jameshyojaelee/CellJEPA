"""Gene-level prediction decoder (P3).

Maps post-perturbation latent embeddings to per-gene log-fold change
predictions. Lightweight (1–2 layer MLP), designed to be trained with
a frozen encoder (standard transfer learning protocol).

Usage::

    from celljepa.models.decoder import DecoderConfig, GeneLevelDecoder

    cfg = DecoderConfig(embed_dim=256, n_genes=5000)
    decoder = GeneLevelDecoder(cfg)
    # From cell embeddings:
    lfc_pred = decoder(z_pert=z_pert, z_ctrl=z_ctrl)  # (B, n_genes)
    # From per-gene deltas:
    lfc_pred = decoder.from_gene_deltas(gene_deltas)   # (B, n_genes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class DecoderConfig:
    """Configuration for the gene-level decoder."""

    embed_dim: int = 256
    n_genes: int = 5000
    hidden_dim: int = 512
    n_layers: int = 2  # 1 or 2
    dropout: float = 0.1
    use_difference: bool = True  # If True, decode from z_pert - z_ctrl


class GeneLevelDecoder(nn.Module):
    """Decode latent embeddings into per-gene log-fold change predictions.

    Two input modes:
    1. **Cell-level**: from post-perturbation and control cell embeddings,
       predict per-gene LFCs via ``forward(z_pert, z_ctrl)``.
    2. **Gene-level**: from per-gene deltas (from attention/graph world models),
       predict per-gene LFCs via ``from_gene_deltas(deltas)``.
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Cell-level decoder: embedding → per-gene LFCs
        layers = []
        in_dim = cfg.embed_dim
        if cfg.n_layers == 1:
            layers.append(nn.Linear(in_dim, cfg.n_genes))
        else:
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(cfg.dropout))
            for _ in range(cfg.n_layers - 2):
                layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(cfg.dropout))
            layers.append(nn.Linear(cfg.hidden_dim, cfg.n_genes))
        self.cell_decoder = nn.Sequential(*layers)

        # Gene-level decoder: per-gene embedding delta → scalar LFC
        self.gene_decoder = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

        self.n_genes = cfg.n_genes

    def forward(
        self,
        z_pert: torch.Tensor,
        z_ctrl: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-gene LFCs from cell-level embeddings.

        Args:
            z_pert: (B, D) predicted post-perturbation cell embedding.
            z_ctrl: (B, D) control cell embedding.

        Returns:
            (B, n_genes) predicted log-fold changes.
        """
        if self.cfg.use_difference:
            x = z_pert - z_ctrl
        else:
            x = z_pert
        return self.cell_decoder(x)  # (B, n_genes)

    def from_gene_deltas(
        self,
        gene_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-gene LFCs from per-gene embedding deltas.

        Args:
            gene_deltas: (B, G, D) per-gene embedding deltas from
                         attention/graph world models.

        Returns:
            (B, G) predicted log-fold changes per gene.
        """
        return self.gene_decoder(gene_deltas).squeeze(-1)  # (B, G)
