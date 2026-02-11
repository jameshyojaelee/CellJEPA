"""Protein (ADT / CITE-seq) tokenizer for multi-modal CellJEPA (P6).

Converts surface protein abundance counts into per-protein tokens
with the same shape contract as ``GeneTokenizer``:
output ``(batch, n_proteins, token_dim)``.

Protein tokens consist of:
  - Protein identity embedding (learned per surface protein)
  - Fourier-encoded log-normalized abundance

Usage::

    tokenizer = ProteinTokenizer(ProteinTokenizerConfig(n_proteins=200))
    # x: (batch, n_proteins) ADT counts
    # protein_ids: (n_proteins,) integer indices
    tokens = tokenizer(x, protein_ids)  # (batch, n_proteins, token_dim)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from celljepa.models.gene_tokenizer import FourierExpressionEncoder


@dataclass
class ProteinTokenizerConfig:
    """Configuration for protein tokenization."""

    n_proteins: int = 200
    protein_embed_dim: int = 128
    n_fourier_features: int = 64
    token_dim: int = 256
    fourier_scale: float = 10.0
    learnable_fourier: bool = True
    dropout: float = 0.1
    log_normalize: bool = True  # log1p-normalize ADT counts


class ProteinIdentityEmbedding(nn.Module):
    """Learned embedding per surface protein."""

    def __init__(self, n_proteins: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(n_proteins + 1, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].zero_()

    def forward(self, protein_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(protein_ids)


class ProteinTokenizer(nn.Module):
    """Tokenize CITE-seq protein abundance into per-protein tokens.

    Same shape contract as ``GeneTokenizer``:
    input (batch, n_proteins) → output (batch, n_proteins, token_dim).
    """

    def __init__(self, cfg: ProteinTokenizerConfig):
        super().__init__()
        self.cfg = cfg

        self.protein_embedding = ProteinIdentityEmbedding(
            n_proteins=cfg.n_proteins,
            embed_dim=cfg.protein_embed_dim,
        )
        self.abundance_encoder = FourierExpressionEncoder(
            n_features=cfg.n_fourier_features,
            scale=cfg.fourier_scale,
            learnable=cfg.learnable_fourier,
        )

        raw_dim = cfg.protein_embed_dim + cfg.n_fourier_features
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, cfg.token_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.token_dim),
            nn.Dropout(cfg.dropout),
        )

        self.token_dim = cfg.token_dim

    def forward(
        self,
        abundance: torch.Tensor,
        protein_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenize protein abundance into per-protein tokens.

        Args:
            abundance: (batch, n_proteins) ADT count matrix.
            protein_ids: (n_proteins,) integer protein indices.

        Returns:
            (batch, n_proteins, token_dim) tensor of protein tokens.
        """
        batch_size, n_proteins = abundance.shape

        # Optional log normalization
        x = abundance
        if self.cfg.log_normalize:
            x = torch.log1p(x)

        protein_emb = self.protein_embedding(protein_ids)  # (n_proteins, embed_dim)
        protein_emb = protein_emb.unsqueeze(0).expand(batch_size, -1, -1)

        abundance_features = self.abundance_encoder(x)  # (batch, n_proteins, n_fourier)

        combined = torch.cat([protein_emb, abundance_features], dim=-1)
        tokens = self.projection(combined)  # (batch, n_proteins, token_dim)

        return tokens
