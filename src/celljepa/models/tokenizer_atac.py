"""ATAC-seq peak tokenizer for multi-modal CellJEPA (P6).

Converts peak accessibility vectors into per-peak tokens with the same
shape contract as ``GeneTokenizer``: output ``(batch, n_peaks, token_dim)``.

Peak tokens consist of:
  - Peak identity embedding (learned per peak region)
  - Fourier-encoded accessibility value

Usage::

    tokenizer = ATACPeakTokenizer(ATACPeakTokenizerConfig(n_peaks=50000))
    # x: (batch, n_peaks) accessibility matrix
    # peak_ids: (n_peaks,) integer indices
    tokens = tokenizer(x, peak_ids)  # (batch, n_peaks, token_dim)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from celljepa.models.gene_tokenizer import FourierExpressionEncoder


@dataclass
class ATACPeakTokenizerConfig:
    """Configuration for ATAC peak tokenization."""

    n_peaks: int = 50_000
    peak_embed_dim: int = 128
    n_fourier_features: int = 64
    token_dim: int = 256
    fourier_scale: float = 10.0
    learnable_fourier: bool = True
    dropout: float = 0.1


class PeakIdentityEmbedding(nn.Module):
    """Learned embedding per chromatin peak region.

    Analogous to ``GeneIdentityEmbedding``: each peak region gets a unique
    learned vector encoding its genomic context (promoter, enhancer, etc.).
    """

    def __init__(self, n_peaks: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(n_peaks + 1, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].zero_()

    def forward(self, peak_ids: torch.Tensor) -> torch.Tensor:
        """Lookup peak identity embeddings.

        Args:
            peak_ids: (n,) or (batch, n) integer peak indices.

        Returns:
            (n, embed_dim) or (batch, n, embed_dim) embeddings.
        """
        return self.embedding(peak_ids)


class ATACPeakTokenizer(nn.Module):
    """Tokenize ATAC-seq accessibility into per-peak tokens.

    Same shape contract as ``GeneTokenizer``:
    input (batch, n_peaks) → output (batch, n_peaks, token_dim).
    """

    def __init__(self, cfg: ATACPeakTokenizerConfig):
        super().__init__()
        self.cfg = cfg

        self.peak_embedding = PeakIdentityEmbedding(
            n_peaks=cfg.n_peaks,
            embed_dim=cfg.peak_embed_dim,
        )
        self.accessibility_encoder = FourierExpressionEncoder(
            n_features=cfg.n_fourier_features,
            scale=cfg.fourier_scale,
            learnable=cfg.learnable_fourier,
        )

        raw_dim = cfg.peak_embed_dim + cfg.n_fourier_features
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, cfg.token_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.token_dim),
            nn.Dropout(cfg.dropout),
        )

        self.token_dim = cfg.token_dim

    def forward(
        self,
        accessibility: torch.Tensor,
        peak_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenize accessibility data into per-peak tokens.

        Args:
            accessibility: (batch, n_peaks) accessibility matrix.
            peak_ids: (n_peaks,) integer peak region indices.

        Returns:
            (batch, n_peaks, token_dim) tensor of peak tokens.
        """
        batch_size, n_peaks = accessibility.shape

        peak_emb = self.peak_embedding(peak_ids)  # (n_peaks, embed_dim)
        peak_emb = peak_emb.unsqueeze(0).expand(batch_size, -1, -1)

        acc_features = self.accessibility_encoder(accessibility)  # (batch, n_peaks, n_fourier)

        combined = torch.cat([peak_emb, acc_features], dim=-1)
        tokens = self.projection(combined)  # (batch, n_peaks, token_dim)

        return tokens
