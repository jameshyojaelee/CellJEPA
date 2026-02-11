"""Gene tokenization: shared foundation for all encoder backends.

Converts raw expression data into per-gene tokens of the form
(gene_identity_embedding, expression_fourier_features).

Usage::

    tokenizer = GeneTokenizer(GeneTokenizerConfig(
        n_genes=20000,
        gene_embed_dim=128,
        n_fourier_features=64,
    ))
    # x: (batch, n_genes) dense expression matrix
    # gene_ids: (n_genes,) integer indices for gene identity
    tokens = tokenizer(x, gene_ids)  # (batch, n_genes, token_dim)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


@dataclass
class GeneTokenizerConfig:
    """Configuration for gene tokenization."""

    n_genes: int = 20_000
    gene_embed_dim: int = 128
    n_fourier_features: int = 64
    token_dim: int = 256  # final token dimension
    fourier_scale: float = 10.0
    learnable_fourier: bool = True
    dropout: float = 0.1


class FourierExpressionEncoder(nn.Module):
    """Encode continuous expression values via Fourier features.

    Instead of binning expression values (which loses information),
    we project them through sinusoidal features at multiple frequencies,
    producing a smooth, continuous representation.

    Following GeneJEPA: expression value e -> [sin(w_1*e), cos(w_1*e), ..., sin(w_k*e), cos(w_k*e)]
    """

    def __init__(self, n_features: int = 64, scale: float = 10.0, learnable: bool = True):
        super().__init__()
        assert n_features % 2 == 0, "n_features must be even (sin + cos pairs)"
        n_freq = n_features // 2

        # Initialize frequencies log-linearly spaced
        freqs = torch.exp(torch.linspace(0.0, math.log(scale), n_freq))
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode expression values.

        Args:
            x: (...,) arbitrary shape of expression values.

        Returns:
            (..., n_features) Fourier features.
        """
        # x: (...) -> (..., 1) for broadcasting
        x_unsqueezed = x.unsqueeze(-1)  # (..., 1)
        # (..., n_freq) after broadcasting
        angles = x_unsqueezed * self.freqs  # broadcast: (..., 1) * (n_freq,)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (..., n_features)


class GeneIdentityEmbedding(nn.Module):
    """Learned embedding per gene (keyed by integer index).

    Maps ~20K gene IDs to d-dimensional vectors. These embeddings encode
    what each gene *is* (function, regulatory role, pathway membership)
    in a way learned end-to-end.

    The same embeddings can be reused as perturbation encodings for
    single-gene perturbations (Phase 2).
    """

    def __init__(self, n_genes: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(n_genes + 1, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim
        # Scale initialization to prevent vanishing at init
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].zero_()

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """Lookup gene identity embeddings.

        Args:
            gene_ids: (n,) or (batch, n) integer gene indices.

        Returns:
            (n, embed_dim) or (batch, n, embed_dim) embeddings.
        """
        return self.embedding(gene_ids)


class GeneTokenizer(nn.Module):
    """Combine gene identity embeddings and expression Fourier features
    into per-gene tokens.

    Each gene token is: projection(concat(gene_identity, fourier_features))

    This is the shared input layer for all encoder backends.
    """

    def __init__(self, cfg: GeneTokenizerConfig):
        super().__init__()
        self.cfg = cfg

        self.gene_embedding = GeneIdentityEmbedding(
            n_genes=cfg.n_genes,
            embed_dim=cfg.gene_embed_dim,
        )
        self.expression_encoder = FourierExpressionEncoder(
            n_features=cfg.n_fourier_features,
            scale=cfg.fourier_scale,
            learnable=cfg.learnable_fourier,
        )

        # Project concatenated (gene_id ‖ fourier_expr) -> token_dim
        raw_dim = cfg.gene_embed_dim + cfg.n_fourier_features
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, cfg.token_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.token_dim),
            nn.Dropout(cfg.dropout),
        )

        self.token_dim = cfg.token_dim

    def forward(
        self,
        expression: torch.Tensor,
        gene_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenize expression data into per-gene tokens.

        Args:
            expression: (batch, n_genes) expression matrix (log1p-normalized).
            gene_ids: (n_genes,) integer indices for gene identity lookup.
                      Shared across the batch (all cells use the same gene panel).

        Returns:
            (batch, n_genes, token_dim) tensor of gene tokens.
        """
        batch_size, n_genes = expression.shape

        # Gene identity: (n_genes,) -> (n_genes, gene_embed_dim)
        gene_emb = self.gene_embedding(gene_ids)  # (n_genes, gene_embed_dim)
        # Expand for batch: (1, n_genes, gene_embed_dim) -> (batch, n_genes, gene_embed_dim)
        gene_emb = gene_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Expression Fourier features: (batch, n_genes) -> (batch, n_genes, n_fourier_features)
        expr_features = self.expression_encoder(expression)

        # Concatenate and project
        combined = torch.cat([gene_emb, expr_features], dim=-1)  # (batch, n_genes, raw_dim)
        tokens = self.projection(combined)  # (batch, n_genes, token_dim)

        return tokens


class CLSToken(nn.Module):
    """Learnable [CLS] token prepended to gene token sequences.

    Used by the Transformer encoder to provide a cell-level readout.
    """

    def __init__(self, token_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.token_dim = token_dim

    def prepend(self, tokens: torch.Tensor) -> torch.Tensor:
        """Prepend [CLS] to token sequence.

        Args:
            tokens: (batch, seq_len, token_dim)

        Returns:
            (batch, seq_len + 1, token_dim) with [CLS] at position 0.
        """
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls, tokens], dim=1)
