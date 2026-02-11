"""Transformer encoder operating on gene tokens (scGPT-inspired).

Self-attention across the gene dimension: each gene token attends to every
other gene token, naturally capturing gene-gene regulatory relationships.

No positional encoding — genes have no spatial order. Gene identity
is provided by the GeneIdentityEmbedding in the tokenizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerEncoderConfig:
    token_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    use_cls_token: bool = True


class TransformerGeneEncoder(nn.Module):
    """Gene-token Transformer encoder.

    Each gene is a token; self-attention learns gene-gene interactions.
    Cell-level representation obtained via [CLS] token pooling or mean pooling.
    """

    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.cfg = cfg

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
        )
        self.norm = nn.LayerNorm(cfg.token_dim)

        # CLS token for cell-level readout
        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.token_dim) * 0.02)
        else:
            self.cls_token = None

        self.embed_dim = cfg.token_dim

    def forward(
        self,
        gene_tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict:
        """Encode gene tokens.

        Args:
            gene_tokens: (batch, n_genes, token_dim) from the GeneTokenizer.
            mask: (batch, n_genes) bool tensor. True = this gene is available
                  (visible). If None, all genes are visible.

        Returns:
            dict with:
                - "gene_embeddings": (batch, n_genes, token_dim) per-gene output
                - "cell_embedding": (batch, token_dim) cell-level representation
        """
        batch_size, n_genes, _ = gene_tokens.shape
        x = gene_tokens

        # Prepend CLS token if configured
        if self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (batch, 1 + n_genes, token_dim)

        # Build attention mask: mask out non-visible genes
        src_key_padding_mask = None
        if mask is not None:
            if self.cls_token is not None:
                # CLS is always visible
                cls_visible = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
                full_mask = torch.cat([cls_visible, mask], dim=1)
            else:
                full_mask = mask
            # TransformerEncoder expects True = IGNORE (inverted from our convention)
            src_key_padding_mask = ~full_mask

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        # Split CLS from gene tokens
        if self.cls_token is not None:
            cell_embedding = x[:, 0]  # (batch, token_dim)
            gene_embeddings = x[:, 1:]  # (batch, n_genes, token_dim)
        else:
            gene_embeddings = x
            # Mean pool over visible genes
            if mask is not None:
                mask_float = mask.unsqueeze(-1).float()  # (batch, n_genes, 1)
                cell_embedding = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
            else:
                cell_embedding = x.mean(dim=1)

        return {
            "gene_embeddings": gene_embeddings,
            "cell_embedding": cell_embedding,
        }
