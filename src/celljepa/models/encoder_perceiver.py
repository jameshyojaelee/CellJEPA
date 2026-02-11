"""Perceiver encoder for gene tokens (GeneJEPA-inspired).

Cross-attention from a fixed set of latent tokens to variable-length
gene tokens. Provides fixed computational cost regardless of the number
of expressed genes — critical for full-transcriptome input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PerceiverEncoderConfig:
    token_dim: int = 256
    n_latent_tokens: int = 128
    n_cross_attn_layers: int = 2
    n_self_attn_layers: int = 4
    n_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to key-value pairs."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention.

        Args:
            query: (batch, n_q, dim) query tokens.
            kv: (batch, n_kv, dim) key-value tokens.
            kv_mask: (batch, n_kv) bool, True = attend-to. None = all valid.

        Returns:
            (batch, n_q, dim) attended output.
        """
        B, Nq, D = query.shape
        _, Nkv, _ = kv.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(query).view(B, Nq, H, Dh).transpose(1, 2)  # (B, H, Nq, Dh)
        k = self.k_proj(kv).view(B, Nkv, H, Dh).transpose(1, 2)    # (B, H, Nkv, Dh)
        v = self.v_proj(kv).view(B, Nkv, H, Dh).transpose(1, 2)    # (B, H, Nkv, Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nkv)

        if kv_mask is not None:
            # kv_mask: (B, Nkv) -> (B, 1, 1, Nkv)
            attn = attn.masked_fill(~kv_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return self.out_proj(out)


class PerceiverBlock(nn.Module):
    """One Perceiver layer: cross-attention + self-attention + FFN."""

    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = CrossAttention(dim, n_heads, dropout)
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.self_norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perceiver block.

        Args:
            latents: (batch, n_latent, dim) latent tokens.
            context: (batch, n_context, dim) gene tokens as KV.
            context_mask: (batch, n_context) bool, True = attend-to.

        Returns:
            (batch, n_latent, dim) updated latent tokens.
        """
        # Cross-attend from latents to gene context
        q_norm = self.cross_norm_q(latents)
        kv_norm = self.cross_norm_kv(context)
        latents = latents + self.cross_attn(q_norm, kv_norm, context_mask)

        # Self-attend among latent tokens
        x_norm = self.self_norm(latents)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        latents = latents + sa_out

        # FFN
        latents = latents + self.ffn(self.ffn_norm(latents))

        return latents


class PerceiverGeneEncoder(nn.Module):
    """Perceiver-based encoder for gene tokens.

    A fixed set of learned latent tokens cross-attend to variable-length
    gene tokens, then process via self-attention. This produces a fixed-size
    representation regardless of input gene count.
    """

    def __init__(self, cfg: PerceiverEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable latent tokens
        self.latent_tokens = nn.Parameter(
            torch.randn(1, cfg.n_latent_tokens, cfg.token_dim) * 0.02
        )

        # Perceiver blocks (each has cross-attn + self-attn)
        self.blocks = nn.ModuleList([
            PerceiverBlock(
                cfg.token_dim, cfg.n_heads, cfg.ff_dim, cfg.dropout
            )
            for _ in range(cfg.n_cross_attn_layers)
        ])

        # Additional self-attention-only layers for latent processing
        self_attn_layers = []
        for _ in range(cfg.n_self_attn_layers):
            self_attn_layers.append(nn.TransformerEncoderLayer(
                d_model=cfg.token_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ))
        self.self_attn_stack = nn.ModuleList(self_attn_layers)

        self.final_norm = nn.LayerNorm(cfg.token_dim)
        self.embed_dim = cfg.token_dim

    def forward(
        self,
        gene_tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict:
        """Encode gene tokens via Perceiver.

        Args:
            gene_tokens: (batch, n_genes, token_dim) from GeneTokenizer.
            mask: (batch, n_genes) bool, True = visible gene.

        Returns:
            dict with "latent_embeddings" and "cell_embedding".
        """
        batch_size = gene_tokens.shape[0]

        # Expand latent tokens for batch
        latents = self.latent_tokens.expand(batch_size, -1, -1)

        # Cross-attend from latents to gene tokens
        for block in self.blocks:
            latents = block(latents, gene_tokens, context_mask=mask)

        # Self-attend among latent tokens
        for layer in self.self_attn_stack:
            latents = layer(latents)

        latents = self.final_norm(latents)

        # Cell-level readout: mean pool over latent tokens
        cell_embedding = latents.mean(dim=1)  # (batch, token_dim)

        return {
            "latent_embeddings": latents,  # (batch, n_latent, token_dim)
            "cell_embedding": cell_embedding,  # (batch, token_dim)
        }
