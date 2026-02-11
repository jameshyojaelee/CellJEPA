"""Multi-modal fusion encoders for CellJEPA (P6).

Three fusion strategies for combining gene, ATAC, and protein tokens:

1. **EarlyFusionEncoder**: concatenate all modality tokens, then single Transformer.
2. **CrossModalEncoder**: per-modality Transformer stacks + cross-attention layers.
3. **LateFusionEncoder**: independent per-modality encoders, fuse cell embeddings.

All produce a unified output dict with ``cell_embedding`` and per-modality
``gene_embeddings`` / ``peak_embeddings`` / ``protein_embeddings``.

Usage::

    from celljepa.models.encoder_multimodal import (
        MultiModalEncoderConfig, build_multimodal_encoder,
    )
    cfg = MultiModalEncoderConfig(fusion="cross_modal")
    encoder = build_multimodal_encoder(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn


@dataclass
class MultiModalEncoderConfig:
    """Configuration for multi-modal encoder."""

    token_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    # Fusion strategy: "early", "cross_modal", "late"
    fusion: str = "cross_modal"

    # Which modalities are active
    modalities: List[str] = field(default_factory=lambda: ["rna", "atac"])

    # Cross-modal specific
    n_cross_attn_layers: int = 2

    # Late fusion specific
    late_fusion_method: str = "attention"  # "attention" or "mlp"


# ---------------------------------------------------------------------------
# Modality tag embeddings
# ---------------------------------------------------------------------------

class ModalityEmbedding(nn.Module):
    """Learned modality identifier added to tokens."""

    def __init__(self, n_modalities: int, token_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_modalities, token_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, modality_idx: int) -> torch.Tensor:
        """Add modality embedding to tokens.

        Args:
            tokens: (batch, n_tokens, token_dim)
            modality_idx: integer modality index
        """
        mod_emb = self.embedding(
            torch.tensor(modality_idx, device=tokens.device)
        )  # (token_dim,)
        return tokens + mod_emb.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Cross-attention layer
# ---------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    """Bidirectional cross-attention between two modality token sequences."""

    def __init__(self, token_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # A attends to B
        self.cross_attn_a2b = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_a = nn.LayerNorm(token_dim)
        self.ff_a = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim),
            nn.Dropout(dropout),
        )
        self.norm_a_ff = nn.LayerNorm(token_dim)

        # B attends to A
        self.cross_attn_b2a = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_b = nn.LayerNorm(token_dim)
        self.ff_b = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim),
            nn.Dropout(dropout),
        )
        self.norm_b_ff = nn.LayerNorm(token_dim)

    def forward(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
    ) -> tuple:
        """Apply bidirectional cross-attention.

        Args:
            tokens_a: (batch, n_a, dim)
            tokens_b: (batch, n_b, dim)

        Returns:
            (updated_a, updated_b)
        """
        # A queries, B provides keys/values
        a_cross, _ = self.cross_attn_a2b(
            query=self.norm_a(tokens_a),
            key=tokens_b,
            value=tokens_b,
        )
        tokens_a = tokens_a + a_cross
        tokens_a = tokens_a + self.ff_a(self.norm_a_ff(tokens_a))

        # B queries, A provides keys/values
        b_cross, _ = self.cross_attn_b2a(
            query=self.norm_b(tokens_b),
            key=tokens_a,
            value=tokens_a,
        )
        tokens_b = tokens_b + b_cross
        tokens_b = tokens_b + self.ff_b(self.norm_b_ff(tokens_b))

        return tokens_a, tokens_b


# ---------------------------------------------------------------------------
# 1. Early Fusion
# ---------------------------------------------------------------------------

class EarlyFusionEncoder(nn.Module):
    """Concatenate all modality tokens, then encode with shared Transformer.

    Simple and parameter-efficient. Cross-modal interactions emerge naturally
    through self-attention over the concatenated sequence.
    """

    def __init__(self, cfg: MultiModalEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.modality_embedding = ModalityEmbedding(
            n_modalities=len(cfg.modalities), token_dim=cfg.token_dim,
        )

        # Shared Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.token_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.token_dim) * 0.02)
        self.embed_dim = cfg.token_dim

    def forward(self, modality_tokens: Dict[str, torch.Tensor]) -> dict:
        """Encode concatenated multi-modal tokens.

        Args:
            modality_tokens: dict mapping modality name → (batch, n_tokens, token_dim)

        Returns:
            dict with "cell_embedding" and per-modality embeddings.
        """
        batch_size = next(iter(modality_tokens.values())).shape[0]
        all_tokens = []
        boundaries = {}
        offset = 1  # Account for CLS

        for i, (mod_name, tokens) in enumerate(modality_tokens.items()):
            tagged = self.modality_embedding(tokens, i)
            n = tagged.shape[1]
            boundaries[mod_name] = (offset, offset + n)
            offset += n
            all_tokens.append(tagged)

        # Prepend CLS
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls] + all_tokens, dim=1)

        x = self.encoder(x)
        x = self.norm(x)

        result = {"cell_embedding": x[:, 0]}  # CLS output

        for mod_name, (start, end) in boundaries.items():
            result[f"{mod_name}_embeddings"] = x[:, start:end]

        return result


# ---------------------------------------------------------------------------
# 2. Cross-Modal Encoder
# ---------------------------------------------------------------------------

class CrossModalEncoder(nn.Module):
    """Separate per-modality Transformers + cross-attention layers.

    Each modality has its own self-attention stack, with periodic cross-attention
    layers that allow information flow between modalities.
    """

    def __init__(self, cfg: MultiModalEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.modality_embedding = ModalityEmbedding(
            n_modalities=len(cfg.modalities), token_dim=cfg.token_dim,
        )

        # Per-modality self-attention stacks
        self.modality_encoders = nn.ModuleDict()
        for mod_name in cfg.modalities:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.token_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.modality_encoders[mod_name] = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.n_layers,
            )

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(cfg.token_dim, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_cross_attn_layers)
        ])

        self.norm = nn.LayerNorm(cfg.token_dim)
        self.fusion_proj = nn.Linear(cfg.token_dim * len(cfg.modalities), cfg.token_dim)
        self.embed_dim = cfg.token_dim

    def forward(self, modality_tokens: Dict[str, torch.Tensor]) -> dict:
        """Encode with per-modality stacks + cross-attention.

        Args:
            modality_tokens: dict mapping modality name → (batch, n_tokens, token_dim)
        """
        # Tag with modality embeddings
        encoded = {}
        for i, (mod_name, tokens) in enumerate(modality_tokens.items()):
            tagged = self.modality_embedding(tokens, i)
            encoded[mod_name] = self.modality_encoders[mod_name](tagged)

        # Apply cross-attention between first two modalities
        mod_names = list(encoded.keys())
        if len(mod_names) >= 2:
            a_name, b_name = mod_names[0], mod_names[1]
            for cross_layer in self.cross_attn_layers:
                encoded[a_name], encoded[b_name] = cross_layer(
                    encoded[a_name], encoded[b_name],
                )

        # Normalize
        for mod_name in encoded:
            encoded[mod_name] = self.norm(encoded[mod_name])

        # Cell-level embedding: concatenate mean pools, project
        pooled = []
        for mod_name in mod_names:
            pooled.append(encoded[mod_name].mean(dim=1))  # (batch, token_dim)
        cell_embedding = self.fusion_proj(torch.cat(pooled, dim=-1))

        result = {"cell_embedding": cell_embedding}
        for mod_name in encoded:
            result[f"{mod_name}_embeddings"] = encoded[mod_name]

        return result


# ---------------------------------------------------------------------------
# 3. Late Fusion
# ---------------------------------------------------------------------------

class LateFusionEncoder(nn.Module):
    """Independent per-modality encoders → fuse cell embeddings.

    Most modular: each modality is encoded independently, then cell-level
    representations are combined via attention or MLP.
    """

    def __init__(self, cfg: MultiModalEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.modality_embedding = ModalityEmbedding(
            n_modalities=len(cfg.modalities), token_dim=cfg.token_dim,
        )

        # Per-modality encoders
        self.modality_encoders = nn.ModuleDict()
        for mod_name in cfg.modalities:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.token_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.modality_encoders[mod_name] = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.n_layers,
            )

        self.norm = nn.LayerNorm(cfg.token_dim)

        # Fusion
        n_mod = len(cfg.modalities)
        if cfg.late_fusion_method == "attention":
            self.fusion = nn.MultiheadAttention(
                embed_dim=cfg.token_dim, num_heads=cfg.n_heads,
                dropout=cfg.dropout, batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(cfg.token_dim)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(cfg.token_dim * n_mod, cfg.token_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.token_dim),
            )
            self.fusion_norm = None

        self.embed_dim = cfg.token_dim

    def forward(self, modality_tokens: Dict[str, torch.Tensor]) -> dict:
        """Encode each modality independently, then fuse.

        Args:
            modality_tokens: dict mapping modality name → (batch, n_tokens, token_dim)
        """
        encoded = {}
        cell_embeddings = []

        for i, (mod_name, tokens) in enumerate(modality_tokens.items()):
            tagged = self.modality_embedding(tokens, i)
            out = self.modality_encoders[mod_name](tagged)
            out = self.norm(out)
            encoded[mod_name] = out
            cell_embeddings.append(out.mean(dim=1))  # (batch, token_dim)

        # Fuse cell embeddings
        if self.cfg.late_fusion_method == "attention":
            # Stack as sequence: (batch, n_mod, token_dim)
            stacked = torch.stack(cell_embeddings, dim=1)
            fused, _ = self.fusion(stacked, stacked, stacked)
            cell_embedding = self.fusion_norm(fused.mean(dim=1))
        else:
            cell_embedding = self.fusion(torch.cat(cell_embeddings, dim=-1))

        result = {"cell_embedding": cell_embedding}
        for mod_name in encoded:
            result[f"{mod_name}_embeddings"] = encoded[mod_name]

        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENCODER_REGISTRY = {
    "early": EarlyFusionEncoder,
    "cross_modal": CrossModalEncoder,
    "late": LateFusionEncoder,
}


def build_multimodal_encoder(cfg: MultiModalEncoderConfig) -> nn.Module:
    """Build a multi-modal encoder from config."""
    cls = _ENCODER_REGISTRY.get(cfg.fusion)
    if cls is None:
        raise ValueError(
            f"Unknown fusion: {cfg.fusion}. Available: {list(_ENCODER_REGISTRY.keys())}"
        )
    return cls(cfg)
