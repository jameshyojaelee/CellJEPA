"""Biologically-grounded perturbation encoders (P2).

Five encoder classes that produce fixed-dimensional perturbation embeddings
from diverse input modalities: gene identity, gene interaction graphs,
chemical fingerprints, combinatorial perturbations, and dose/time covariates.

These replace the v1 embedding-lookup approach (``transition.py``
``PerturbationEmbedding``) with encodings that carry biological priors.

Usage::

    from celljepa.models.perturbation_encoders import (
        PerturbationEncoderConfig,
        build_perturbation_encoder,
    )

    cfg = PerturbationEncoderConfig(encoder_type="gene_identity", embed_dim=256)
    encoder = build_perturbation_encoder(cfg)
    # gene_idx: (batch,) integer gene indices
    emb = encoder(gene_idx=gene_idx)  # (batch, embed_dim)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from celljepa.models.encoder_gnn import GATLayer
from celljepa.models.gene_tokenizer import GeneIdentityEmbedding


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PerturbationEncoderConfig:
    """Configuration for perturbation encoders."""

    encoder_type: str = "gene_identity"  # gene_identity | gene_graph | chemical_fingerprint | combinatorial | dose_time
    embed_dim: int = 256

    # Gene identity / graph shared
    n_genes: int = 20_000
    gene_embed_dim: int = 128

    # Gene graph encoder
    gnn_layers: int = 2
    gnn_heads: int = 4
    gnn_dropout: float = 0.1

    # Chemical fingerprint encoder
    fingerprint_dim: int = 2048
    fingerprint_hidden: int = 512

    # Combinatorial encoder
    combo_n_heads: int = 4
    combo_max_perts: int = 16

    # Dose/time encoder
    n_covariates: int = 2  # [log_dose, log_time]
    covariate_hidden: int = 128

    # General
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# 1. Gene Identity Perturbation Encoder
# ---------------------------------------------------------------------------

class GeneIdentityPerturbationEncoder(nn.Module):
    """Encode perturbations via gene identity embeddings.

    The simplest biologically-grounded approach: reuse the learned gene
    identity embeddings (which encode what a gene *is*) as the perturbation
    representation for single-gene perturbations.

    Optionally shares weights with the tokenizer's GeneIdentityEmbedding.
    """

    def __init__(
        self,
        cfg: PerturbationEncoderConfig,
        shared_embedding: Optional[GeneIdentityEmbedding] = None,
    ):
        super().__init__()
        if shared_embedding is not None:
            self.gene_embedding = shared_embedding
            source_dim = shared_embedding.embed_dim
        else:
            self.gene_embedding = GeneIdentityEmbedding(
                n_genes=cfg.n_genes,
                embed_dim=cfg.gene_embed_dim,
            )
            source_dim = cfg.gene_embed_dim

        # Project to embed_dim if dimensions differ
        if source_dim != cfg.embed_dim:
            self.projection = nn.Sequential(
                nn.Linear(source_dim, cfg.embed_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.embed_dim),
            )
        else:
            self.projection = nn.LayerNorm(cfg.embed_dim)

        self.embed_dim = cfg.embed_dim

    def forward(self, gene_idx: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode single-gene perturbations.

        Args:
            gene_idx: (batch,) integer gene indices.

        Returns:
            (batch, embed_dim) perturbation embeddings.
        """
        emb = self.gene_embedding(gene_idx)  # (batch, gene_embed_dim)
        return self.projection(emb)  # (batch, embed_dim)


# ---------------------------------------------------------------------------
# 2. Gene Graph Perturbation Encoder
# ---------------------------------------------------------------------------

class GeneGraphPerturbationEncoder(nn.Module):
    """Encode perturbations via GNN message passing on PPI/GO graph.

    The perturbed gene's embedding is enriched by its neighborhood in the
    gene interaction graph, capturing pathway context and regulatory partners.

    Uses GAT layers from ``encoder_gnn.py`` for message passing.
    """

    def __init__(
        self,
        cfg: PerturbationEncoderConfig,
        shared_embedding: Optional[GeneIdentityEmbedding] = None,
    ):
        super().__init__()
        if shared_embedding is not None:
            self.gene_embedding = shared_embedding
            source_dim = shared_embedding.embed_dim
        else:
            self.gene_embedding = GeneIdentityEmbedding(
                n_genes=cfg.n_genes,
                embed_dim=cfg.gene_embed_dim,
            )
            source_dim = cfg.gene_embed_dim

        # Project gene embeddings to working dimension for GNN
        self.input_proj = nn.Linear(source_dim, cfg.embed_dim)

        # GNN message passing layers
        gnn_layers = []
        gnn_norms = []
        for _ in range(cfg.gnn_layers):
            gnn_layers.append(
                GATLayer(cfg.embed_dim, cfg.embed_dim, cfg.gnn_heads, cfg.gnn_dropout)
            )
            gnn_norms.append(nn.LayerNorm(cfg.embed_dim))
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.gnn_norms = nn.ModuleList(gnn_norms)

        self.dropout = nn.Dropout(cfg.dropout)
        self.embed_dim = cfg.embed_dim

    def forward(
        self,
        gene_idx: torch.Tensor,
        edge_index: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode perturbations via graph-contextualized gene embeddings.

        Args:
            gene_idx: (batch,) integer indices of perturbed genes.
            edge_index: (2, n_edges) gene interaction graph edges.
            gene_ids: (n_genes,) optional full gene index tensor. If None,
                      uses arange(n_genes) based on the embedding table size.

        Returns:
            (batch, embed_dim) perturbation embeddings.
        """
        # Build full gene node features
        if gene_ids is None:
            n_genes = self.gene_embedding.embedding.num_embeddings - 1  # exclude padding
            gene_ids = torch.arange(1, n_genes + 1, device=gene_idx.device)

        all_emb = self.gene_embedding(gene_ids)  # (n_genes, gene_embed_dim)
        x = self.input_proj(all_emb)  # (n_genes, embed_dim)

        # Message passing
        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            h = layer(x, edge_index)
            h = norm(h)
            x = x + self.dropout(h)  # residual

        # Extract perturbed gene embeddings
        # gene_idx values should correspond to positions in gene_ids
        # gene_ids is 1-indexed (to avoid padding_idx=0), so adjust
        out = x[gene_idx - 1] if gene_ids[0] == 1 else x[gene_idx]
        return out  # (batch, embed_dim)


# ---------------------------------------------------------------------------
# 3. Chemical Fingerprint Encoder
# ---------------------------------------------------------------------------

class ChemicalFingerprintEncoder(nn.Module):
    """Encode drug perturbations via precomputed Morgan fingerprints.

    SMILES → Morgan fingerprint conversion is a preprocessing step.
    This module maps the binary fingerprint vector to a dense embedding.
    """

    def __init__(self, cfg: PerturbationEncoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.fingerprint_dim, cfg.fingerprint_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fingerprint_hidden, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
        )
        self.embed_dim = cfg.embed_dim

    def forward(self, fingerprint: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode drug perturbation from fingerprint.

        Args:
            fingerprint: (batch, fingerprint_dim) binary or float fingerprint vector.

        Returns:
            (batch, embed_dim) perturbation embedding.
        """
        return self.net(fingerprint)


# ---------------------------------------------------------------------------
# 4. Combinatorial Perturbation Encoder
# ---------------------------------------------------------------------------

class CombinatorialPerturbationEncoder(nn.Module):
    """Encode combinatorial perturbations via attentive pooling.

    For multi-gene or multi-drug perturbations, individual perturbation
    embeddings are combined via cross-attention from a learned query token.
    This captures interaction effects beyond simple averaging.
    """

    def __init__(self, cfg: PerturbationEncoderConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim

        # Learned query token for attention pooling
        self.combo_query = nn.Parameter(torch.randn(1, 1, cfg.embed_dim) * 0.02)

        # Multi-head cross-attention: query attends to individual perturbation embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.combo_n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
        )
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

    def forward(
        self,
        pert_embeddings: torch.Tensor,
        pert_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Combine multiple perturbation embeddings into one.

        Args:
            pert_embeddings: (batch, n_perts, embed_dim) individual perturbation
                             embeddings (from any single-perturbation encoder).
            pert_mask: (batch, n_perts) bool mask, True = valid perturbation,
                       False = padding. If None, all positions are valid.

        Returns:
            (batch, embed_dim) combined perturbation embedding.
        """
        batch_size = pert_embeddings.shape[0]
        query = self.combo_query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)

        # key_padding_mask expects True = *ignore*, so invert pert_mask
        key_padding_mask = None
        if pert_mask is not None:
            key_padding_mask = ~pert_mask  # True = padding position

        # Cross-attention: query attends to perturbation embeddings
        attn_out, _ = self.cross_attn(
            query=query,
            key=pert_embeddings,
            value=pert_embeddings,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm(query + attn_out)  # residual + norm

        # FFN
        x = self.norm2(x + self.ffn(x))

        return x.squeeze(1)  # (batch, embed_dim)


# ---------------------------------------------------------------------------
# 5. Dose/Time Encoder
# ---------------------------------------------------------------------------

class DoseTimeEncoder(nn.Module):
    """Encode continuous dose and time covariates.

    Maps (log_dose, log_time) or arbitrary continuous covariates to a
    dense embedding. Designed to be summed or concatenated with other
    perturbation embeddings for conditioning.
    """

    def __init__(self, cfg: PerturbationEncoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_covariates, cfg.covariate_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.covariate_hidden, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
        )
        self.embed_dim = cfg.embed_dim

    def forward(self, covariates: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode dose/time covariates.

        Args:
            covariates: (batch, n_covariates) continuous values,
                        e.g. [log(dose), log(time_hours)].

        Returns:
            (batch, embed_dim) covariate embedding.
        """
        return self.net(covariates)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_perturbation_encoder(
    cfg: PerturbationEncoderConfig,
    shared_embedding: Optional[GeneIdentityEmbedding] = None,
) -> nn.Module:
    """Factory: build the perturbation encoder specified by ``cfg.encoder_type``.

    Args:
        cfg: Perturbation encoder configuration.
        shared_embedding: Optional shared GeneIdentityEmbedding (for weight tying
                          with the gene tokenizer). Used by gene_identity and
                          gene_graph encoders.

    Returns:
        An ``nn.Module`` perturbation encoder.
    """
    t = cfg.encoder_type
    if t == "gene_identity":
        return GeneIdentityPerturbationEncoder(cfg, shared_embedding=shared_embedding)
    elif t == "gene_graph":
        return GeneGraphPerturbationEncoder(cfg, shared_embedding=shared_embedding)
    elif t == "chemical_fingerprint":
        return ChemicalFingerprintEncoder(cfg)
    elif t == "combinatorial":
        return CombinatorialPerturbationEncoder(cfg)
    elif t == "dose_time":
        return DoseTimeEncoder(cfg)
    else:
        raise ValueError(
            f"Unknown perturbation encoder type: {t!r}. "
            f"Expected one of: gene_identity, gene_graph, chemical_fingerprint, "
            f"combinatorial, dose_time."
        )
