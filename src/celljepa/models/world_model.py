"""World model / transition predictors (P3 overhaul).

Three world model architectures that predict post-perturbation cell state
from control cell state + perturbation embedding (from P2 encoders).
All retain safety-by-construction: residual connection + shrinkage α + Δ clamping.

Architectures:
  1. AttentionWorldModel — cross-attention from perturbation to gene tokens
  2. GraphConditionedWorldModel — perturbation as node intervention on gene graph
  3. DisentangledWorldModel — factorized latent space

Usage::

    from celljepa.models.world_model import WorldModelV2Config, build_world_model

    cfg = WorldModelV2Config(model_type="attention", embed_dim=256)
    model = build_world_model(cfg)
    out = model(
        cell_embedding=z_ctrl,         # (B, D)
        perturbation_embedding=p_emb,  # (B, D)
        gene_embeddings=gene_embs,     # (B, G, D)  (optional, for attention/graph)
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from celljepa.models.encoder_gnn import GATLayer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp_by_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clamp tensor norms to max_norm."""
    if max_norm <= 0:
        return torch.zeros_like(x)
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
    return x * scale


def _apply_safety(
    control: torch.Tensor,
    delta: torch.Tensor,
    residual: bool,
    alpha: float,
    max_delta_norm: Optional[float],
) -> torch.Tensor:
    """Apply safety-by-construction: z_pred = z_ctrl + α * clamp(Δ)."""
    if max_delta_norm is not None:
        delta = _clamp_by_norm(delta, max_delta_norm)
    if residual:
        return control + alpha * delta
    return alpha * delta


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WorldModelV2Config:
    """Configuration for v2 world models."""

    model_type: str = "attention"  # attention | graph | disentangled
    embed_dim: int = 256
    hidden_dim: int = 512

    # Attention world model
    attn_n_heads: int = 4
    attn_n_layers: int = 2

    # Graph-conditioned world model
    gnn_layers: int = 2
    gnn_heads: int = 4

    # Disentangled world model
    n_factors: int = 3  # cell_state, perturbation, covariates
    factor_dim: int = 128

    # Safety-by-construction
    residual: bool = True
    alpha: float = 1.0
    max_delta_norm: float | None = None

    # General
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# 1. Attention-Based World Model
# ---------------------------------------------------------------------------

class AttentionWorldModel(nn.Module):
    """Cross-attention world model: perturbation attends to gene tokens.

    The perturbation embedding queries the gene-token embeddings via
    cross-attention, producing per-gene perturbation effects. This is
    interpretable: attention weights reveal which genes the perturbation
    affects most.
    """

    def __init__(self, cfg: WorldModelV2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim

        # Project perturbation embedding to query
        self.pert_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        # Cross-attention layers: perturbation queries gene tokens
        cross_attn_layers = []
        cross_norms = []
        ffn_layers = []
        ffn_norms = []
        for _ in range(cfg.attn_n_layers):
            cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.attn_n_heads,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
            )
            cross_norms.append(nn.LayerNorm(cfg.embed_dim))
            ffn_layers.append(nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.embed_dim),
            ))
            ffn_norms.append(nn.LayerNorm(cfg.embed_dim))

        self.cross_attn_layers = nn.ModuleList(cross_attn_layers)
        self.cross_norms = nn.ModuleList(cross_norms)
        self.ffn_layers = nn.ModuleList(ffn_layers)
        self.ffn_norms = nn.ModuleList(ffn_norms)

        # Per-gene delta predictor: gene embedding + perturbation context → scalar delta
        self.delta_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(
        self,
        cell_embedding: torch.Tensor,
        perturbation_embedding: torch.Tensor,
        gene_embeddings: Optional[torch.Tensor] = None,
        covariate_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Predict post-perturbation state via cross-attention.

        Args:
            cell_embedding: (B, D) control cell embedding.
            perturbation_embedding: (B, D) perturbation embedding from P2 encoder.
            gene_embeddings: (B, G, D) per-gene embeddings from encoder.
            covariate_embedding: (B, D) optional dose/time embedding.

        Returns:
            dict with "cell_embedding" (B, D) and "gene_deltas" (B, G, D).
        """
        B = cell_embedding.shape[0]

        # Combine perturbation + optional covariates
        pert_q = self.pert_proj(perturbation_embedding)
        if covariate_embedding is not None:
            pert_q = pert_q + covariate_embedding
        pert_q = pert_q.unsqueeze(1)  # (B, 1, D) — single query token

        # If gene_embeddings not available, create from cell_embedding
        if gene_embeddings is None:
            gene_embeddings = cell_embedding.unsqueeze(1)  # (B, 1, D)

        # Cross-attention: perturbation queries gene tokens
        x = gene_embeddings  # (B, G, D)
        for cross_attn, cross_norm, ffn, ffn_norm in zip(
            self.cross_attn_layers, self.cross_norms,
            self.ffn_layers, self.ffn_norms,
        ):
            # Perturbation-to-gene cross-attention (updates gene representations)
            attn_out, _ = cross_attn(
                query=pert_q, key=x, value=x,
            )
            # Broadcast perturbation effect back to gene tokens
            pert_effect = attn_out.expand_as(x)
            x = cross_norm(x + pert_effect)
            x = ffn_norm(x + ffn(x))

        # Per-gene deltas
        gene_deltas = self.delta_head(x)  # (B, G, D)

        # Cell-level delta: mean-pool gene deltas
        cell_delta = gene_deltas.mean(dim=1)  # (B, D)

        # Apply safety
        pred_cell = _apply_safety(
            cell_embedding, cell_delta,
            self.cfg.residual, self.cfg.alpha, self.cfg.max_delta_norm,
        )

        return {
            "cell_embedding": pred_cell,
            "gene_deltas": gene_deltas,
        }


# ---------------------------------------------------------------------------
# 2. Graph-Conditioned World Model
# ---------------------------------------------------------------------------

class GraphConditionedWorldModel(nn.Module):
    """Graph-conditioned world model: perturbation as node intervention.

    The perturbation signal is injected at the perturbed gene's node, then
    propagated through the gene interaction graph via GAT message passing.
    This directly models the biological causal mechanism: a perturbation
    affects a gene, and its effects propagate through regulatory networks.
    """

    def __init__(self, cfg: WorldModelV2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim

        # Perturbation injection: combine perturbation embedding with node features
        self.pert_inject = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

        # GNN message passing to propagate perturbation signal
        gnn_layers = []
        gnn_norms = []
        for _ in range(cfg.gnn_layers):
            gnn_layers.append(
                GATLayer(cfg.embed_dim, cfg.embed_dim, cfg.gnn_heads, cfg.dropout)
            )
            gnn_norms.append(nn.LayerNorm(cfg.embed_dim))
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.gnn_norms = nn.ModuleList(gnn_norms)

        # Delta head
        self.delta_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(
        self,
        cell_embedding: torch.Tensor,
        perturbation_embedding: torch.Tensor,
        gene_embeddings: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        perturbed_gene_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Predict post-perturbation state via graph message passing.

        Args:
            cell_embedding: (B, D) control cell embedding.
            perturbation_embedding: (B, D) perturbation embedding.
            gene_embeddings: (B, G, D) per-gene embeddings from encoder.
            edge_index: (2, E) gene interaction graph edges.
            perturbed_gene_idx: (B,) index of perturbed gene in the gene list.

        Returns:
            dict with "cell_embedding" (B, D) and "gene_deltas" (B, G, D).
        """
        B = cell_embedding.shape[0]

        if gene_embeddings is None:
            gene_embeddings = cell_embedding.unsqueeze(1)

        _, G, D = gene_embeddings.shape

        all_gene_deltas = []
        for b in range(B):
            x = gene_embeddings[b]  # (G, D)

            # Inject perturbation at perturbed gene node(s)
            if perturbed_gene_idx is not None and edge_index is not None:
                node_idx = perturbed_gene_idx[b].item()
                node_feat = x[node_idx]  # (D,)
                pert_feat = perturbation_embedding[b]  # (D,)
                injected = self.pert_inject(
                    torch.cat([node_feat, pert_feat], dim=-1).unsqueeze(0)
                ).squeeze(0)
                x = x.clone()
                x[node_idx] = x[node_idx] + injected

            # Message passing
            if edge_index is not None:
                for layer, norm in zip(self.gnn_layers, self.gnn_norms):
                    h = layer(x, edge_index)
                    h = norm(h)
                    x = x + F.dropout(h, p=self.cfg.dropout, training=self.training)

            # Compute delta from original
            gene_delta = self.delta_head(x - gene_embeddings[b])  # (G, D)
            all_gene_deltas.append(gene_delta)

        gene_deltas = torch.stack(all_gene_deltas)  # (B, G, D)
        cell_delta = gene_deltas.mean(dim=1)  # (B, D)

        pred_cell = _apply_safety(
            cell_embedding, cell_delta,
            self.cfg.residual, self.cfg.alpha, self.cfg.max_delta_norm,
        )

        return {
            "cell_embedding": pred_cell,
            "gene_deltas": gene_deltas,
        }


# ---------------------------------------------------------------------------
# 3. Disentangled World Model
# ---------------------------------------------------------------------------

class DisentangledWorldModel(nn.Module):
    """Disentangled world model: factorized latent space.

    Factorizes the prediction into separate components:
    - Base cell state
    - Perturbation effect
    - Covariate modulation (dose, time, cell type)

    These are combined via learned gating to predict the post-perturbation
    state. Enables counterfactual reasoning by swapping components.
    """

    def __init__(self, cfg: WorldModelV2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim

        factor_dim = cfg.factor_dim

        # Factor encoders
        self.cell_factor = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, factor_dim),
        )
        self.pert_factor = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, factor_dim),
        )
        self.cov_factor = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, factor_dim),
        )

        # Gated fusion
        total_factor_dim = factor_dim * 3
        self.gate = nn.Sequential(
            nn.Linear(total_factor_dim, total_factor_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(total_factor_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(
        self,
        cell_embedding: torch.Tensor,
        perturbation_embedding: torch.Tensor,
        covariate_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Predict post-perturbation state via factorized latent space.

        Args:
            cell_embedding: (B, D) control cell embedding.
            perturbation_embedding: (B, D) perturbation embedding.
            covariate_embedding: (B, D) optional dose/time embedding.
                                 If None, uses zeros.

        Returns:
            dict with "cell_embedding" (B, D) and "factors" dict.
        """
        B, D = cell_embedding.shape

        # Extract factors
        f_cell = self.cell_factor(cell_embedding)        # (B, factor_dim)
        f_pert = self.pert_factor(perturbation_embedding) # (B, factor_dim)

        if covariate_embedding is not None:
            f_cov = self.cov_factor(covariate_embedding)  # (B, factor_dim)
        else:
            f_cov = torch.zeros_like(f_cell)

        # Gated fusion
        combined = torch.cat([f_cell, f_pert, f_cov], dim=-1)  # (B, 3*factor_dim)
        gate = self.gate(combined)
        gated = combined * gate

        delta = self.fusion(gated)  # (B, D)

        pred_cell = _apply_safety(
            cell_embedding, delta,
            self.cfg.residual, self.cfg.alpha, self.cfg.max_delta_norm,
        )

        return {
            "cell_embedding": pred_cell,
            "factors": {
                "cell": f_cell,
                "perturbation": f_pert,
                "covariate": f_cov,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_world_model(cfg: WorldModelV2Config) -> nn.Module:
    """Factory: build the world model specified by ``cfg.model_type``."""
    t = cfg.model_type
    if t == "attention":
        return AttentionWorldModel(cfg)
    elif t == "graph":
        return GraphConditionedWorldModel(cfg)
    elif t == "disentangled":
        return DisentangledWorldModel(cfg)
    else:
        raise ValueError(
            f"Unknown world model type: {t!r}. "
            f"Expected one of: attention, graph, disentangled."
        )


# ---------------------------------------------------------------------------
# v1 World Model (preserved for backward compatibility)
# ---------------------------------------------------------------------------

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
    """Minimal set-to-set predictor conditioned on action embeddings (v1)."""

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
