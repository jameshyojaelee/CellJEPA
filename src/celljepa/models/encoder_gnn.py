"""GNN encoder operating on a gene interaction graph (GEARS-inspired).

Gene tokens are node features; message passing along PPI/GO edges
propagates information through known gene-gene interactions, providing
a biologically grounded inductive bias for perturbation prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GNNEncoderConfig:
    token_dim: int = 256
    n_layers: int = 4
    n_heads: int = 4  # for GAT
    dropout: float = 0.1
    residual: bool = True
    pooling: str = "mean"  # "mean" or "attention"
    gnn_type: str = "gat"  # "gat" or "gin"


class GATLayer(nn.Module):
    """Graph Attention Network layer (simplified, no external dependency).

    Multi-head attention over graph neighbors, using edge indices.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.01)
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.01)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Graph attention forward.

        Args:
            x: (n_nodes, in_dim) node features.
            edge_index: (2, n_edges) edge list [src, dst].

        Returns:
            (n_nodes, out_dim) updated node features.
        """
        n_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Project and reshape for multi-head: (n_nodes, n_heads, head_dim)
        h = self.W(x).view(n_nodes, self.n_heads, self.head_dim)

        # Attention scores
        e_src = (h[src] * self.a_src).sum(dim=-1)  # (n_edges, n_heads)
        e_dst = (h[dst] * self.a_dst).sum(dim=-1)
        e = self.leaky_relu(e_src + e_dst)  # (n_edges, n_heads)

        # Softmax over neighbors
        alpha = self._sparse_softmax(e, dst, n_nodes)  # (n_edges, n_heads)
        alpha = self.dropout(alpha)

        # Aggregate
        msg = alpha.unsqueeze(-1) * h[src]  # (n_edges, n_heads, head_dim)
        out = torch.zeros(n_nodes, self.n_heads, self.head_dim, device=x.device)
        dst_expanded = dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg)
        out.scatter_add_(0, dst_expanded, msg)

        return out.view(n_nodes, -1)  # (n_nodes, out_dim)

    @staticmethod
    def _sparse_softmax(
        scores: torch.Tensor,
        index: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """Softmax over sparse neighborhoods."""
        # Numerical stability
        max_scores = torch.zeros(n_nodes, scores.shape[-1], device=scores.device)
        max_scores.scatter_reduce_(0, index.unsqueeze(-1).expand_as(scores), scores, reduce="amax")
        scores = scores - max_scores[index]

        exp_scores = scores.exp()
        sum_exp = torch.zeros(n_nodes, scores.shape[-1], device=scores.device)
        sum_exp.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_scores), exp_scores)
        return exp_scores / (sum_exp[index] + 1e-8)


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer.

    GIN has maximal discriminative power among message-passing GNNs.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GIN forward.

        Args:
            x: (n_nodes, in_dim) node features.
            edge_index: (2, n_edges) edge list [src, dst].

        Returns:
            (n_nodes, out_dim) updated node features.
        """
        src, dst = edge_index[0], edge_index[1]
        # Aggregate neighbors
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.shape[-1]), x[src])
        out = self.mlp((1 + self.epsilon) * x + agg)
        return self.dropout(out)


class GNNGeneEncoder(nn.Module):
    """Graph Neural Network encoder for gene tokens.

    Gene tokens are placed as node features on a gene interaction graph.
    Message passing propagates information along known biological edges.
    """

    def __init__(self, cfg: GNNEncoderConfig):
        super().__init__()
        self.cfg = cfg

        layers = []
        norms = []
        for _ in range(cfg.n_layers):
            if cfg.gnn_type == "gat":
                layers.append(GATLayer(cfg.token_dim, cfg.token_dim, cfg.n_heads, cfg.dropout))
            else:
                layers.append(GINLayer(cfg.token_dim, cfg.token_dim, cfg.dropout))
            norms.append(nn.LayerNorm(cfg.token_dim))
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)

        self.residual = cfg.residual
        self.dropout = nn.Dropout(cfg.dropout)

        # Attention pooling for cell-level readout
        if cfg.pooling == "attention":
            self.pool_attn = nn.Linear(cfg.token_dim, 1)
        else:
            self.pool_attn = None

        self.embed_dim = cfg.token_dim

    def forward(
        self,
        gene_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict:
        """Encode gene tokens via GNN message passing.

        Args:
            gene_tokens: (batch, n_genes, token_dim) from GeneTokenizer.
            edge_index: (2, n_edges) graph topology (shared across batch).
            mask: (batch, n_genes) bool, True = visible gene. If None, all visible.

        Returns:
            dict with "gene_embeddings" and "cell_embedding".
        """
        batch_size, n_genes, token_dim = gene_tokens.shape

        # Process each cell independently through the graph
        # (GNN operates on node features; batch dimension is over cells)
        all_gene_embs = []
        all_cell_embs = []

        for b in range(batch_size):
            x = gene_tokens[b]  # (n_genes, token_dim)

            for layer, norm in zip(self.layers, self.norms):
                h = layer(x, edge_index)
                h = norm(h)
                if self.residual:
                    x = x + self.dropout(h)
                else:
                    x = self.dropout(h)

            all_gene_embs.append(x)

            # Pool to cell level
            if mask is not None:
                cell_mask = mask[b]  # (n_genes,)
                visible_x = x[cell_mask]  # (n_visible, token_dim)
            else:
                visible_x = x

            if self.pool_attn is not None and visible_x.shape[0] > 0:
                weights = torch.softmax(self.pool_attn(visible_x), dim=0)
                cell_emb = (weights * visible_x).sum(dim=0)
            elif visible_x.shape[0] > 0:
                cell_emb = visible_x.mean(dim=0)
            else:
                cell_emb = torch.zeros(token_dim, device=x.device)

            all_cell_embs.append(cell_emb)

        gene_embeddings = torch.stack(all_gene_embs)  # (batch, n_genes, token_dim)
        cell_embedding = torch.stack(all_cell_embs)  # (batch, token_dim)

        return {
            "gene_embeddings": gene_embeddings,
            "cell_embedding": cell_embedding,
        }
