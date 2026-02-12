"""Encoder-agnostic JEPA framework for gene-token architectures.

Supports pluggable encoder backends (Transformer, GNN, Perceiver) with
gene-token-level masking and prediction. Retains VICReg-style anti-collapse
regularization.

This supersedes the v1 JEPA class (which used a hardcoded MLP encoder).
The v1 JEPA class is preserved in this file as JEPAv1 for backward
compatibility with existing runs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from celljepa.models.gene_tokenizer import GeneTokenizer, GeneTokenizerConfig
from celljepa.models.masking import MaskResult


# ---------------------------------------------------------------------------
# v2 JEPA (encoder-agnostic, gene-token-level)
# ---------------------------------------------------------------------------

@dataclass
class JEPAv2Config:
    """Configuration for the encoder-agnostic JEPA."""

    # Tokenizer
    tokenizer: GeneTokenizerConfig = field(default_factory=GeneTokenizerConfig)

    # Encoder backend: "transformer", "gnn", "perceiver"
    encoder_type: str = "transformer"

    # Encoder configs are passed as dicts and resolved at build time
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Predictor
    predictor_hidden: int = 512
    predictor_layers: int = 2

    # EMA and training
    ema_decay: float = 0.996
    ema_decay_end: float = 1.0
    ema_warmup_steps: int = 10_000

    # Anti-collapse (VICReg)
    variance_target: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 1.0

    # Loss
    loss_type: str = "smooth_l1"  # "mse", "smooth_l1", "cosine"


def _build_encoder(encoder_type: str, token_dim: int, **kwargs) -> nn.Module:
    """Factory for encoder backends."""
    if encoder_type == "transformer":
        from celljepa.models.encoder_transformer import (
            TransformerGeneEncoder,
            TransformerEncoderConfig,
        )
        cfg = TransformerEncoderConfig(token_dim=token_dim, **kwargs)
        return TransformerGeneEncoder(cfg)

    elif encoder_type == "gnn":
        from celljepa.models.encoder_gnn import (
            GNNGeneEncoder,
            GNNEncoderConfig,
        )
        cfg = GNNEncoderConfig(token_dim=token_dim, **kwargs)
        return GNNGeneEncoder(cfg)

    elif encoder_type == "perceiver":
        from celljepa.models.encoder_perceiver import (
            PerceiverGeneEncoder,
            PerceiverEncoderConfig,
        )
        cfg = PerceiverEncoderConfig(token_dim=token_dim, **kwargs)
        return PerceiverGeneEncoder(cfg)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class GeneTokenPredictor(nn.Module):
    """Predict masked gene-token representations from visible context.

    Takes the cell-level embedding from the student encoder (computed on
    visible tokens) and predicts the teacher representations of masked tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_dim = embed_dim + 1  # +1 for mask ratio
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else embed_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        cell_embedding: torch.Tensor,
        mask_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target representation.

        Args:
            cell_embedding: (batch, embed_dim) from student encoder on visible tokens.
            mask_ratio: (batch,) or (batch, 1) fraction of masked tokens.

        Returns:
            (batch, embed_dim) predicted target representation.
        """
        if mask_ratio.ndim == 1:
            mask_ratio = mask_ratio.unsqueeze(-1)
        x = torch.cat([cell_embedding, mask_ratio], dim=-1)
        return self.net(x)


class JEPAv2(nn.Module):
    """Encoder-agnostic JEPA for gene-token architectures.

    Student processes *visible* gene tokens → cell embedding → predictor → predicted target.
    Teacher processes *all* gene tokens → cell embedding as target.

    Loss = regression(predicted_target, stop_grad(teacher_target)) + VICReg regularization.
    """

    def __init__(self, cfg: JEPAv2Config):
        super().__init__()
        self.cfg = cfg

        # Shared tokenizer
        self.tokenizer = GeneTokenizer(cfg.tokenizer)
        token_dim = self.tokenizer.token_dim

        # Student and teacher share architecture, diverge via EMA
        self.student = _build_encoder(cfg.encoder_type, token_dim, **cfg.encoder_kwargs)
        self.teacher = _build_encoder(cfg.encoder_type, token_dim, **cfg.encoder_kwargs)
        self._init_teacher()

        # Predictor (student side only)
        self.predictor = GeneTokenPredictor(
            embed_dim=token_dim,
            hidden_dim=cfg.predictor_hidden,
            n_layers=cfg.predictor_layers,
        )

        self._step = 0

    def _init_teacher(self) -> None:
        """Initialize teacher as a copy of student; freeze gradients."""
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.copy_(s.data)
            t.requires_grad = False

    @torch.no_grad()
    def update_teacher(self) -> None:
        """EMA update: teacher ← decay * teacher + (1-decay) * student."""
        self._step += 1
        decay = self._current_ema_decay()
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)

    def _current_ema_decay(self) -> float:
        """Cosine schedule for EMA decay warmup."""
        if self._step >= self.cfg.ema_warmup_steps:
            return self.cfg.ema_decay_end
        import math
        ratio = self._step / self.cfg.ema_warmup_steps
        return self.cfg.ema_decay + (self.cfg.ema_decay_end - self.cfg.ema_decay) * (
            1 - math.cos(math.pi * ratio)
        ) / 2

    def forward(
        self,
        expression: torch.Tensor,
        gene_ids: torch.Tensor,
        mask_result: MaskResult,
        edge_index: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """JEPA forward pass.

        Args:
            expression: (batch, n_genes) expression values.
            gene_ids: (n_genes,) gene identity indices.
            mask_result: MaskResult from a masking strategy.
            edge_index: (2, n_edges) optional gene graph for GNN encoders.

        Returns:
            dict with "z_pred", "z_target", "z_student", "mask_ratio",
            and diagnostic tensors.
        """
        # Tokenize (shared between student and teacher)
        tokens = self.tokenizer(expression, gene_ids)  # (batch, n_genes, token_dim)

        # Prepare masks — ensure on same device as tokens
        visible = mask_result.visible_mask.to(tokens.device)  # (n_genes,)
        batch_size = tokens.shape[0]

        # Teacher: full tokens (no masking), no gradients
        with torch.no_grad():
            teacher_out = self._encode(
                self.teacher, tokens, mask=None, edge_index=edge_index,
            )
            z_target = teacher_out["cell_embedding"].detach()

        # Student: only visible tokens
        visible_batch = visible.unsqueeze(0).expand(batch_size, -1)  # (batch, n_genes)
        student_out = self._encode(
            self.student, tokens, mask=visible_batch, edge_index=edge_index,
        )
        z_student = student_out["cell_embedding"]

        # Predict target from student context
        mask_ratio = torch.tensor(
            mask_result.mask_ratio, device=z_student.device
        ).expand(batch_size)
        z_pred = self.predictor(z_student, mask_ratio)

        return {
            "z_pred": z_pred,
            "z_target": z_target,
            "z_student": z_student,
            "mask_ratio": mask_ratio,
        }

    def _encode(
        self,
        encoder: nn.Module,
        tokens: torch.Tensor,
        mask: torch.Tensor | None,
        edge_index: torch.Tensor | None,
    ) -> dict:
        """Route to the appropriate encoder forward method."""
        if self.cfg.encoder_type == "gnn" and edge_index is not None:
            return encoder(tokens, edge_index=edge_index, mask=mask)
        else:
            return encoder(tokens, mask=mask)

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute JEPA loss + VICReg regularization.

        Args:
            outputs: dict from forward() with z_pred, z_target, z_student.

        Returns:
            dict with "loss", "repr_loss", "var_loss", "cov_loss".
        """
        z_pred = outputs["z_pred"]
        z_target = outputs["z_target"]
        z_student = outputs["z_student"]

        # Representation loss
        if self.cfg.loss_type == "mse":
            repr_loss = nn.functional.mse_loss(z_pred, z_target)
        elif self.cfg.loss_type == "smooth_l1":
            repr_loss = nn.functional.smooth_l1_loss(z_pred, z_target)
        elif self.cfg.loss_type == "cosine":
            repr_loss = 1 - nn.functional.cosine_similarity(z_pred, z_target, dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.cfg.loss_type}")

        # VICReg anti-collapse
        vicreg = variance_covariance_loss(z_student, self.cfg.variance_target)

        loss = (
            repr_loss
            + self.cfg.variance_weight * vicreg["var_loss"]
            + self.cfg.covariance_weight * vicreg["cov_loss"]
        )

        return {
            "loss": loss,
            "repr_loss": repr_loss,
            "var_loss": vicreg["var_loss"],
            "cov_loss": vicreg["cov_loss"],
        }


# ---------------------------------------------------------------------------
# Anti-collapse regularization (shared)
# ---------------------------------------------------------------------------

def variance_covariance_loss(z: torch.Tensor, target_var: float = 1.0) -> Dict[str, torch.Tensor]:
    """VICReg-style variance/covariance regularizer.

    Prevents representational collapse by encouraging:
    - Variance of each embedding dimension ≥ target_var
    - Off-diagonal covariance ≈ 0 (decorrelation)
    """
    z = z - z.mean(dim=0)
    var = z.var(dim=0) + 1e-4
    var_loss = torch.mean(torch.relu(target_var - var))

    cov = (z.T @ z) / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).mean()

    return {"var_loss": var_loss, "cov_loss": cov_loss}


# ---------------------------------------------------------------------------
# v1 JEPA (preserved for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class JepaConfig:
    """v1 JEPA config (MLP encoder). Kept for backward compatibility."""
    input_dim: int
    embed_dim: int = 256
    hidden_dim: int = 512
    predictor_hidden: int = 512
    ema_decay: float = 0.99
    mask_ratio: float = 0.25
    variance_target: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 1.0


class MLPEncoder(nn.Module):
    """v1 MLP encoder (flat vector input). Kept for backward compatibility."""
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """v1 predictor. Kept for backward compatibility."""
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z_ctx: torch.Tensor, mask_ratio: torch.Tensor) -> torch.Tensor:
        if mask_ratio.ndim == 1:
            mask_ratio = mask_ratio[:, None]
        x = torch.cat([z_ctx, mask_ratio], dim=1)
        return self.net(x)


class JEPA(nn.Module):
    """v1 JEPA with MLP encoder. Kept for backward compatibility with M0–M3 runs."""

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg
        self.student = MLPEncoder(cfg.input_dim, cfg.embed_dim, cfg.hidden_dim)
        self.teacher = MLPEncoder(cfg.input_dim, cfg.embed_dim, cfg.hidden_dim)
        self.predictor = Predictor(cfg.embed_dim, cfg.predictor_hidden)
        self._init_teacher()

    def _init_teacher(self) -> None:
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.copy_(s.data)
            t.requires_grad = False

    @torch.no_grad()
    def update_teacher(self) -> None:
        m = self.cfg.ema_decay
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_ctx = x * (1.0 - mask)
        x_tgt = x * mask
        z_ctx = self.student(x_ctx)
        with torch.no_grad():
            z_tgt = self.teacher(x_tgt)
        mask_ratio = mask.mean(dim=1)
        z_pred = self.predictor(z_ctx, mask_ratio)
        return z_pred, z_tgt, z_ctx
