"""Cross-modal JEPA for multi-modal CellJEPA (P6).

Extends JEPAv2 with cross-modal prediction: mask tokens in one modality,
predict teacher targets in another. Forces the model to learn cross-modal
dependencies (e.g., chromatin accessibility → gene expression).

Usage::

    from celljepa.models.jepa_multimodal import (
        MultiModalJEPAConfig, MultiModalJEPA,
    )
    cfg = MultiModalJEPAConfig(
        modalities=["rna", "atac"],
        fusion="cross_modal",
    )
    model = MultiModalJEPA(cfg)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from celljepa.models.encoder_multimodal import (
    MultiModalEncoderConfig,
    build_multimodal_encoder,
)


@dataclass
class MultiModalJEPAConfig:
    """Configuration for cross-modal JEPA."""

    # Modalities
    modalities: List[str] = field(default_factory=lambda: ["rna", "atac"])
    fusion: str = "cross_modal"

    # Encoder
    token_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    # Predictor
    predictor_hidden: int = 256
    predictor_layers: int = 2

    # EMA
    ema_decay: float = 0.996
    ema_decay_end: float = 1.0
    ema_warmup_steps: int = 5000

    # VICReg anti-collapse
    vicreg_weight: float = 0.04

    # Cross-modal masking
    mask_ratio: float = 0.5
    cross_modal_prediction: bool = True  # mask A → predict B


class CrossModalPredictor(nn.Module):
    """Predicts masked token representations in a target modality
    from visible context in another modality.
    """

    def __init__(self, token_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers = []
        in_dim = token_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else token_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim),
            ])
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalJEPA(nn.Module):
    """Cross-modal JEPA: mask one modality, predict another.

    Architecture:
      - Student encoder: processes visible tokens from all modalities
      - Teacher encoder: processes all tokens from all modalities (EMA)
      - Cross-modal predictor: predicts masked targets in modality B
        from student embeddings of modality A
    """

    def __init__(self, cfg: MultiModalJEPAConfig):
        super().__init__()
        self.cfg = cfg

        # Build encoder config
        encoder_cfg = MultiModalEncoderConfig(
            token_dim=cfg.token_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            fusion=cfg.fusion,
            modalities=cfg.modalities,
        )

        # Student and teacher
        self.student = build_multimodal_encoder(encoder_cfg)
        self.teacher = build_multimodal_encoder(encoder_cfg)
        self._init_teacher()

        # Per-modality-pair predictors
        self.predictors = nn.ModuleDict()
        for source_mod in cfg.modalities:
            for target_mod in cfg.modalities:
                if source_mod != target_mod or not cfg.cross_modal_prediction:
                    key = f"{source_mod}_to_{target_mod}"
                    self.predictors[key] = CrossModalPredictor(
                        token_dim=cfg.token_dim,
                        hidden_dim=cfg.predictor_hidden,
                        n_layers=cfg.predictor_layers,
                    )

        self._step = 0

    def _init_teacher(self):
        """Initialize teacher as copy of student, no gradients."""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """EMA update of teacher parameters."""
        cfg = self.cfg
        # Linear warmup of decay
        progress = min(self._step / max(cfg.ema_warmup_steps, 1), 1.0)
        decay = cfg.ema_decay + (cfg.ema_decay_end - cfg.ema_decay) * progress

        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)

        self._step += 1

    def generate_mask(
        self, n_tokens: int, batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """Generate random mask: True = visible, False = masked."""
        n_visible = max(1, int(n_tokens * (1 - self.cfg.mask_ratio)))
        mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)
        for b in range(batch_size):
            perm = torch.randperm(n_tokens, device=device)[:n_visible]
            mask[b, perm] = True
        return mask

    def forward(
        self,
        modality_tokens: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> dict:
        """Forward pass with cross-modal masking.

        Args:
            modality_tokens: {modality_name: (batch, n_tokens, token_dim)}
            masks: optional {modality_name: (batch, n_tokens) bool mask}.
                   If None, random masks are generated.

        Returns:
            dict with "loss", "student_output", "teacher_output", per-modality losses.
        """
        batch_size = next(iter(modality_tokens.values())).shape[0]
        device = next(iter(modality_tokens.values())).device

        # Generate masks if not provided
        if masks is None:
            masks = {}
            for mod_name, tokens in modality_tokens.items():
                masks[mod_name] = self.generate_mask(
                    tokens.shape[1], batch_size, device,
                )

        # Student: process visible tokens only
        student_input = {}
        for mod_name, tokens in modality_tokens.items():
            mask = masks[mod_name]  # (batch, n_tokens)
            # Apply mask: keep only visible tokens
            # For simplicity, zero out masked tokens (Transformer handles via attention)
            masked_tokens = tokens * mask.unsqueeze(-1).float()
            student_input[mod_name] = masked_tokens

        student_out = self.student(student_input)

        # Teacher: all tokens, no gradient
        with torch.no_grad():
            teacher_out = self.teacher(modality_tokens)

        # Compute losses
        total_loss = torch.tensor(0.0, device=device)
        losses = {}

        if self.cfg.cross_modal_prediction:
            # Cross-modal: predict modality B targets from modality A context
            mod_names = list(modality_tokens.keys())
            for source_mod in mod_names:
                for target_mod in mod_names:
                    if source_mod == target_mod:
                        continue
                    key = f"{source_mod}_to_{target_mod}"
                    if key not in self.predictors:
                        continue

                    # Student embeddings from source modality
                    s_emb = student_out.get(f"{source_mod}_embeddings")
                    # Teacher embeddings from target modality
                    t_emb = teacher_out.get(f"{target_mod}_embeddings")

                    if s_emb is None or t_emb is None:
                        continue

                    # Mean pool source → predict target (cell-level cross-modal)
                    s_pooled = s_emb.mean(dim=1)  # (batch, token_dim)
                    t_pooled = t_emb.mean(dim=1)  # (batch, token_dim)

                    pred = self.predictors[key](s_pooled)
                    loss = F.mse_loss(pred, t_pooled.detach())
                    losses[key] = loss.item()
                    total_loss = total_loss + loss

        # VICReg anti-collapse on cell embeddings
        if self.cfg.vicreg_weight > 0:
            cell_emb = student_out["cell_embedding"]
            vicreg = self._vicreg_loss(cell_emb)
            total_loss = total_loss + self.cfg.vicreg_weight * vicreg
            losses["vicreg"] = vicreg.item()

        losses["total"] = total_loss.item()

        return {
            "loss": total_loss,
            "losses": losses,
            "student_output": student_out,
            "teacher_output": teacher_out,
        }

    def _vicreg_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """VICReg variance + covariance regularization."""
        x = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Variance: std should be > 1
        std = x.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()

        # Covariance: off-diagonal should be 0
        n = x.shape[0]
        if n > 1:
            cov = (x.T @ x) / (n - 1)
            d = cov.shape[0]
            off_diag = cov.flatten()[1:].view(d - 1, d + 1)[:, :-1]
            cov_loss = (off_diag ** 2).mean()
        else:
            cov_loss = torch.tensor(0.0, device=embeddings.device)

        return var_loss + cov_loss
