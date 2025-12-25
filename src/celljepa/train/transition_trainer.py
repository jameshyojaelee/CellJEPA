"""Training utilities for transition predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from celljepa.models.transition import PrototypePredictor, SetPredictor


@dataclass
class PairProto:
    context_id: str
    perturbation_id: str
    control_proto: np.ndarray
    pert_proto: np.ndarray


@dataclass
class PairSet:
    context_id: str
    perturbation_id: str
    control_indices: np.ndarray
    pert_indices: np.ndarray


def energy_distance_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d_xy = torch.cdist(x, y).mean()
    d_xx = torch.cdist(x, x).mean()
    d_yy = torch.cdist(y, y).mean()
    return 2 * d_xy - d_xx - d_yy


def train_prototype(
    model: PrototypePredictor,
    optimizer: torch.optim.Optimizer,
    pairs: List[PairProto],
    pert_to_idx: Dict[str, int],
    device: str,
    epochs: int = 10,
    batch_size: int = 128,
) -> Dict[str, float]:
    model.train()
    losses = []
    skipped = 0

    for epoch in range(epochs):
        np.random.shuffle(pairs)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            if not batch:
                continue
            x = torch.tensor(np.stack([p.control_proto for p in batch]), dtype=torch.float32, device=device)
            y = torch.tensor(np.stack([p.pert_proto for p in batch]), dtype=torch.float32, device=device)
            idx = torch.tensor([pert_to_idx.get(p.perturbation_id, 0) for p in batch], device=device, dtype=torch.long)
            mask = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
            if mask.sum().item() == 0:
                skipped += len(batch)
                continue
            x = x[mask]
            y = y[mask]
            idx = idx[mask]
            pred = model(x, idx)
            if not torch.isfinite(pred).all():
                skipped += len(batch)
                continue
            loss = nn.functional.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return {"loss": float(np.mean(losses)) if losses else float("nan"), "skipped_train": skipped}


def train_set(
    model: SetPredictor,
    optimizer: torch.optim.Optimizer,
    pairs: List[PairSet],
    embeddings: np.ndarray,
    pert_to_idx: Dict[str, int],
    device: str,
    epochs: int = 10,
    sample_size: int = 128,
) -> Dict[str, float]:
    model.train()
    losses = []
    rng = np.random.default_rng(0)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for pair in pairs:
            c_idx = pair.control_indices
            p_idx = pair.pert_indices
            if c_idx.size == 0 or p_idx.size == 0:
                continue
            c_sel = rng.choice(c_idx, size=min(sample_size, c_idx.size), replace=False)
            p_sel = rng.choice(p_idx, size=min(sample_size, p_idx.size), replace=False)

            c = torch.tensor(embeddings[c_sel], dtype=torch.float32, device=device)
            y = torch.tensor(embeddings[p_sel], dtype=torch.float32, device=device)
            idx = torch.tensor([pert_to_idx.get(pair.perturbation_id, 0)], device=device, dtype=torch.long)
            pred = model(c, idx)
            loss = energy_distance_torch(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return {"loss": float(np.mean(losses)) if losses else float("nan")}
