"""Evaluation metrics for CellJEPA.

Embedding-level metrics (M0/v1):
  - cosine_distance, energy_distance, bootstrap_mean

Gene-level metrics (P3/v2):
  - lfc_pearson_correlation, top_k_deg_recall, direction_accuracy
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return 1.0 - float(np.dot(a, b) / denom)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute energy distance between two samples in embedding space."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("energy_distance expects 2D arrays: (n, d)")
    if x.shape[1] != y.shape[1]:
        raise ValueError("energy_distance expects matching feature dimensions")

    def pairwise_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        count = 0
        for i in range(a.shape[0]):
            diff = b - a[i]
            dists = np.sqrt(np.sum(diff * diff, axis=1))
            total += float(np.sum(dists))
            count += b.shape[0]
        return total / max(count, 1)

    d_xy = pairwise_mean_distance(x, y)
    d_xx = pairwise_mean_distance(x, x)
    d_yy = pairwise_mean_distance(y, y)

    return 2.0 * d_xy - d_xx - d_yy


def bootstrap_mean(values: Iterable[float], num_samples: int = 1000, seed: int = 0):
    """Simple bootstrap mean (stub)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(list(values))
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(num_samples):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    means = np.asarray(means)
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Gene-level metrics (P3)
# ---------------------------------------------------------------------------

def lfc_pearson_correlation(
    pred: np.ndarray,
    obs: np.ndarray,
) -> float:
    """Pearson correlation between predicted and observed per-gene LFCs.

    Args:
        pred: (n_genes,) predicted log-fold changes.
        obs: (n_genes,) observed log-fold changes.

    Returns:
        Pearson correlation coefficient. NaN if either vector has zero variance.
    """
    pred = np.asarray(pred).ravel()
    obs = np.asarray(obs).ravel()
    if pred.shape[0] != obs.shape[0]:
        raise ValueError(
            f"Shape mismatch: pred has {pred.shape[0]} genes, obs has {obs.shape[0]}"
        )
    if pred.shape[0] < 2:
        return float("nan")

    p_std = np.std(pred)
    o_std = np.std(obs)
    if p_std == 0 or o_std == 0:
        return float("nan")

    return float(np.corrcoef(pred, obs)[0, 1])


def top_k_deg_recall(
    pred: np.ndarray,
    obs: np.ndarray,
    k: int = 20,
) -> float:
    """Top-k DEG recall: fraction of true top-k DEGs in predicted top-k.

    DEGs are ranked by absolute LFC magnitude.

    Args:
        pred: (n_genes,) predicted log-fold changes.
        obs: (n_genes,) observed log-fold changes.
        k: number of top DEGs to consider.

    Returns:
        Recall ∈ [0, 1]. Returns NaN if fewer than k genes available.
    """
    pred = np.asarray(pred).ravel()
    obs = np.asarray(obs).ravel()
    if pred.shape[0] != obs.shape[0]:
        raise ValueError(
            f"Shape mismatch: pred has {pred.shape[0]} genes, obs has {obs.shape[0]}"
        )
    n = pred.shape[0]
    if n < k:
        return float("nan")

    # Top-k by absolute magnitude
    true_top_k = set(np.argsort(np.abs(obs))[-k:])
    pred_top_k = set(np.argsort(np.abs(pred))[-k:])

    overlap = len(true_top_k & pred_top_k)
    return float(overlap / k)


def direction_accuracy(
    pred: np.ndarray,
    obs: np.ndarray,
) -> float:
    """Direction accuracy: fraction of genes with correct up/down sign.

    Genes with observed LFC = 0 are excluded from the calculation.

    Args:
        pred: (n_genes,) predicted log-fold changes.
        obs: (n_genes,) observed log-fold changes.

    Returns:
        Accuracy ∈ [0, 1]. NaN if no non-zero observed genes.
    """
    pred = np.asarray(pred).ravel()
    obs = np.asarray(obs).ravel()
    if pred.shape[0] != obs.shape[0]:
        raise ValueError(
            f"Shape mismatch: pred has {pred.shape[0]} genes, obs has {obs.shape[0]}"
        )

    # Exclude genes with zero observed LFC
    nonzero = obs != 0
    if not np.any(nonzero):
        return float("nan")

    pred_sign = np.sign(pred[nonzero])
    obs_sign = np.sign(obs[nonzero])

    return float(np.mean(pred_sign == obs_sign))

