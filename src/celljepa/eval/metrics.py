"""Evaluation metrics for CellJEPA.

Embedding-level metrics (M0/v1):
  - cosine_distance, energy_distance, bootstrap_mean

Gene-level metrics (P3/v2):
  - lfc_pearson_correlation, top_k_deg_recall, direction_accuracy

Benchmark metrics (P5):
  - perturbench_rank_metric, knn_retrieval_accuracy,
    mean_reciprocal_rank, calibrated_energy_distance
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


# ---------------------------------------------------------------------------
# Benchmark metrics (P5)
# ---------------------------------------------------------------------------

def perturbench_rank_metric(
    predicted_embeddings: np.ndarray,
    true_embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """PerturBench-style rank metric: how well are perturbation effects ordered?

    For each perturbation, compute the rank of the true match among all
    candidates by embedding distance. Returns mean normalized rank
    (0 = perfect, 1 = worst).

    Args:
        predicted_embeddings: (N, D) predicted perturbation effect embeddings.
        true_embeddings: (N, D) ground-truth perturbation effect embeddings.
        labels: (N,) perturbation labels for grouping.

    Returns:
        Mean normalized rank ∈ [0, 1]. Lower is better.
    """
    predicted_embeddings = np.asarray(predicted_embeddings)
    true_embeddings = np.asarray(true_embeddings)
    labels = np.asarray(labels)
    n = predicted_embeddings.shape[0]
    if n < 2:
        return float("nan")

    ranks = []
    for i in range(n):
        # Distance from prediction i to all true embeddings
        dists = np.linalg.norm(true_embeddings - predicted_embeddings[i], axis=1)
        # Rank of the correct match (0-indexed)
        sorted_idx = np.argsort(dists)
        rank = int(np.where(sorted_idx == i)[0][0])
        # Normalize: 0 = correct match is closest, 1 = farthest
        ranks.append(rank / (n - 1))

    return float(np.mean(ranks))


def knn_retrieval_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> float:
    """kNN retrieval accuracy: fraction of k nearest neighbors sharing label.

    Args:
        embeddings: (N, D) embedding vectors.
        labels: (N,) perturbation labels.
        k: number of neighbors.

    Returns:
        Mean accuracy ∈ [0, 1].
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    n = embeddings.shape[0]
    if n < k + 1:
        return float("nan")

    # Pairwise distances
    # Use broadcasting: (N, 1, D) - (1, N, D) → (N, N)
    dists = np.linalg.norm(
        embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :], axis=2
    )

    accuracies = []
    for i in range(n):
        # Exclude self (set self-distance to inf)
        d = dists[i].copy()
        d[i] = np.inf
        neighbors = np.argsort(d)[:k]
        same_label = np.sum(labels[neighbors] == labels[i])
        accuracies.append(same_label / k)

    return float(np.mean(accuracies))


def mean_reciprocal_rank(
    predicted_embeddings: np.ndarray,
    true_embeddings: np.ndarray,
) -> float:
    """Mean Reciprocal Rank: how quickly the correct match is found.

    For each predicted embedding, find the rank of the true match among
    all candidates (by L2 distance). MRR = mean(1/rank).

    Args:
        predicted_embeddings: (N, D) predicted embeddings.
        true_embeddings: (N, D) true embeddings (one-to-one correspondence).

    Returns:
        MRR ∈ (0, 1]. Higher is better.
    """
    predicted_embeddings = np.asarray(predicted_embeddings)
    true_embeddings = np.asarray(true_embeddings)
    n = predicted_embeddings.shape[0]
    if n < 1:
        return float("nan")

    reciprocal_ranks = []
    for i in range(n):
        dists = np.linalg.norm(true_embeddings - predicted_embeddings[i], axis=1)
        sorted_idx = np.argsort(dists)
        rank = int(np.where(sorted_idx == i)[0][0]) + 1  # 1-indexed
        reciprocal_ranks.append(1.0 / rank)

    return float(np.mean(reciprocal_ranks))


def calibrated_energy_distance(
    pred: np.ndarray,
    obs: np.ndarray,
    n_permutations: int = 100,
    seed: int = 42,
) -> float:
    """Calibrated E-distance: E-distance normalized by permuted null.

    Computes E-distance between predicted and observed, then divides by
    the mean E-distance under random permutations of the labels.
    Values < 1 mean predictions are better than chance.

    Args:
        pred: (N, D) predicted embeddings.
        obs: (N, D) observed embeddings.
        n_permutations: number of permutation samples for null.
        seed: random seed.

    Returns:
        Calibrated E-distance ratio. Lower is better; < 1 beats null.
    """
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    if pred.shape[0] < 2 or obs.shape[0] < 2:
        return float("nan")

    actual_ed = energy_distance(pred, obs)

    rng = np.random.default_rng(seed)
    null_eds = []
    combined = np.vstack([pred, obs])
    n_pred = pred.shape[0]
    for _ in range(n_permutations):
        perm = rng.permutation(combined.shape[0])
        null_pred = combined[perm[:n_pred]]
        null_obs = combined[perm[n_pred:]]
        null_eds.append(energy_distance(null_pred, null_obs))

    mean_null = float(np.mean(null_eds))
    if mean_null == 0:
        return float("nan")

    return actual_ed / mean_null

