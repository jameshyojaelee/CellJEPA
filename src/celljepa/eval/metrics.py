"""Metric stubs for M0 (simple, dependency-free implementations)."""

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

