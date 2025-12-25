"""Baseline evaluation utilities for M1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .metrics import cosine_distance, bootstrap_mean


@dataclass
class PairData:
    context_id: str
    perturbation_id: str
    control: np.ndarray
    perturbed: np.ndarray


def _mean_vector(matrix) -> np.ndarray:
    mean = matrix.mean(axis=0)
    if hasattr(mean, "A1"):
        return mean.A1  # sparse matrix mean returns matrix
    return np.asarray(mean).ravel()


def build_pairs(
    adata,
    max_cells_per_group: int | None = None,
    seed: int = 0,
) -> List[PairData]:
    rng = np.random.default_rng(seed)
    obs = adata.obs
    X = adata.X

    control_proto: Dict[str, np.ndarray] = {}
    pert_proto: Dict[Tuple[str, str], np.ndarray] = {}

    # Compute control prototypes per context
    for context_id, idx in obs[obs["is_control"]].groupby("context_id").indices.items():
        indices = np.array(idx)
        if indices.size == 0:
            continue
        if max_cells_per_group and indices.size > max_cells_per_group:
            indices = rng.choice(indices, size=max_cells_per_group, replace=False)
        control_proto[context_id] = _mean_vector(X[indices])

    # Compute perturbed prototypes per (context, perturbation)
    pert_obs = obs[~obs["is_control"]]
    for (context_id, perturbation_id), idx in pert_obs.groupby(["context_id", "perturbation_id"]).indices.items():
        indices = np.array(idx)
        if indices.size == 0:
            continue
        if max_cells_per_group and indices.size > max_cells_per_group:
            indices = rng.choice(indices, size=max_cells_per_group, replace=False)
        pert_proto[(context_id, perturbation_id)] = _mean_vector(X[indices])

    pairs: List[PairData] = []
    for (context_id, perturbation_id), y in pert_proto.items():
        x = control_proto.get(context_id)
        if x is None:
            continue
        pairs.append(
            PairData(
                context_id=str(context_id),
                perturbation_id=str(perturbation_id),
                control=x,
                perturbed=y,
            )
        )

    return pairs


def split_pairs(
    pairs: List[PairData],
    group_key: str,
    train_groups: Iterable[str],
    val_groups: Iterable[str],
    test_groups: Iterable[str],
):
    train_set, val_set, test_set = [], [], []
    train_groups = set(map(str, train_groups))
    val_groups = set(map(str, val_groups))
    test_groups = set(map(str, test_groups))

    for pair in pairs:
        group_val = pair.perturbation_id if group_key == "perturbation_id" else pair.context_id
        if group_val in test_groups:
            test_set.append(pair)
        elif group_val in val_groups:
            val_set.append(pair)
        else:
            train_set.append(pair)

    return train_set, val_set, test_set


def fit_pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    return mean, Vt[:k]


def transform_pca(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X - mean) @ components.T


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    XtX = Xb.T @ Xb
    XtX += alpha * np.eye(XtX.shape[0])
    W = np.linalg.solve(XtX, Xb.T @ Y)
    return W


def _ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W


def evaluate_baselines(
    pairs: List[PairData],
    group_key: str,
    train_groups: Iterable[str],
    val_groups: Iterable[str],
    test_groups: Iterable[str],
    pca_components: int = 50,
    ridge_alphas: Iterable[float] = (0.1, 1.0, 10.0, 100.0),
):
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, group_key, train_groups, val_groups, test_groups)

    def to_arrays(items: List[PairData]):
        X = np.stack([p.control for p in items])
        Y = np.stack([p.perturbed for p in items])
        P = [p.perturbation_id for p in items]
        return X, Y, P

    X_train, Y_train, P_train = to_arrays(train_pairs)
    X_val, Y_val, P_val = to_arrays(val_pairs) if val_pairs else (None, None, None)
    X_test, Y_test, P_test = to_arrays(test_pairs)

    mean, components = fit_pca(X_train, pca_components)
    X_train_p = transform_pca(X_train, mean, components)
    Y_train_p = transform_pca(Y_train, mean, components)
    X_test_p = transform_pca(X_test, mean, components)
    Y_test_p = transform_pca(Y_test, mean, components)

    # Baseline 1: no-change
    no_change_pred = X_test_p

    # Baseline 2: mean-shift per perturbation
    shift_by_pert: Dict[str, np.ndarray] = {}
    for pid in set(P_train):
        idx = [i for i, p in enumerate(P_train) if p == pid]
        if not idx:
            continue
        shift_by_pert[pid] = (Y_train_p[idx] - X_train_p[idx]).mean(axis=0)

    mean_shift_pred = np.zeros_like(X_test_p)
    for i, pid in enumerate(P_test):
        shift = shift_by_pert.get(pid)
        mean_shift_pred[i] = X_test_p[i] + shift if shift is not None else X_test_p[i]

    # Baseline 3: ridge regression (global)
    best_alpha = None
    best_mse = float("inf")
    best_W = None
    for alpha in ridge_alphas:
        W = _ridge_fit(X_train_p, Y_train_p, alpha)
        if X_val is not None and len(X_val) > 0:
            X_val_p = transform_pca(X_val, mean, components)
            Y_val_p = transform_pca(Y_val, mean, components)
            pred = _ridge_predict(X_val_p, W)
            mse = float(np.mean((pred - Y_val_p) ** 2))
        else:
            pred = _ridge_predict(X_train_p, W)
            mse = float(np.mean((pred - Y_train_p) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_W = W

    ridge_pred = _ridge_predict(X_test_p, best_W)

    def eval_metrics(pred: np.ndarray, target: np.ndarray):
        mse = np.mean((pred - target) ** 2, axis=1)
        cos = np.array([cosine_distance(pred[i], target[i]) for i in range(pred.shape[0])])
        mse_mean, mse_lo, mse_hi = bootstrap_mean(mse)
        cos_mean, cos_lo, cos_hi = bootstrap_mean(cos)
        return {
            "mse_mean": mse_mean,
            "mse_ci95": (mse_lo, mse_hi),
            "cosine_mean": cos_mean,
            "cosine_ci95": (cos_lo, cos_hi),
        }

    results = {
        "no_change": eval_metrics(no_change_pred, Y_test_p),
        "mean_shift": eval_metrics(mean_shift_pred, Y_test_p),
        "ridge_pca": eval_metrics(ridge_pred, Y_test_p),
        "ridge_alpha": best_alpha,
        "n_pairs_test": len(test_pairs),
        "n_pairs_train": len(train_pairs),
        "n_pairs_val": len(val_pairs),
    }
    return results
