#!/usr/bin/env python3
"""Train transition predictors in embedding space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import anndata as ad
import numpy as np
import torch

from celljepa.models.jepa import JEPA, JepaConfig
from celljepa.models.transition import PrototypePredictor, SetPredictor, TransitionConfig
from celljepa.train.transition_trainer import PairProto, PairSet, train_prototype, train_set, energy_distance_torch
from celljepa.eval.metrics import cosine_distance, bootstrap_mean
from celljepa.utils.attempt_log import append_attempt


def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def embed_cells(adata, checkpoint_path: Path, indices: np.ndarray, batch_size: int = 512, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = JepaConfig(**ckpt["config"])
    model = JEPA(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    X = adata.X
    indices = np.asarray(indices)
    out = np.zeros((indices.size, cfg.embed_dim), dtype=np.float32)

    for i in range(0, indices.size, batch_size):
        batch_idx = indices[i : i + batch_size]
        x = _to_dense(X[batch_idx]).astype(np.float32)
        with torch.no_grad():
            z = model.student(torch.from_numpy(x).to(device)).cpu().numpy()
        out[i : i + batch_idx.size] = z
    return out


def build_pairs(
    adata,
    max_cells_per_group: int | None = None,
    min_cells_per_condition: int | None = None,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    obs = adata.obs
    control_groups = {}
    pert_groups = {}
    for context_id, idx in obs[obs["is_control"]].groupby("context_id").indices.items():
        control_groups[context_id] = np.array(idx)
    for (context_id, perturbation_id), idx in obs[~obs["is_control"]].groupby(["context_id", "perturbation_id"]).indices.items():
        pert_groups[(context_id, perturbation_id)] = np.array(idx, dtype=np.int64)

    pairs_set = []
    for (context_id, perturbation_id), p_idx in pert_groups.items():
        pid_str = str(perturbation_id).strip().lower()
        if pid_str in {"nan", "none", ""}:
            continue
        c_idx = control_groups.get(context_id)
        if c_idx is None or p_idx.size == 0:
            continue
        if min_cells_per_condition:
            if c_idx.size < min_cells_per_condition or p_idx.size < min_cells_per_condition:
                continue
        if max_cells_per_group:
            if c_idx.size > max_cells_per_group:
                c_idx = rng.choice(c_idx, size=max_cells_per_group, replace=False)
            if p_idx.size > max_cells_per_group:
                p_idx = rng.choice(p_idx, size=max_cells_per_group, replace=False)
        pairs_set.append(
            PairSet(
                context_id=str(context_id),
                perturbation_id=str(perturbation_id),
                control_indices=np.array(c_idx, dtype=np.int64),
                pert_indices=np.array(p_idx, dtype=np.int64),
            )
        )
    pairs_set.sort(key=lambda p: (p.context_id, p.perturbation_id))
    return pairs_set


def split_pairs(pairs, group_key, split):
    train_groups = set(map(str, split["train_groups"]))
    val_groups = set(map(str, split["val_groups"]))
    test_groups = set(map(str, split["test_groups"]))

    train, val, test = [], [], []
    for p in pairs:
        group_val = p.perturbation_id if group_key == "perturbation_id" else p.context_id
        if group_val in test_groups:
            test.append(p)
        elif group_val in val_groups:
            val.append(p)
        else:
            train.append(p)
    return train, val, test


def _summarize(values: list[float], bootstrap_samples: int, bootstrap_seed: int) -> tuple[float, tuple[float, float], int]:
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float("nan"), (float("nan"), float("nan")), 0
    mean, lo, hi = bootstrap_mean(values, num_samples=bootstrap_samples, seed=bootstrap_seed)
    return mean, (lo, hi), len(values)


def _sample_indices(rng: np.random.Generator, indices: np.ndarray, n: int) -> np.ndarray | None:
    if indices.size == 0 or n <= 0:
        return None
    if n > indices.size:
        n = indices.size
    return rng.choice(indices, size=n, replace=False)


def _resampled_edist(
    control_idx: np.ndarray,
    pert_idx: np.ndarray,
    embeddings: np.ndarray,
    rng: np.random.Generator,
    sample_size: int,
    resamples: int,
    device: str,
    pred_fn,
) -> list[float]:
    if control_idx.size == 0 or pert_idx.size == 0:
        return []
    n = min(sample_size, control_idx.size, pert_idx.size)
    if n <= 0:
        return []
    edists: list[float] = []
    for _ in range(resamples):
        c_sel = _sample_indices(rng, control_idx, n)
        t_sel = _sample_indices(rng, pert_idx, n)
        if c_sel is None or t_sel is None:
            continue
        c = torch.tensor(embeddings[c_sel], dtype=torch.float32, device=device)
        y = torch.tensor(embeddings[t_sel], dtype=torch.float32, device=device)
        if not torch.isfinite(c).all() or not torch.isfinite(y).all():
            continue
        pred = pred_fn(c)
        if not torch.isfinite(pred).all():
            continue
        loss = energy_distance_torch(pred, y)
        loss_val = float(loss.detach().cpu().numpy())
        if not np.isfinite(loss_val):
            continue
        edists.append(loss_val)
    return edists


def build_proto_pairs(pairs: list[PairSet], embeddings: np.ndarray) -> tuple[list[PairProto], int]:
    pairs_proto: list[PairProto] = []
    skipped = 0
    for p in pairs:
        c = embeddings[p.control_indices]
        y = embeddings[p.pert_indices]
        if c.size == 0 or y.size == 0:
            skipped += 1
            continue
        c_mean = np.mean(c, axis=0)
        y_mean = np.mean(y, axis=0)
        if not np.isfinite(c_mean).all() or not np.isfinite(y_mean).all():
            skipped += 1
            continue
        pairs_proto.append(PairProto(p.context_id, p.perturbation_id, c_mean, y_mean))
    return pairs_proto, skipped


def eval_prototype_model(
    model: PrototypePredictor,
    pairs: list[PairProto],
    pert_to_idx: dict[str, int],
    device: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    mse_vals: list[float] = []
    cos_vals: list[float] = []
    skipped = 0
    model.eval()
    with torch.no_grad():
        for p in pairs:
            if not np.isfinite(p.control_proto).all() or not np.isfinite(p.pert_proto).all():
                skipped += 1
                continue
            control = torch.tensor(p.control_proto[None, :], dtype=torch.float32, device=device)
            idx = torch.tensor([pert_to_idx.get(p.perturbation_id, 0)], device=device, dtype=torch.long)
            pred = model(control, idx).cpu().numpy()[0]
            if not np.isfinite(pred).all():
                skipped += 1
                continue
            mse_val = float(np.mean((pred - p.pert_proto) ** 2))
            cos_val = cosine_distance(pred, p.pert_proto)
            if not np.isfinite(mse_val) or not np.isfinite(cos_val):
                skipped += 1
                continue
            mse_vals.append(mse_val)
            cos_vals.append(cos_val)
    mse_mean, mse_ci, n_eval = _summarize(mse_vals, bootstrap_samples, bootstrap_seed)
    cos_mean, cos_ci, _ = _summarize(cos_vals, bootstrap_samples, bootstrap_seed)
    return {
        "mse_mean": mse_mean,
        "mse_ci95": mse_ci,
        "cosine_mean": cos_mean,
        "cosine_ci95": cos_ci,
        "skipped_pairs": skipped,
        "n_eval": n_eval,
    }


def eval_set_model(
    model: SetPredictor,
    pairs: list[PairSet],
    embeddings: np.ndarray,
    pert_to_idx: dict[str, int],
    device: str,
    sample_size: int,
    resamples: int,
    seed: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    per_pair: list[float] = []
    skipped = 0
    model.eval()
    for p in pairs:
        idx = torch.tensor([pert_to_idx.get(p.perturbation_id, 0)], device=device, dtype=torch.long)

        def pred_fn(c):
            with torch.no_grad():
                return model(c, idx)

        edists = _resampled_edist(
            p.control_indices,
            p.pert_indices,
            embeddings,
            rng,
            sample_size,
            resamples,
            device,
            pred_fn,
        )
        if not edists:
            skipped += 1
            continue
        per_pair.append(float(np.mean(edists)))

    edist_mean, edist_ci, n_eval = _summarize(per_pair, bootstrap_samples, bootstrap_seed)
    return {
        "edist_mean": edist_mean,
        "edist_ci95": edist_ci,
        "skipped_pairs": skipped,
        "n_eval": n_eval,
    }


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    XtX = Xb.T @ Xb
    XtX += alpha * np.eye(XtX.shape[0])
    W = np.linalg.solve(XtX, Xb.T @ Y)
    return W


def _ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W


def _mean_shift_by_pert(train_proto: list[PairProto]) -> dict[str, np.ndarray]:
    shifts: dict[str, list[np.ndarray]] = {}
    for p in train_proto:
        shifts.setdefault(p.perturbation_id, []).append(p.pert_proto - p.control_proto)
    shift_by_pert = {}
    for pid, vals in shifts.items():
        shift_by_pert[pid] = np.mean(np.stack(vals), axis=0)
    return shift_by_pert


def _fit_ridge_proto(
    train_proto: list[PairProto],
    val_proto: list[PairProto],
    ridge_alphas: list[float],
) -> tuple[np.ndarray, float]:
    if not train_proto:
        raise ValueError("No training pairs available for ridge baseline.")
    X_train = np.stack([p.control_proto for p in train_proto])
    Y_train = np.stack([p.pert_proto for p in train_proto])
    if val_proto:
        X_val = np.stack([p.control_proto for p in val_proto])
        Y_val = np.stack([p.pert_proto for p in val_proto])
    else:
        X_val, Y_val = None, None

    best_alpha = ridge_alphas[0]
    best_score = float("inf")
    best_W = None
    for alpha in ridge_alphas:
        W = _ridge_fit(X_train, Y_train, alpha)
        if X_val is not None and X_val.size > 0:
            pred = _ridge_predict(X_val, W)
            mse = float(np.mean((pred - Y_val) ** 2))
        else:
            pred = _ridge_predict(X_train, W)
            mse = float(np.mean((pred - Y_train) ** 2))
        if mse < best_score:
            best_score = mse
            best_alpha = alpha
            best_W = W
    return best_W, best_alpha


def evaluate_set_baselines(
    train_proto: list[PairProto],
    val_proto: list[PairProto],
    test_pairs: list[PairSet],
    test_proto_map: dict[tuple[str, str], PairProto],
    embeddings: np.ndarray,
    sample_size: int,
    resamples: int,
    seed: int,
    ridge_alphas: list[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
    device: str,
) -> dict:
    shift_by_pert = _mean_shift_by_pert(train_proto) if train_proto else {}
    ridge_W, ridge_alpha = _fit_ridge_proto(train_proto, val_proto, ridge_alphas) if train_proto else (None, float("nan"))

    def eval_shift(name: str, shift_fn, rng_seed: int):
        rng = np.random.default_rng(rng_seed)
        per_pair: list[float] = []
        skipped = 0
        for p in test_pairs:
            proto = test_proto_map.get((p.context_id, p.perturbation_id))
            if proto is None:
                skipped += 1
                continue
            shift = shift_fn(proto)
            if shift is None or not np.isfinite(shift).all():
                skipped += 1
                continue
            shift_t = torch.tensor(shift, dtype=torch.float32, device=device)

            def pred_fn(c):
                return c + shift_t

            edists = _resampled_edist(
                p.control_indices,
                p.pert_indices,
                embeddings,
                rng,
                sample_size,
                resamples,
                device,
                pred_fn,
            )
            if not edists:
                skipped += 1
                continue
            per_pair.append(float(np.mean(edists)))
        mean, ci, n_eval = _summarize(per_pair, bootstrap_samples, bootstrap_seed)
        return {
            "edist_mean": mean,
            "edist_ci95": ci,
            "skipped_pairs": skipped,
            "n_eval": n_eval,
        }

    results = {}
    results["no_change"] = eval_shift("no_change", lambda proto: np.zeros_like(proto.control_proto), seed + 1)
    results["mean_shift"] = eval_shift(
        "mean_shift",
        lambda proto: shift_by_pert.get(proto.perturbation_id, np.zeros_like(proto.control_proto)),
        seed + 2,
    ) if train_proto else {}

    def ridge_shift(proto: PairProto):
        if ridge_W is None:
            return None
        pred = _ridge_predict(proto.control_proto[None, :], ridge_W)[0]
        return pred - proto.control_proto

    results["ridge"] = eval_shift("ridge", ridge_shift, seed + 3) if train_proto else {}
    results["ridge_alpha"] = ridge_alpha
    return results


def build_baseline_shift_map(
    baseline: str,
    train_proto: list[PairProto],
    val_proto: list[PairProto],
    all_proto: list[PairProto],
    ridge_alphas: list[float],
) -> tuple[dict[tuple[str, str], np.ndarray], dict]:
    if baseline == "none":
        return {}, {}

    meta: dict = {}
    if baseline == "no_change":
        shift_map = {(p.context_id, p.perturbation_id): np.zeros_like(p.control_proto) for p in all_proto}
        return shift_map, meta

    if not train_proto:
        raise ValueError("No training pairs available for baseline residual.")

    if baseline == "mean_shift":
        shift_by_pert = _mean_shift_by_pert(train_proto)
        shift_map = {}
        for p in all_proto:
            shift = shift_by_pert.get(p.perturbation_id)
            if shift is None:
                shift = np.zeros_like(p.control_proto)
            shift_map[(p.context_id, p.perturbation_id)] = shift
        return shift_map, meta

    if baseline == "ridge":
        ridge_W, ridge_alpha = _fit_ridge_proto(train_proto, val_proto, ridge_alphas)
        meta["ridge_alpha"] = ridge_alpha
        shift_map = {}
        for p in all_proto:
            pred = _ridge_predict(p.control_proto[None, :], ridge_W)[0]
            shift_map[(p.context_id, p.perturbation_id)] = pred - p.control_proto
        return shift_map, meta

    raise ValueError(f"Unknown residual baseline: {baseline}")


def train_set_residual(
    model: SetPredictor,
    optimizer: torch.optim.Optimizer,
    pairs: list[PairSet],
    embeddings: np.ndarray,
    pert_to_idx: dict[str, int],
    baseline_shift_map: dict[tuple[str, str], np.ndarray],
    device: str,
    epochs: int,
    sample_size: int,
    seed: int,
) -> dict:
    model.train()
    losses = []
    skipped = 0
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for pair in pairs:
            shift = baseline_shift_map.get((pair.context_id, pair.perturbation_id))
            if shift is None:
                skipped += 1
                continue
            c_idx = pair.control_indices
            p_idx = pair.pert_indices
            if c_idx.size == 0 or p_idx.size == 0:
                skipped += 1
                continue
            c_sel = rng.choice(c_idx, size=min(sample_size, c_idx.size), replace=False)
            p_sel = rng.choice(p_idx, size=min(sample_size, p_idx.size), replace=False)

            c = torch.tensor(embeddings[c_sel], dtype=torch.float32, device=device)
            y = torch.tensor(embeddings[p_sel], dtype=torch.float32, device=device)
            idx = torch.tensor([pert_to_idx.get(pair.perturbation_id, 0)], device=device, dtype=torch.long)
            shift_t = torch.tensor(shift, dtype=torch.float32, device=device)
            pred = c + shift_t
            pred = pred + model(c, idx)
            loss = energy_distance_torch(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return {"loss": float(np.mean(losses)) if losses else float("nan"), "skipped_train": skipped}


def _compute_residual_edists(
    model: SetPredictor,
    pairs: list[PairSet],
    embeddings: np.ndarray,
    pert_to_idx: dict[str, int],
    baseline_shift_map: dict[tuple[str, str], np.ndarray],
    device: str,
    sample_size: int,
    resamples: int,
    seed: int,
    alpha: float,
) -> tuple[list[float], int]:
    rng = np.random.default_rng(seed)
    per_pair: list[float] = []
    skipped = 0
    model.eval()
    for p in pairs:
        shift = baseline_shift_map.get((p.context_id, p.perturbation_id))
        if shift is None:
            skipped += 1
            continue
        idx = torch.tensor([pert_to_idx.get(p.perturbation_id, 0)], device=device, dtype=torch.long)
        shift_t = torch.tensor(shift, dtype=torch.float32, device=device)

        def pred_fn(c):
            with torch.no_grad():
                return c + shift_t + alpha * model(c, idx)

        edists = _resampled_edist(
            p.control_indices,
            p.pert_indices,
            embeddings,
            rng,
            sample_size,
            resamples,
            device,
            pred_fn,
        )
        if not edists:
            skipped += 1
            continue
        per_pair.append(float(np.mean(edists)))
    return per_pair, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transition predictors (prototype or set).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["prototype", "set"], default="prototype")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--eval-sample-size", type=int, default=None)
    parser.add_argument("--eval-resamples", type=int, default=5)
    parser.add_argument("--max-cells-per-group", type=int, default=5000)
    parser.add_argument("--min-cells-per-condition", type=int, default=30)
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on number of condition pairs.")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--eval-baselines", action="store_true", help="Compute set-level baseline E-distance metrics.")
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0,100.0")
    parser.add_argument("--residual-baseline", choices=["none", "no_change", "mean_shift", "ridge"], default="none")
    parser.add_argument("--residual-alpha-grid", type=str, default="0,0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    eval_sample_size = args.eval_sample_size or args.sample_size
    ridge_alphas = [float(x) for x in args.ridge_alphas.split(",") if x]
    residual_alpha_grid = [float(x) for x in args.residual_alpha_grid.split(",") if x]

    adata = ad.read_h5ad(args.dataset)
    split = json.loads(Path(args.split).read_text())
    group_key = split["group_key"]
    dataset_id = split.get("dataset_id") or adata.uns.get("dataset_id") or Path(args.dataset).stem
    split_name = split.get("split_name") or Path(args.split).stem

    pairs_set = build_pairs(
        adata,
        max_cells_per_group=args.max_cells_per_group,
        min_cells_per_condition=args.min_cells_per_condition,
        seed=args.seed,
    )
    if args.max_pairs and len(pairs_set) > args.max_pairs:
        rng = np.random.default_rng(args.seed)
        pairs_set = list(rng.choice(pairs_set, size=args.max_pairs, replace=False))

    # Collect indices to embed
    if not pairs_set:
        raise ValueError("No condition pairs available after filtering.")
    indices = np.unique(np.concatenate([p.control_indices for p in pairs_set] + [p.pert_indices for p in pairs_set]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = embed_cells(adata, Path(args.checkpoint), indices, device=device)
    idx_map = {idx: i for i, idx in enumerate(indices)}

    # remap to embedding indices
    for p in pairs_set:
        p.control_indices = np.array([idx_map[i] for i in p.control_indices if i in idx_map], dtype=np.int64)
        p.pert_indices = np.array([idx_map[i] for i in p.pert_indices if i in idx_map], dtype=np.int64)

    finite_mask = np.isfinite(embeddings).all(axis=1)
    nonfinite_cells = int(np.sum(~finite_mask))
    filtered_pairs = []
    dropped_nonfinite = 0
    for p in pairs_set:
        c_idx = p.control_indices[finite_mask[p.control_indices]]
        t_idx = p.pert_indices[finite_mask[p.pert_indices]]
        if c_idx.size == 0 or t_idx.size == 0:
            dropped_nonfinite += 1
            continue
        if args.min_cells_per_condition:
            if c_idx.size < args.min_cells_per_condition or t_idx.size < args.min_cells_per_condition:
                dropped_nonfinite += 1
                continue
        p.control_indices = c_idx
        p.pert_indices = t_idx
        filtered_pairs.append(p)
    pairs_set = filtered_pairs

    train_pairs, val_pairs, test_pairs = split_pairs(pairs_set, group_key, split)

    # Build perturbation vocab from train only
    train_perturbations = sorted({p.perturbation_id for p in train_pairs})
    pert_to_idx = {"<UNK>": 0}
    for i, p in enumerate(train_perturbations, 1):
        pert_to_idx[p] = i

    ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
    jepa_cfg = ckpt_meta.get("config", {})
    embed_dim = jepa_cfg.get("embed_dim", args.embed_dim)
    if embed_dim != args.embed_dim:
        print(f"Warning: overriding embed_dim={args.embed_dim} with checkpoint embed_dim={embed_dim}.")

    cfg = TransitionConfig(embed_dim=embed_dim, perturbation_vocab=len(pert_to_idx), hidden_dim=args.hidden_dim)

    train_proto, skipped_train_proto = build_proto_pairs(train_pairs, embeddings)
    val_proto, skipped_val_proto = build_proto_pairs(val_pairs, embeddings)
    test_proto, skipped_test_proto = build_proto_pairs(test_pairs, embeddings)
    test_proto_map = {(p.context_id, p.perturbation_id): p for p in test_proto}

    metrics = {
        "mode": args.mode,
        "dataset_id": dataset_id,
        "split_name": split_name,
        "group_key": group_key,
        "seed": args.seed,
        "min_cells_per_condition": args.min_cells_per_condition,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_pairs),
        "pairs_total": len(pairs_set),
        "nonfinite_cells": nonfinite_cells,
        "pairs_dropped_nonfinite": dropped_nonfinite,
        "proto_skipped": {
            "train": skipped_train_proto,
            "val": skipped_val_proto,
            "test": skipped_test_proto,
        },
        "eval_config": {
            "sample_size": eval_sample_size,
            "resamples": args.eval_resamples,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "residual_config": {
            "baseline": args.residual_baseline,
            "alpha_grid": residual_alpha_grid,
        },
        "jepa_config": jepa_cfg,
    }
    cfg_path = Path(args.checkpoint).with_name("config.json")
    if cfg_path.exists():
        try:
            metrics["jepa_run_config"] = json.loads(cfg_path.read_text())
        except Exception:
            metrics["jepa_run_config"] = None
    if args.mode == "prototype":
        model = PrototypePredictor(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        metrics.update(train_prototype(model, opt, train_proto, pert_to_idx, device, epochs=args.epochs, batch_size=args.batch_size))
        metrics["test"] = eval_prototype_model(
            model,
            test_proto,
            pert_to_idx,
            device,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        torch.save({"model": model.state_dict(), "pert_to_idx": pert_to_idx, "config": cfg.__dict__}, out_dir / "model.pt")
    else:
        model = SetPredictor(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        if args.residual_baseline != "none":
            all_proto = train_proto + val_proto + test_proto
            baseline_shift_map, baseline_meta = build_baseline_shift_map(
                args.residual_baseline,
                train_proto,
                val_proto,
                all_proto,
                ridge_alphas,
            )
            metrics["residual"] = {
                "baseline": args.residual_baseline,
                "alpha_grid": residual_alpha_grid,
                **baseline_meta,
            }
            metrics.update(
                train_set_residual(
                    model,
                    opt,
                    train_pairs,
                    embeddings,
                    pert_to_idx,
                    baseline_shift_map,
                    device,
                    epochs=args.epochs,
                    sample_size=args.sample_size,
                    seed=args.seed,
                )
            )
            if val_pairs:
                best_alpha = None
                best_val = float("inf")
                val_scores = {}
                for alpha in residual_alpha_grid:
                    vals, skipped = _compute_residual_edists(
                        model,
                        val_pairs,
                        embeddings,
                        pert_to_idx,
                        baseline_shift_map,
                        device,
                        eval_sample_size,
                        args.eval_resamples,
                        args.seed + 13,
                        alpha,
                    )
                    mean_val = float(np.mean(vals)) if vals else float("nan")
                    val_scores[alpha] = mean_val
                    if np.isfinite(mean_val) and mean_val < best_val:
                        best_val = mean_val
                        best_alpha = alpha
                if best_alpha is None:
                    best_alpha = 1.0
                    best_val = float("nan")
                metrics["residual"]["alpha"] = best_alpha
                metrics["residual"]["val_edist"] = best_val
                metrics["residual"]["val_scores"] = val_scores
            else:
                metrics["residual"]["alpha"] = 1.0

            alpha = metrics["residual"]["alpha"]
            per_pair, skipped = _compute_residual_edists(
                model,
                test_pairs,
                embeddings,
                pert_to_idx,
                baseline_shift_map,
                device,
                eval_sample_size,
                args.eval_resamples,
                args.seed + 17,
                alpha,
            )
            edist_mean, edist_ci, n_eval = _summarize(per_pair, args.bootstrap_samples, args.bootstrap_seed)
            metrics["test"] = {
                "edist_mean": edist_mean,
                "edist_ci95": edist_ci,
                "skipped_pairs": skipped,
                "n_eval": n_eval,
            }
        else:
            metrics.update(
                train_set(
                    model,
                    opt,
                    train_pairs,
                    embeddings,
                    pert_to_idx,
                    device,
                    epochs=args.epochs,
                    sample_size=args.sample_size,
                    seed=args.seed,
                )
            )
            metrics["test"] = eval_set_model(
                model,
                test_pairs,
                embeddings,
                pert_to_idx,
                device,
                eval_sample_size,
                args.eval_resamples,
                args.seed,
                args.bootstrap_samples,
                args.bootstrap_seed,
            )

        if args.eval_baselines:
            metrics["baselines"] = evaluate_set_baselines(
                train_proto,
                val_proto,
                test_pairs,
                test_proto_map,
                embeddings,
                eval_sample_size,
                args.eval_resamples,
                args.seed,
                ridge_alphas,
                args.bootstrap_samples,
                args.bootstrap_seed,
                device,
            )
        torch.save({"model": model.state_dict(), "pert_to_idx": pert_to_idx, "config": cfg.__dict__}, out_dir / "model.pt")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {out_dir / 'metrics.json'}")
    try:
        append_attempt(
            {
                "script": "train_transition",
                "run_dir": str(out_dir),
                "dataset_id": dataset_id,
                "split_name": split_name,
                "mode": args.mode,
                "seed": args.seed,
                "test": metrics.get("test"),
                "baselines": metrics.get("baselines"),
                "residual": metrics.get("residual"),
            }
        )
    except Exception as exc:
        print(f"Attempt log skipped: {exc}")


if __name__ == "__main__":
    main()
