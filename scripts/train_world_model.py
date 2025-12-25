#!/usr/bin/env python3
"""Train an action-conditioned set-to-set world model in embedding space."""

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
from celljepa.models.world_model import WorldModel, WorldModelConfig
from celljepa.train.transition_trainer import PairSet, energy_distance_torch
from celljepa.eval.metrics import bootstrap_mean
from celljepa.utils.attempt_log import append_attempt


def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def load_action_embeddings(path: Path) -> tuple[dict[str, np.ndarray], int]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("Action embeddings JSON must be a non-empty mapping.")
    action_map: dict[str, np.ndarray] = {}
    dim = None
    for key, value in data.items():
        vec = np.asarray(value, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"Action embedding for {key} must be a 1D vector.")
        if dim is None:
            dim = vec.shape[0]
        elif vec.shape[0] != dim:
            raise ValueError(f"Action embedding for {key} has dim {vec.shape[0]} != {dim}.")
        action_map[str(key)] = vec
    if dim is None:
        raise ValueError("Action embeddings JSON had no vectors.")
    return action_map, dim


def apply_action_embeddings(
    model: WorldModel,
    pert_to_idx: dict[str, int],
    action_map: dict[str, np.ndarray],
) -> None:
    weight = model.action_emb.emb.weight
    with torch.no_grad():
        weight[0].zero_()
        for key, vec in action_map.items():
            idx = pert_to_idx.get(key)
            if idx is None:
                continue
            if vec.shape[0] != weight.shape[1]:
                raise ValueError(f"Action embedding for {key} has dim {vec.shape[0]} != {weight.shape[1]}.")
            weight[idx].copy_(torch.tensor(vec, dtype=weight.dtype, device=weight.device))


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
    return out, ckpt.get("config", {})


def build_pairs(
    adata,
    max_cells_per_group: int | None,
    min_cells_per_condition: int,
    seed: int,
) -> list[PairSet]:
    rng = np.random.default_rng(seed)
    obs = adata.obs
    control_groups = {}
    pert_groups = {}
    for context_id, idx in obs[obs["is_control"]].groupby("context_id").indices.items():
        control_groups[str(context_id)] = np.array(idx, dtype=np.int64)
    for (context_id, perturbation_id), idx in obs[~obs["is_control"]].groupby(["context_id", "perturbation_id"]).indices.items():
        pert_groups[(str(context_id), str(perturbation_id))] = np.array(idx, dtype=np.int64)

    pairs = []
    for (context_id, perturbation_id), p_idx in pert_groups.items():
        pid_str = str(perturbation_id).strip().lower()
        if pid_str in {"nan", "none", ""}:
            continue
        c_idx = control_groups.get(context_id)
        if c_idx is None or p_idx.size == 0:
            continue
        if c_idx.size < min_cells_per_condition or p_idx.size < min_cells_per_condition:
            continue
        if max_cells_per_group:
            if c_idx.size > max_cells_per_group:
                c_idx = rng.choice(c_idx, size=max_cells_per_group, replace=False)
            if p_idx.size > max_cells_per_group:
                p_idx = rng.choice(p_idx, size=max_cells_per_group, replace=False)
        pairs.append(
            PairSet(
                context_id=context_id,
                perturbation_id=perturbation_id,
                control_indices=np.array(c_idx, dtype=np.int64),
                pert_indices=np.array(p_idx, dtype=np.int64),
            )
        )
    pairs.sort(key=lambda p: (p.context_id, p.perturbation_id))
    return pairs


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


def build_proto_pairs(pairs: list[PairSet], embeddings: np.ndarray):
    pairs_proto = []
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
        pairs_proto.append((p.context_id, p.perturbation_id, c_mean, y_mean))
    return pairs_proto, skipped


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    XtX = Xb.T @ Xb
    XtX += alpha * np.eye(XtX.shape[0])
    W = np.linalg.solve(XtX, Xb.T @ Y)
    return W


def _ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W


def _mean_shift_by_pert(train_proto):
    shifts: dict[str, list[np.ndarray]] = {}
    for _, pid, c, y in train_proto:
        shifts.setdefault(pid, []).append(y - c)
    shift_by_pert = {}
    for pid, vals in shifts.items():
        shift_by_pert[pid] = np.mean(np.stack(vals), axis=0)
    return shift_by_pert


def _summarize(values: list[float], bootstrap_samples: int, seed: int):
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float("nan"), (float("nan"), float("nan")), 0
    mean, lo, hi = bootstrap_mean(values, num_samples=bootstrap_samples, seed=seed)
    return mean, (lo, hi), len(values)


def _resampled_edist(control_idx, pert_idx, embeddings, rng, sample_size, resamples, device, pred_fn):
    if control_idx.size == 0 or pert_idx.size == 0:
        return []
    n = min(sample_size, control_idx.size, pert_idx.size)
    if n <= 0:
        return []
    edists = []
    for _ in range(resamples):
        c_sel = rng.choice(control_idx, size=n, replace=False)
        t_sel = rng.choice(pert_idx, size=n, replace=False)
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


def evaluate_set_baselines(
    train_proto,
    val_proto,
    test_pairs,
    embeddings,
    sample_size,
    resamples,
    seed,
    ridge_alphas,
    bootstrap_samples,
    bootstrap_seed,
    device,
):
    shift_by_pert = _mean_shift_by_pert(train_proto) if train_proto else {}

    X_train = np.stack([c for _, _, c, _ in train_proto])
    Y_train = np.stack([y for _, _, _, y in train_proto])
    if val_proto:
        X_val = np.stack([c for _, _, c, _ in val_proto])
        Y_val = np.stack([y for _, _, _, y in val_proto])
    else:
        X_val = Y_val = None

    best_alpha = ridge_alphas[0]
    best_mse = float("inf")
    best_W = None
    for alpha in ridge_alphas:
        W = _ridge_fit(X_train, Y_train, alpha)
        if X_val is not None and X_val.size > 0:
            pred = _ridge_predict(X_val, W)
            mse = float(np.mean((pred - Y_val) ** 2))
        else:
            pred = _ridge_predict(X_train, W)
            mse = float(np.mean((pred - Y_train) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_W = W

    def eval_shift(shift_fn, rng_seed):
        rng = np.random.default_rng(rng_seed)
        per_pair = []
        skipped = 0
        for p in test_pairs:
            c = embeddings[p.control_indices]
            if c.size == 0:
                skipped += 1
                continue
            c = np.mean(c, axis=0)
            shift = shift_fn(p.perturbation_id, c)
            shift_t = torch.tensor(shift, dtype=torch.float32, device=device)

            def pred_fn(x):
                return x + shift_t

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
        return {"edist_mean": mean, "edist_ci95": ci, "n_eval": n_eval, "skipped_pairs": skipped}

    results = {}
    results["no_change"] = eval_shift(lambda _pid, c: np.zeros_like(c), seed + 1)
    results["mean_shift"] = eval_shift(lambda pid, c: shift_by_pert.get(pid, np.zeros_like(c)), seed + 2)

    def ridge_shift(_pid, c):
        pred = _ridge_predict(c[None, :], best_W)[0]
        return pred - c

    results["ridge"] = eval_shift(ridge_shift, seed + 3)
    results["ridge_alpha"] = best_alpha
    return results


def build_baseline_shift_map(
    baseline: str,
    train_proto,
    val_proto,
    pairs,
    embeddings,
    ridge_alphas,
):
    if baseline == "none":
        return {}, {}

    meta = {}
    if baseline == "no_change":
        shift_map = {}
        for p in pairs:
            shift_map[(p.context_id, p.perturbation_id)] = np.zeros(embeddings.shape[1], dtype=np.float32)
        return shift_map, meta

    if not train_proto:
        raise ValueError("No training pairs available for baseline residual.")

    if baseline == "mean_shift":
        shift_by_pert = _mean_shift_by_pert(train_proto)
        shift_map = {}
        for p in pairs:
            c = embeddings[p.control_indices]
            if c.size == 0:
                continue
            c = np.mean(c, axis=0)
            shift = shift_by_pert.get(p.perturbation_id, np.zeros_like(c))
            shift_map[(p.context_id, p.perturbation_id)] = shift
        return shift_map, meta

    if baseline == "ridge":
        X_train = np.stack([c for _, _, c, _ in train_proto])
        Y_train = np.stack([y for _, _, _, y in train_proto])
        if val_proto:
            X_val = np.stack([c for _, _, c, _ in val_proto])
            Y_val = np.stack([y for _, _, _, y in val_proto])
        else:
            X_val = Y_val = None

        best_alpha = ridge_alphas[0]
        best_mse = float("inf")
        best_W = None
        for alpha in ridge_alphas:
            W = _ridge_fit(X_train, Y_train, alpha)
            if X_val is not None and X_val.size > 0:
                pred = _ridge_predict(X_val, W)
                mse = float(np.mean((pred - Y_val) ** 2))
            else:
                pred = _ridge_predict(X_train, W)
                mse = float(np.mean((pred - Y_train) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
                best_W = W

        meta["ridge_alpha"] = best_alpha
        shift_map = {}
        for p in pairs:
            c = embeddings[p.control_indices]
            if c.size == 0:
                continue
            c = np.mean(c, axis=0)
            pred = _ridge_predict(c[None, :], best_W)[0]
            shift_map[(p.context_id, p.perturbation_id)] = pred - c
        return shift_map, meta

    raise ValueError(f"Unknown residual baseline: {baseline}")


def train_world_model_residual(
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    pairs: list[PairSet],
    embeddings: np.ndarray,
    pert_to_idx: dict[str, int],
    baseline_shift_map: dict[tuple[str, str], np.ndarray],
    device: str,
    epochs: int,
    sample_size: int,
    seed: int,
):
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
            pred = c + shift_t + model(c, idx)
            loss = energy_distance_torch(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return {"loss": float(np.mean(losses)) if losses else float("nan"), "skipped_train": skipped}


def _compute_residual_edists(
    model: WorldModel,
    pairs: list[PairSet],
    embeddings: np.ndarray,
    pert_to_idx: dict[str, int],
    baseline_shift_map: dict[tuple[str, str], np.ndarray],
    device: str,
    sample_size: int,
    resamples: int,
    seed: int,
    alpha: float,
):
    rng = np.random.default_rng(seed)
    per_pair = []
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


def train_world_model(
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    pairs: list[PairSet],
    embeddings: np.ndarray,
    pert_to_idx: dict[str, int],
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
            pred = model(c, idx)
            loss = energy_distance_torch(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return {"loss": float(np.mean(losses)) if losses else float("nan"), "skipped_train": skipped}


def eval_world_model(
    model: WorldModel,
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
    per_pair = []
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

    mean, ci, n_eval = _summarize(per_pair, bootstrap_samples, bootstrap_seed)
    return {"edist_mean": mean, "edist_ci95": ci, "n_eval": n_eval, "skipped_pairs": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action-conditioned set-to-set world model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--eval-sample-size", type=int, default=None)
    parser.add_argument("--eval-resamples", type=int, default=5)
    parser.add_argument("--min-cells-per-condition", type=int, default=30)
    parser.add_argument("--max-cells-per-group", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--residual-baseline", choices=["none", "no_change", "mean_shift", "ridge"], default="none")
    parser.add_argument("--residual-alpha-grid", type=str, default="0;0.25;0.5;0.75;1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-val-frac", type=float, default=0.0, help="Fraction of train pairs to hold out for alpha tuning when val split is empty.")
    parser.add_argument("--pair-val-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0,100.0")
    parser.add_argument("--eval-baselines", action="store_true")
    parser.add_argument("--action-embeddings", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    eval_sample_size = args.eval_sample_size or args.sample_size
    ridge_alphas = [float(x) for x in args.ridge_alphas.split(",") if x]
    grid_raw = args.residual_alpha_grid
    sep = ";" if ";" in grid_raw else ","
    residual_alpha_grid = [float(x) for x in grid_raw.split(sep) if x]

    adata = ad.read_h5ad(args.dataset)
    split = json.loads(Path(args.split).read_text())
    dataset_id = split.get("dataset_id") or adata.uns.get("dataset_id") or Path(args.dataset).stem
    split_name = split.get("split_name") or Path(args.split).stem

    pairs = build_pairs(
        adata,
        max_cells_per_group=args.max_cells_per_group,
        min_cells_per_condition=args.min_cells_per_condition,
        seed=args.seed,
    )
    if not pairs:
        raise ValueError("No condition pairs available after filtering.")
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, split["group_key"], split)

    # Build perturbation vocab from train only
    train_perturbations = sorted({p.perturbation_id for p in train_pairs})
    action_map = None
    action_dim = None
    missing_train = []
    if args.action_embeddings:
        action_map, action_dim = load_action_embeddings(Path(args.action_embeddings))
        pert_to_idx = {"<UNK>": 0}
        for key in sorted(action_map.keys()):
            pert_to_idx[key] = len(pert_to_idx)
        for p in train_perturbations:
            if p not in pert_to_idx:
                pert_to_idx[p] = len(pert_to_idx)
                missing_train.append(p)
    else:
        pert_to_idx = {"<UNK>": 0}
        for i, p in enumerate(train_perturbations, 1):
            pert_to_idx[p] = i

    indices = np.unique(np.concatenate([p.control_indices for p in pairs] + [p.pert_indices for p in pairs]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, jepa_cfg = embed_cells(adata, Path(args.checkpoint), indices, device=device)
    idx_map = {idx: i for i, idx in enumerate(indices)}

    for p in pairs:
        p.control_indices = np.array([idx_map[i] for i in p.control_indices if i in idx_map], dtype=np.int64)
        p.pert_indices = np.array([idx_map[i] for i in p.pert_indices if i in idx_map], dtype=np.int64)

    finite_mask = np.isfinite(embeddings).all(axis=1)
    filtered = []
    dropped = 0
    for p in pairs:
        c_idx = p.control_indices[finite_mask[p.control_indices]]
        t_idx = p.pert_indices[finite_mask[p.pert_indices]]
        if c_idx.size < args.min_cells_per_condition or t_idx.size < args.min_cells_per_condition:
            dropped += 1
            continue
        p.control_indices = c_idx
        p.pert_indices = t_idx
        filtered.append(p)
    pairs = filtered
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, split["group_key"], split)

    train_proto, skipped_train_proto = build_proto_pairs(train_pairs, embeddings)
    val_proto, skipped_val_proto = build_proto_pairs(val_pairs, embeddings)
    test_proto, skipped_test_proto = build_proto_pairs(test_pairs, embeddings)

    embed_dim = jepa_cfg.get("embed_dim", embeddings.shape[1])
    residual_mode = args.residual
    if args.residual_baseline != "none":
        residual_mode = False

    cfg = WorldModelConfig(
        embed_dim=embed_dim,
        action_vocab=len(pert_to_idx),
        hidden_dim=args.hidden_dim,
        residual=residual_mode,
        action_dim=action_dim,
    )

    model = WorldModel(cfg).to(device)
    if action_map is not None:
        apply_action_embeddings(model, pert_to_idx, action_map)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    metrics = {
        "mode": "world_model",
        "dataset_id": dataset_id,
        "split_name": split_name,
        "group_key": split.get("group_key"),
        "seed": args.seed,
        "min_cells_per_condition": args.min_cells_per_condition,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_pairs),
        "pairs_total": len(pairs),
        "pairs_dropped": dropped,
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
        "world_model_config": cfg.__dict__,
        "jepa_config": jepa_cfg,
    }
    if action_map is not None:
        metrics["action_embeddings"] = {
            "path": args.action_embeddings,
            "action_dim": action_dim,
            "n_embeddings": len(action_map),
            "missing_train": len(missing_train),
        }

    if args.residual_baseline != "none":
        all_pairs = train_pairs + val_pairs + test_pairs
        baseline_shift_map, baseline_meta = build_baseline_shift_map(
            args.residual_baseline,
            train_proto,
            val_proto,
            all_pairs,
            embeddings,
            ridge_alphas,
        )
        metrics["residual"] = {
            "baseline": args.residual_baseline,
            "alpha_grid": residual_alpha_grid,
            **baseline_meta,
        }
        metrics.update(
            train_world_model_residual(
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

        tune_pairs = val_pairs if val_pairs else train_pairs
        tuned_on = "val" if val_pairs else "train"
        if not val_pairs and args.pair_val_frac > 0:
            rng = np.random.default_rng(args.pair_val_seed)
            n_val = max(1, int(len(train_pairs) * args.pair_val_frac))
            idx = rng.choice(len(train_pairs), size=n_val, replace=False)
            tune_pairs = [train_pairs[i] for i in idx]
            tuned_on = "pair_val"
        best_alpha = None
        best_val = float("inf")
        val_scores = {}
        for alpha in residual_alpha_grid:
            vals, _ = _compute_residual_edists(
                model,
                tune_pairs,
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
            best_alpha = 0.0
            best_val = float("nan")
        metrics["residual"]["alpha"] = best_alpha
        metrics["residual"]["alpha_tuned_on"] = tuned_on
        if tuned_on == "pair_val":
            metrics["residual"]["pair_val_frac"] = args.pair_val_frac
            metrics["residual"]["pair_val_seed"] = args.pair_val_seed
        metrics["residual"]["val_edist"] = best_val
        metrics["residual"]["val_scores"] = val_scores

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
            best_alpha,
        )
        mean, ci, n_eval = _summarize(per_pair, args.bootstrap_samples, args.bootstrap_seed)
        metrics["test"] = {
            "edist_mean": mean,
            "edist_ci95": ci,
            "n_eval": n_eval,
            "skipped_pairs": skipped,
        }
    else:
        metrics.update(
            train_world_model(
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
        metrics["test"] = eval_world_model(
            model,
            test_pairs,
            embeddings,
            pert_to_idx,
            device,
            eval_sample_size,
            args.eval_resamples,
            args.seed + 7,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )

    if args.eval_baselines:
        metrics["baselines"] = evaluate_set_baselines(
            train_proto,
            val_proto,
            test_pairs,
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
                "script": "train_world_model",
                "run_dir": str(out_dir),
                "dataset_id": dataset_id,
                "split_name": split_name,
                "seed": args.seed,
                "test": metrics.get("test"),
                "baselines": metrics.get("baselines"),
                "residual": metrics.get("residual"),
                "world_model_config": metrics.get("world_model_config"),
            }
        )
    except Exception as exc:
        print(f"Attempt log skipped: {exc}")


if __name__ == "__main__":
    main()
