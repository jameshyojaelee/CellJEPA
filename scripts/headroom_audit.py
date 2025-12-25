#!/usr/bin/env python3
"""Headroom audit for M3.

This script quantifies whether an evaluation setting is baseline-saturated by comparing:
- split-safe baselines (trained on train split only), and
- explicitly labeled ORACLE (leaky) upper bounds computed using test labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from celljepa.train.transition_trainer import energy_distance_torch
from celljepa.eval.metrics import bootstrap_mean


def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


@dataclass
class Pair:
    context_id: str
    perturbation_id: str
    control_indices: np.ndarray
    pert_indices: np.ndarray


def embed_cells(
    adata,
    checkpoint_path: Path,
    indices: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, dict]:
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

    meta = {
        "jepa_config": ckpt.get("config", {}),
        "checkpoint": str(checkpoint_path),
    }
    return out, meta


def build_pairs(
    adata,
    *,
    max_cells_per_group: int | None,
    min_cells_per_condition: int,
    seed: int,
) -> list[Pair]:
    rng = np.random.default_rng(seed)
    obs = adata.obs

    control_groups: dict[str, np.ndarray] = {}
    for context_id, idx in obs[obs["is_control"]].groupby("context_id").indices.items():
        control_groups[str(context_id)] = np.asarray(idx, dtype=np.int64)

    pairs: list[Pair] = []
    pert_obs = obs[~obs["is_control"]]
    for (context_id, perturbation_id), idx in pert_obs.groupby(["context_id", "perturbation_id"]).indices.items():
        pid_str = str(perturbation_id).strip().lower()
        if pid_str in {"nan", "none", ""}:
            continue

        c_idx = control_groups.get(str(context_id))
        p_idx = np.asarray(idx, dtype=np.int64)
        if c_idx is None or c_idx.size == 0 or p_idx.size == 0:
            continue
        if c_idx.size < min_cells_per_condition or p_idx.size < min_cells_per_condition:
            continue

        if max_cells_per_group:
            if c_idx.size > max_cells_per_group:
                c_idx = rng.choice(c_idx, size=max_cells_per_group, replace=False)
            if p_idx.size > max_cells_per_group:
                p_idx = rng.choice(p_idx, size=max_cells_per_group, replace=False)

        pairs.append(
            Pair(
                context_id=str(context_id),
                perturbation_id=str(perturbation_id),
                control_indices=c_idx,
                pert_indices=p_idx,
            )
        )

    pairs.sort(key=lambda p: (p.context_id, p.perturbation_id))
    return pairs


def split_pairs(pairs: list[Pair], split: dict) -> tuple[list[Pair], list[Pair], list[Pair]]:
    group_key = split["group_key"]
    train_groups = set(map(str, split["train_groups"]))
    val_groups = set(map(str, split["val_groups"]))
    test_groups = set(map(str, split["test_groups"]))

    train, val, test = [], [], []
    for p in pairs:
        g = p.perturbation_id if group_key == "perturbation_id" else p.context_id
        if g in test_groups:
            test.append(p)
        elif g in val_groups:
            val.append(p)
        else:
            train.append(p)
    return train, val, test


def prototypes(pairs: list[Pair], embeddings: np.ndarray) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    out = {}
    for p in pairs:
        c = embeddings[p.control_indices]
        y = embeddings[p.pert_indices]
        out[(p.context_id, p.perturbation_id)] = (c.mean(axis=0), y.mean(axis=0))
    return out


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    XtX = Xb.T @ Xb
    XtX += alpha * np.eye(XtX.shape[0])
    W = np.linalg.solve(XtX, Xb.T @ Y)
    return W


def _ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W


def _summarize(values: list[float], bootstrap_samples: int, seed: int) -> dict:
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return {"mean": float("nan"), "ci95": (float("nan"), float("nan")), "n_eval": 0}
    mean, lo, hi = bootstrap_mean(values, num_samples=bootstrap_samples, seed=seed)
    return {"mean": mean, "ci95": (lo, hi), "n_eval": len(values)}


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


def eval_shift_model(
    pairs: list[Pair],
    embeddings: np.ndarray,
    shift_by_pair: dict[tuple[str, str], np.ndarray],
    *,
    sample_size: int,
    resamples: int,
    seed: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    device: str,
) -> dict:
    rng = np.random.default_rng(seed)
    per_pair = []
    skipped = 0
    for p in pairs:
        shift = shift_by_pair.get((p.context_id, p.perturbation_id))
        if shift is None:
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

    summary = _summarize(per_pair, bootstrap_samples, bootstrap_seed)
    return {
        "edist_mean": summary["mean"],
        "edist_ci95": summary["ci95"],
        "n_eval": summary["n_eval"],
        "skipped_pairs": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Headroom audit for M3 (baselines + oracles).")
    parser.add_argument("--dataset", required=True, help="Processed .h5ad path.")
    parser.add_argument("--checkpoint", required=True, help="JEPA checkpoint path.")
    parser.add_argument("--split", required=True, help="Split JSON path.")
    parser.add_argument("--out", required=True, help="Run output directory (writes metrics + report).")
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--resamples", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--min-cells-per-condition", type=int, default=30)
    parser.add_argument("--max-cells-per-group", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0,100.0")
    parser.add_argument("--include-oracle-ridge", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ridge_alphas = [float(x) for x in args.ridge_alphas.split(",") if x]

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
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, split)

    indices = np.unique(np.concatenate([p.control_indices for p in pairs] + [p.pert_indices for p in pairs]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, embed_meta = embed_cells(adata, Path(args.checkpoint), indices, device=device)

    idx_map = {idx: i for i, idx in enumerate(indices)}
    for p in pairs:
        p.control_indices = np.array([idx_map[i] for i in p.control_indices if i in idx_map], dtype=np.int64)
        p.pert_indices = np.array([idx_map[i] for i in p.pert_indices if i in idx_map], dtype=np.int64)

    finite_mask = np.isfinite(embeddings).all(axis=1)
    nonfinite_cells = int(np.sum(~finite_mask))
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
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, split)

    train_proto = prototypes(train_pairs, embeddings)
    val_proto = prototypes(val_pairs, embeddings) if val_pairs else {}
    test_proto = prototypes(test_pairs, embeddings)

    # Baseline: no-change (shift = 0)
    zero_shift = {(p.context_id, p.perturbation_id): np.zeros(embeddings.shape[1], dtype=np.float32) for p in test_pairs}

    # Baseline: mean shift by perturbation (train contexts only)
    shift_by_pert: dict[str, list[np.ndarray]] = {}
    for (ctx, pid), (c, y) in train_proto.items():
        shift_by_pert.setdefault(pid, []).append(y - c)
    mean_shift_by_pert = {pid: np.mean(np.stack(vals), axis=0) for pid, vals in shift_by_pert.items() if vals}
    train_mean_shift = {(p.context_id, p.perturbation_id): mean_shift_by_pert.get(p.perturbation_id, np.zeros(embeddings.shape[1], dtype=np.float32)) for p in test_pairs}

    # Baseline: ridge on prototypes (train contexts only; alpha tuned on val if present)
    X_train = np.stack([c for (c, _) in train_proto.values()]) if train_proto else None
    Y_train = np.stack([y for (_, y) in train_proto.values()]) if train_proto else None
    if val_proto:
        X_val = np.stack([c for (c, _) in val_proto.values()])
        Y_val = np.stack([y for (_, y) in val_proto.values()])
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

    ridge_shift = {}
    for p in test_pairs:
        c, _ = test_proto[(p.context_id, p.perturbation_id)]
        pred = _ridge_predict(c[None, :], best_W)[0]
        ridge_shift[(p.context_id, p.perturbation_id)] = pred - c

    # ORACLE: mean shift by perturbation using test contexts (leaky)
    oracle_shift_by_pert: dict[str, list[np.ndarray]] = {}
    for (ctx, pid), (c, y) in test_proto.items():
        oracle_shift_by_pert.setdefault(pid, []).append(y - c)
    oracle_mean_shift_by_pert = {pid: np.mean(np.stack(vals), axis=0) for pid, vals in oracle_shift_by_pert.items() if vals}
    oracle_mean_shift = {(p.context_id, p.perturbation_id): oracle_mean_shift_by_pert.get(p.perturbation_id, np.zeros(embeddings.shape[1], dtype=np.float32)) for p in test_pairs}

    # ORACLE: per-(context, perturbation) shift using test pairs (leaky)
    oracle_pair_shift = {(ctx, pid): (y - c) for (ctx, pid), (c, y) in test_proto.items()}

    metrics = {
        "dataset_id": dataset_id,
        "split_name": split_name,
        "group_key": split.get("group_key"),
        "seed": args.seed,
        "min_cells_per_condition": args.min_cells_per_condition,
        "max_cells_per_group": args.max_cells_per_group,
        "nonfinite_cells": nonfinite_cells,
        "pairs_dropped": dropped,
        "n_pairs": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
        "eval": {
            "sample_size": args.sample_size,
            "resamples": args.resamples,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "embedding": embed_meta,
        "ridge": {"alpha": best_alpha, "val_mse": best_mse},
        "results": {},
    }

    metrics["results"]["baseline_no_change"] = eval_shift_model(
        test_pairs,
        embeddings,
        zero_shift,
        sample_size=args.sample_size,
        resamples=args.resamples,
        seed=args.seed + 1,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        device=device,
    )
    metrics["results"]["baseline_mean_shift_train"] = eval_shift_model(
        test_pairs,
        embeddings,
        train_mean_shift,
        sample_size=args.sample_size,
        resamples=args.resamples,
        seed=args.seed + 2,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        device=device,
    )
    metrics["results"]["baseline_ridge_train"] = eval_shift_model(
        test_pairs,
        embeddings,
        ridge_shift,
        sample_size=args.sample_size,
        resamples=args.resamples,
        seed=args.seed + 3,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        device=device,
    )
    metrics["results"]["oracle_mean_shift_test"] = eval_shift_model(
        test_pairs,
        embeddings,
        oracle_mean_shift,
        sample_size=args.sample_size,
        resamples=args.resamples,
        seed=args.seed + 4,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        device=device,
    )
    metrics["results"]["oracle_pair_shift_test"] = eval_shift_model(
        test_pairs,
        embeddings,
        oracle_pair_shift,
        sample_size=args.sample_size,
        resamples=args.resamples,
        seed=args.seed + 5,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        device=device,
    )

    if args.include_oracle_ridge:
        # ORACLE ridge: fit on test prototypes, report on the same test pairs (leaky upper bound).
        X_test = np.stack([c for (c, _) in test_proto.values()])
        Y_test = np.stack([y for (_, y) in test_proto.values()])
        oracle_W = _ridge_fit(X_test, Y_test, best_alpha)
        oracle_ridge_shift = {}
        for p in test_pairs:
            c, _ = test_proto[(p.context_id, p.perturbation_id)]
            pred = _ridge_predict(c[None, :], oracle_W)[0]
            oracle_ridge_shift[(p.context_id, p.perturbation_id)] = pred - c
        metrics["results"]["oracle_ridge_test"] = eval_shift_model(
            test_pairs,
            embeddings,
            oracle_ridge_shift,
            sample_size=args.sample_size,
            resamples=args.resamples,
            seed=args.seed + 6,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            device=device,
        )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def fmt_row(name: str, d: dict) -> str:
        ci = d.get("edist_ci95", (float("nan"), float("nan")))
        return (
            f"| {name} | {d.get('edist_mean', float('nan')):.4f} "
            f"| [{ci[0]:.4f}, {ci[1]:.4f}] | {d.get('n_eval', '')} | {d.get('skipped_pairs', '')} |"
        )

    lines = [
        f"# M3 Headroom Audit — {dataset_id} — {split_name}",
        "",
        "This report compares split-safe baselines to explicit ORACLE (leaky) upper bounds to estimate headroom.",
        "",
        "## Pair counts",
        "",
        f"- train pairs: {len(train_pairs)}",
        f"- val pairs: {len(val_pairs)}",
        f"- test pairs: {len(test_pairs)}",
        f"- dropped pairs (filters/nonfinite): {dropped}",
        f"- non-finite embedding rows: {nonfinite_cells}",
        "",
        "## Results (E-distance; mean ± 95% CI)",
        "",
        "| method | edist_mean | ci95 | n_eval | skipped_pairs |",
        "|---|---|---|---|---|",
        fmt_row("baseline_no_change", metrics["results"]["baseline_no_change"]),
        fmt_row("baseline_mean_shift_train", metrics["results"]["baseline_mean_shift_train"]),
        fmt_row(f"baseline_ridge_train (alpha={best_alpha})", metrics["results"]["baseline_ridge_train"]),
        "",
        "### ORACLE (leaky; not valid for acceptance)",
        "",
        fmt_row("oracle_mean_shift_test", metrics["results"]["oracle_mean_shift_test"]),
        fmt_row("oracle_pair_shift_test", metrics["results"]["oracle_pair_shift_test"]),
    ]
    if "oracle_ridge_test" in metrics["results"]:
        lines.append(fmt_row(f"oracle_ridge_test (alpha={best_alpha})", metrics["results"]["oracle_ridge_test"]))

    lines += [
        "",
        "## Notes",
        "- ORACLE rows intentionally use test information and must never be used as acceptance evidence.",
        "- If ORACLE methods barely improve over baselines, the benchmark is likely baseline-saturated under this metric.",
    ]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
