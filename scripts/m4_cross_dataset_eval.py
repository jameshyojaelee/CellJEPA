#!/usr/bin/env python3
"""Evaluate cross-dataset holdout using a pre-trained JEPA checkpoint.

This is a minimal cross-dataset runner that:
- loads train and test datasets from the split JSON,
- trains a set-level predictor on the train dataset,
- evaluates on the test dataset with baselines and CIs.

M4-v2 notes:
- Cross-dataset must be evaluated on **shared actions only** by default (train-vocab actions).
- Embedding calibration via control z-scoring is enabled by default.
- Transition head is safe-by-construction (residual + alpha gating + bounded deltas).
"""

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
from celljepa.models.transition import ResidualSetPredictor, TransitionConfig
from celljepa.train.transition_trainer import PairSet, train_set
from celljepa.train.transition_trainer import energy_distance_torch
import importlib.util


def _load_transition_helpers():
    path = ROOT / "scripts" / "train_transition.py"
    spec = importlib.util.spec_from_file_location("train_transition_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_helpers = _load_transition_helpers()
build_pairs = _helpers.build_pairs
build_proto_pairs = _helpers.build_proto_pairs
evaluate_set_baselines = _helpers.evaluate_set_baselines
eval_set_model = _helpers.eval_set_model
_resampled_edist = _helpers._resampled_edist


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
        x = X[batch_idx]
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x).astype(np.float32)
        with torch.no_grad():
            z = model.student(torch.from_numpy(x).to(device)).cpu().numpy()
        out[i : i + batch_idx.size] = z
    return out


def _parse_alpha_grid(text: str) -> list[float]:
    out: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    if not out:
        raise ValueError("alpha grid is empty")
    return out


def _zscore_by_controls(
    embeddings: np.ndarray,
    is_control_mask: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    is_control_mask = np.asarray(is_control_mask).astype(bool)
    if is_control_mask.size != embeddings.shape[0]:
        raise ValueError("is_control_mask must align with embeddings rows")
    ctrl = embeddings[is_control_mask]
    if ctrl.size == 0:
        raise ValueError("No control rows available for control z-scoring.")
    mu = np.mean(ctrl, axis=0)
    sigma = np.std(ctrl, axis=0)
    sigma = np.maximum(sigma, eps)
    emb = (embeddings - mu) / sigma
    info = {"enabled": True, "n_controls": int(ctrl.shape[0]), "eps": float(eps)}
    return emb, info


def _mean_edist(
    model: torch.nn.Module,
    pairs: list[PairSet],
    embeddings: np.ndarray,
    pert_to_idx: dict[str, int],
    device: str,
    sample_size: int,
    resamples: int,
    seed: int,
) -> tuple[float, int, int]:
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
    if not per_pair:
        return float("nan"), skipped, 0
    return float(np.mean(per_pair)), skipped, len(per_pair)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset holdout evaluator (set predictor).")
    parser.add_argument("--split", required=True, help="Cross-dataset split JSON")
    parser.add_argument("--checkpoint", required=True, help="JEPA checkpoint trained on train dataset")
    parser.add_argument("--out", required=True, help="Output run directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--eval-sample-size", type=int, default=None)
    parser.add_argument("--eval-resamples", type=int, default=5)
    parser.add_argument("--min-cells-per-condition", type=int, default=30)
    parser.add_argument("--max-cells-per-group", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--eval-baselines", action="store_true")
    parser.add_argument(
        "--allow-unk-actions",
        action="store_true",
        help="Do not filter to shared actions only (diagnostic; main M4 table should keep this off).",
    )
    parser.add_argument(
        "--no-control-zscore",
        action="store_true",
        help="Disable control-based embedding z-scoring (enabled by default in M4-v2).",
    )
    parser.add_argument("--alpha-grid", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--alpha-val-frac", type=float, default=0.2)
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument("--effect-top-frac", type=float, default=0.0, help="Filter test pairs to top fraction by effect size.")
    parser.add_argument("--effect-seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split).read_text())
    train_ids = split["train_datasets"]
    test_ids = split["test_datasets"]
    dataset_paths = split["dataset_paths"]
    split_name = split.get("split_name", "cross_dataset")
    gene_set_id = split.get("gene_set_id")

    if len(test_ids) != 1:
        raise ValueError("This minimal runner expects exactly one test dataset.")

    test_path = Path(dataset_paths[test_ids[0]])
    ad_test = ad.read_h5ad(test_path)

    # Concatenate train datasets if multiple are provided.
    train_adatas = []
    for did in train_ids:
        train_path = Path(dataset_paths[did])
        train_adatas.append(ad.read_h5ad(train_path))
    if len(train_adatas) == 1:
        ad_train = train_adatas[0]
    else:
        ad_train = ad.concat(train_adatas, join="inner", label="dataset_id", keys=train_ids)
    # Build pairs within each dataset
    train_pairs = build_pairs(
        ad_train,
        max_cells_per_group=args.max_cells_per_group,
        min_cells_per_condition=args.min_cells_per_condition,
        seed=args.seed,
    )
    test_pairs = build_pairs(
        ad_test,
        max_cells_per_group=args.max_cells_per_group,
        min_cells_per_condition=args.min_cells_per_condition,
        seed=args.seed,
    )

    # Filter by min cells per condition (done in build_pairs); enforce non-empty
    if not train_pairs or not test_pairs:
        raise ValueError("No condition pairs after filtering.")

    # Build perturbation vocab from train dataset only
    train_perturbations = sorted({p.perturbation_id for p in train_pairs})
    pert_to_idx = {"<UNK>": 0}
    for i, p in enumerate(train_perturbations, 1):
        pert_to_idx[p] = i

    test_perturbations = sorted({p.perturbation_id for p in test_pairs})
    overlap = sorted(set(train_perturbations) & set(test_perturbations))

    # Embed both datasets with same JEPA checkpoint
    indices_train = np.unique(np.concatenate([p.control_indices for p in train_pairs] + [p.pert_indices for p in train_pairs]))
    indices_test = np.unique(np.concatenate([p.control_indices for p in test_pairs] + [p.pert_indices for p in test_pairs]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_train = embed_cells(ad_train, Path(args.checkpoint), indices_train, device=device)
    emb_test = embed_cells(ad_test, Path(args.checkpoint), indices_test, device=device)

    # Optional control-based z-scoring (enabled by default).
    control_zscore_enabled = not args.no_control_zscore
    calib = {"control_zscore": {"enabled": bool(control_zscore_enabled)}}
    if control_zscore_enabled:
        is_ctrl_train = np.asarray(ad_train.obs["is_control"].values[indices_train], dtype=bool)
        is_ctrl_test = np.asarray(ad_test.obs["is_control"].values[indices_test], dtype=bool)
        emb_train, train_info = _zscore_by_controls(emb_train, is_ctrl_train)
        emb_test, test_info = _zscore_by_controls(emb_test, is_ctrl_test)
        calib["control_zscore"]["train"] = train_info
        calib["control_zscore"]["test"] = test_info

    idx_map_train = {idx: i for i, idx in enumerate(indices_train)}
    for p in train_pairs:
        p.control_indices = np.array([idx_map_train[i] for i in p.control_indices if i in idx_map_train], dtype=np.int64)
        p.pert_indices = np.array([idx_map_train[i] for i in p.pert_indices if i in idx_map_train], dtype=np.int64)

    idx_map_test = {idx: i for i, idx in enumerate(indices_test)}
    for p in test_pairs:
        p.control_indices = np.array([idx_map_test[i] for i in p.control_indices if i in idx_map_test], dtype=np.int64)
        p.pert_indices = np.array([idx_map_test[i] for i in p.pert_indices if i in idx_map_test], dtype=np.int64)

    # Filter to shared actions only by default (main M4-v2 table).
    n_test_before_action_filter = len(test_pairs)
    if not args.allow_unk_actions:
        test_pairs = [p for p in test_pairs if pert_to_idx.get(p.perturbation_id, 0) != 0]
    n_test_after_action_filter = len(test_pairs)
    if not test_pairs:
        raise ValueError(
            "No shared-action test pairs after filtering. "
            "Adjust cross-dataset split selection or perturbation ID normalization."
        )

    def effect_scores(pairs, embeddings, sample_size, resamples, device, seed):
        rng = np.random.default_rng(seed)
        scores = []
        for p in pairs:
            c_idx = p.control_indices
            t_idx = p.pert_indices
            n = min(sample_size, c_idx.size, t_idx.size)
            if n <= 0:
                scores.append(float("nan"))
                continue
            vals = []
            for _ in range(resamples):
                c_sel = rng.choice(c_idx, size=n, replace=False)
                t_sel = rng.choice(t_idx, size=n, replace=False)
                c = torch.tensor(embeddings[c_sel], dtype=torch.float32, device=device)
                y = torch.tensor(embeddings[t_sel], dtype=torch.float32, device=device)
                if not torch.isfinite(c).all() or not torch.isfinite(y).all():
                    continue
                dist = energy_distance_torch(c, y)
                vals.append(float(dist.detach().cpu().numpy()))
            scores.append(float(np.mean(vals)) if vals else float("nan"))
        return scores

    # Default bounded-step scale from train effect sizes (prototype shift norms).
    if args.max_delta_norm is None:
        shift_norms = []
        for p in train_pairs:
            c = emb_train[p.control_indices]
            y = emb_train[p.pert_indices]
            if c.size == 0 or y.size == 0:
                continue
            c_mean = np.mean(c, axis=0)
            y_mean = np.mean(y, axis=0)
            if not np.isfinite(c_mean).all() or not np.isfinite(y_mean).all():
                continue
            shift_norms.append(float(np.linalg.norm(y_mean - c_mean)))
        max_delta_norm = float(np.percentile(shift_norms, 95)) if shift_norms else 1.0
    else:
        max_delta_norm = float(args.max_delta_norm)

    cfg = TransitionConfig(
        embed_dim=emb_train.shape[1],
        perturbation_vocab=len(pert_to_idx),
        residual=True,
        alpha=1.0,
        max_delta_norm=max_delta_norm,
    )
    model = ResidualSetPredictor(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    metrics = {
        "mode": "cross_dataset_set",
        "split_name": split_name,
        "gene_set_id": gene_set_id,
        "train_dataset": train_ids[0],
        "test_dataset": test_ids[0],
        "seed": args.seed,
        "embedding_calibration": calib,
        "action_overlap": {
            "train_unique": len(train_perturbations),
            "test_unique": len(test_perturbations),
            "overlap": len(overlap),
            "overlap_frac_test": float(len(overlap) / max(len(test_perturbations), 1)),
            "overlap_frac_train": float(len(overlap) / max(len(train_perturbations), 1)),
        },
        "action_filter": {
            "shared_actions_only": (not args.allow_unk_actions),
            "n_before": n_test_before_action_filter,
            "n_after": n_test_after_action_filter,
        },
        "model": {
            "type": "ResidualSetPredictor",
            "max_delta_norm": max_delta_norm,
        },
    }

    # Alpha selection: hold out part of the train data and pick alpha that minimizes mean E-distance.
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    rng = np.random.default_rng(args.seed)
    contexts = sorted({p.context_id for p in train_pairs})
    val_frac = float(args.alpha_val_frac)

    train_fit = train_pairs
    val_pairs: list[PairSet] = []
    val_contexts: set[str] = set()
    alpha_val_strategy = "none"

    if len(train_pairs) >= 2:
        # Prefer context holdout (if possible), otherwise fall back to pair holdout.
        n_val_ctx = max(1, int(len(contexts) * val_frac))
        if len(contexts) >= 2 and n_val_ctx < len(contexts):
            val_contexts = set(rng.choice(np.array(contexts, dtype=object), size=n_val_ctx, replace=False).tolist())
            train_fit = [p for p in train_pairs if p.context_id not in val_contexts]
            val_pairs = [p for p in train_pairs if p.context_id in val_contexts]
            alpha_val_strategy = "context_holdout"
        else:
            n_val_pairs = max(1, int(len(train_pairs) * val_frac))
            if n_val_pairs < len(train_pairs):
                perm = rng.permutation(len(train_pairs))
                val_idx = set(perm[:n_val_pairs].tolist())
                val_pairs = [p for i, p in enumerate(train_pairs) if i in val_idx]
                train_fit = [p for i, p in enumerate(train_pairs) if i not in val_idx]
                alpha_val_strategy = "pair_holdout"

    metrics["alpha_selection"] = {
        "alpha_grid": alpha_grid,
        "val_frac": val_frac,
        "strategy": alpha_val_strategy,
        "n_train_pairs": len(train_fit),
        "n_val_pairs": len(val_pairs),
        "n_train_contexts": len({p.context_id for p in train_fit}),
        "n_val_contexts": len(val_contexts) if val_contexts else len({p.context_id for p in val_pairs}),
    }

    metrics.update(
        train_set(
            model,
            opt,
            train_fit,
            emb_train,
            pert_to_idx,
            device,
            epochs=args.epochs,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    )

    # Select alpha on held-out train contexts (fallback: keep alpha=1.0 if no val pairs).
    best_alpha = 1.0
    alpha_scores = {}
    if val_pairs:
        for a in alpha_grid:
            model.alpha = float(a)
            mean_edist, skipped, n_eval = _mean_edist(
                model,
                val_pairs,
                emb_train,
                pert_to_idx,
                device,
                sample_size=args.sample_size,
                resamples=max(1, args.eval_resamples),
                seed=args.seed,
            )
            alpha_scores[str(a)] = {"mean_edist": mean_edist, "skipped_pairs": skipped, "n_eval": n_eval}
        best_alpha = min(alpha_grid, key=lambda a: alpha_scores[str(a)]["mean_edist"])
    model.alpha = float(best_alpha)
    metrics["alpha_selection"]["scores"] = alpha_scores
    metrics["alpha_selection"]["alpha_selected"] = float(best_alpha)
    metrics["model"]["alpha"] = float(best_alpha)

    eval_sample_size = args.eval_sample_size or args.sample_size
    if args.effect_top_frac and args.effect_top_frac > 0:
        scores = effect_scores(test_pairs, emb_test, eval_sample_size, args.eval_resamples, device, args.effect_seed)
        scored = [(p, s) for p, s in zip(test_pairs, scores) if np.isfinite(s)]
        scored.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(len(scored) * args.effect_top_frac))
        test_pairs = [p for p, _ in scored[:k]]
        kept_scores = [s for _, s in scored[:k]]
        metrics["effect_filter"] = {
            "top_frac": args.effect_top_frac,
            "n_before": len(scores),
            "n_after": len(test_pairs),
            "mean_effect": float(np.mean(kept_scores)) if kept_scores else float("nan"),
        }

    metrics["test"] = eval_set_model(
        model,
        test_pairs,
        emb_test,
        pert_to_idx,
        device,
        eval_sample_size,
        args.eval_resamples,
        args.seed,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    if args.eval_baselines:
        train_proto, _ = build_proto_pairs(train_pairs, emb_train)
        val_proto = []
        test_proto, _ = build_proto_pairs(test_pairs, emb_test)
        test_proto_map = {(p.context_id, p.perturbation_id): p for p in test_proto}
        metrics["baselines"] = evaluate_set_baselines(
            train_proto,
            val_proto,
            test_pairs,
            test_proto_map,
            emb_test,
            eval_sample_size,
            args.eval_resamples,
            args.seed,
            [0.1, 1.0, 10.0, 100.0],
            args.bootstrap_samples,
            args.bootstrap_seed,
            device,
        )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
