#!/usr/bin/env python3
"""Evaluate cross-dataset holdout using a pre-trained JEPA checkpoint.

This is a minimal cross-dataset runner that:
- loads train and test datasets from the split JSON,
- trains a set-level predictor on the train dataset,
- evaluates on the test dataset with baselines and CIs.
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
from celljepa.models.transition import SetPredictor, TransitionConfig
from celljepa.train.transition_trainer import PairSet, train_set
from celljepa.eval.metrics import bootstrap_mean
from scripts.train_transition import (
    build_pairs,
    split_pairs,
    build_proto_pairs,
    evaluate_set_baselines,
    eval_set_model,
)


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
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split).read_text())
    train_ids = split["train_datasets"]
    test_ids = split["test_datasets"]
    dataset_paths = split["dataset_paths"]

    if len(train_ids) != 1 or len(test_ids) != 1:
        raise ValueError("This minimal runner expects exactly one train and one test dataset.")

    train_path = Path(dataset_paths[train_ids[0]])
    test_path = Path(dataset_paths[test_ids[0]])

    ad_train = ad.read_h5ad(train_path)
    ad_test = ad.read_h5ad(test_path)

    # Build pairs within each dataset
    train_pairs = build_pairs(ad_train, max_cells_per_group=args.max_cells_per_group, seed=args.seed)
    test_pairs = build_pairs(ad_test, max_cells_per_group=args.max_cells_per_group, seed=args.seed)

    # Filter by min cells per condition (done in build_pairs); enforce non-empty
    if not train_pairs or not test_pairs:
        raise ValueError("No condition pairs after filtering.")

    # Build perturbation vocab from train dataset only
    train_perturbations = sorted({p.perturbation_id for p in train_pairs})
    pert_to_idx = {"<UNK>": 0}
    for i, p in enumerate(train_perturbations, 1):
        pert_to_idx[p] = i

    # Embed both datasets with same JEPA checkpoint
    indices_train = np.unique(np.concatenate([p.control_indices for p in train_pairs] + [p.pert_indices for p in train_pairs]))
    indices_test = np.unique(np.concatenate([p.control_indices for p in test_pairs] + [p.pert_indices for p in test_pairs]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_train = embed_cells(ad_train, Path(args.checkpoint), indices_train, device=device)
    emb_test = embed_cells(ad_test, Path(args.checkpoint), indices_test, device=device)

    idx_map_train = {idx: i for i, idx in enumerate(indices_train)}
    for p in train_pairs:
        p.control_indices = np.array([idx_map_train[i] for i in p.control_indices if i in idx_map_train], dtype=np.int64)
        p.pert_indices = np.array([idx_map_train[i] for i in p.pert_indices if i in idx_map_train], dtype=np.int64)

    idx_map_test = {idx: i for i, idx in enumerate(indices_test)}
    for p in test_pairs:
        p.control_indices = np.array([idx_map_test[i] for i in p.control_indices if i in idx_map_test], dtype=np.int64)
        p.pert_indices = np.array([idx_map_test[i] for i in p.pert_indices if i in idx_map_test], dtype=np.int64)

    cfg = TransitionConfig(embed_dim=emb_train.shape[1], perturbation_vocab=len(pert_to_idx))
    model = SetPredictor(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    metrics = {
        "mode": "cross_dataset_set",
        "train_dataset": train_ids[0],
        "test_dataset": test_ids[0],
        "seed": args.seed,
    }

    metrics.update(
        train_set(
            model,
            opt,
            train_pairs,
            emb_train,
            pert_to_idx,
            device,
            epochs=args.epochs,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    )

    eval_sample_size = args.eval_sample_size or args.sample_size
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
        test_proto_map = {(p[0], p[1]): p for p in test_proto}
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
