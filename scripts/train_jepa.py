#!/usr/bin/env python3
"""Train a minimal JEPA model."""

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
from celljepa.train.jepa_trainer import TrainConfig, set_seed, train_epoch


def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _sample_rows(X, n: int, seed: int = 0):
    if X.shape[0] <= n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    if hasattr(X, "__getitem__"):
        return X[idx]
    return X[idx]


def load_module_masks(path: Path, gene_names: list[str]) -> list[np.ndarray]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        modules = list(data.values())
    else:
        modules = data
    name_to_idx = {name: i for i, name in enumerate(gene_names)}
    module_indices = []
    for entry in modules:
        genes = entry.get("genes") if isinstance(entry, dict) else entry
        if not genes:
            continue
        idx = [name_to_idx[g] for g in genes if g in name_to_idx]
        if idx:
            module_indices.append(np.array(idx, dtype=np.int64))
    return module_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JEPA (M2 minimal).")
    parser.add_argument("--dataset", required=True, help="Processed .h5ad path.")
    parser.add_argument("--out", required=True, help="Run output directory.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--mask-type", choices=["random", "module"], default="random")
    parser.add_argument("--module-mask-path", type=str, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--variance-target", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=1.0)
    parser.add_argument("--no-reg", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast-dev", action="store_true")
    parser.add_argument("--max-cells", type=int, default=50000, help="Max cells to load into memory.")
    args = parser.parse_args()
    if args.no_ema:
        args.ema_decay = 0.0
    if args.no_reg:
        args.variance_weight = 0.0
        args.covariance_weight = 0.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.dataset)
    gene_names = [str(g) for g in adata.var_names]
    X = adata.X
    if args.max_cells:
        X = _sample_rows(X, args.max_cells, seed=args.seed)
    X = _to_dense(X).astype(np.float32)
    if args.fast_dev:
        X = X[: min(5000, X.shape[0])]

    module_indices = None
    if args.mask_type == "module":
        if not args.module_mask_path:
            raise ValueError("--module-mask-path is required when --mask-type=module")
        module_indices = load_module_masks(Path(args.module_mask_path), gene_names)
        if not module_indices:
            raise ValueError("No valid module masks found for the provided gene set file.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    cfg = JepaConfig(
        input_dim=X.shape[1],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        predictor_hidden=args.hidden_dim,
        ema_decay=args.ema_decay,
        mask_ratio=args.mask_ratio,
        variance_target=args.variance_target,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        device=device,
        seed=args.seed,
        fast_dev=args.fast_dev,
        mask_type=args.mask_type,
        module_mask_path=args.module_mask_path,
    )

    model = JEPA(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
    loader = torch.utils.data.DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)

    history = []
    for epoch in range(train_cfg.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        metrics = train_epoch(
            model,
            opt,
            (batch[0] for batch in loader),
            cfg,
            train_cfg,
            mask_type=args.mask_type,
            module_indices=module_indices,
            rng=rng,
        )
        metrics["epoch"] = epoch
        history.append(metrics)
        print(f"epoch {epoch} loss={metrics['loss']:.4f} mse={metrics['mse']:.4f} var={metrics['var']:.4f} cov={metrics['cov']:.4f}")

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, ckpt_path)

    (out_dir / "metrics.json").write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "train": train_cfg.__dict__,
                "model": cfg.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
