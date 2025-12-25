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
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast-dev", action="store_true")
    parser.add_argument("--max-cells", type=int, default=50000, help="Max cells to load into memory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.dataset)
    X = adata.X
    if args.max_cells:
        X = _sample_rows(X, args.max_cells, seed=args.seed)
    X = _to_dense(X).astype(np.float32)
    if args.fast_dev:
        X = X[: min(5000, X.shape[0])]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    cfg = JepaConfig(
        input_dim=X.shape[1],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        predictor_hidden=args.hidden_dim,
        ema_decay=args.ema_decay,
        mask_ratio=args.mask_ratio,
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
    )

    model = JEPA(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
    loader = torch.utils.data.DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)

    history = []
    for epoch in range(train_cfg.epochs):
        metrics = train_epoch(model, opt, (batch[0] for batch in loader), cfg, train_cfg)
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
