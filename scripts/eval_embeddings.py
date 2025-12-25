#!/usr/bin/env python3
"""Compute embedding diagnostics (variance + optional kNN label retrieval)."""

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


def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T


def knn_retrieval(emb: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    sims = cosine_sim(emb, emb)
    np.fill_diagonal(sims, -np.inf)
    nn_idx = np.argsort(-sims, axis=1)[:, :k]
    nn_labels = labels[nn_idx]
    preds = []
    for row in nn_labels:
        vals, counts = np.unique(row, return_counts=True)
        preds.append(vals[np.argmax(counts)])
    preds = np.array(preds)
    return float(np.mean(preds == labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embeddings from a JEPA checkpoint.")
    parser.add_argument("--dataset", required=True, help="Processed .h5ad path.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--out", required=True, help="Output metrics json")
    parser.add_argument("--label-col", default=None, help="obs column for kNN retrieval (optional)")
    parser.add_argument("--max-cells", type=int, default=5000)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.dataset)
    X = adata.X
    if args.max_cells and X.shape[0] > args.max_cells:
        X = X[: args.max_cells]
        obs = adata.obs.iloc[: args.max_cells]
    else:
        obs = adata.obs
    X = _to_dense(X).astype(np.float32)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = JepaConfig(**ckpt["config"])
    model = JEPA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        emb = model.student(torch.from_numpy(X)).cpu().numpy()

    metrics = {
        "embed_variance_mean": float(emb.var(axis=0).mean()),
        "embed_variance_min": float(emb.var(axis=0).min()),
    }

    if args.label_col and args.label_col in obs.columns:
        labels = obs[args.label_col].astype(str).values
        metrics["knn_acc"] = knn_retrieval(emb, labels, k=args.k)
        metrics["knn_label_col"] = args.label_col

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote embedding metrics to {out_path}")


if __name__ == "__main__":
    main()
