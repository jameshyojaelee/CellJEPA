#!/usr/bin/env python3
"""Build gene-aware action embeddings from training cells.

This uses a PCA embedding over the gene-by-cell matrix restricted to the
training split and writes a JSON mapping: gene -> vector.
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


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())


def select_training_indices(adata, split: dict) -> np.ndarray:
    group_key = split.get("group_key")
    if not group_key:
        raise ValueError("Split JSON missing group_key.")
    if group_key not in adata.obs:
        raise ValueError(f"Split group_key '{group_key}' not found in adata.obs.")
    train_groups = split.get("train_groups") or []
    if not train_groups:
        raise ValueError("Split JSON has no train_groups.")
    groups = set(map(str, train_groups))
    mask = adata.obs[group_key].astype(str).isin(groups)
    idx = np.where(mask.to_numpy())[0]
    if idx.size == 0:
        raise ValueError("No training cells found for split groups.")
    return idx


def maybe_downsample(indices: np.ndarray, max_cells: int | None, seed: int) -> np.ndarray:
    if max_cells and indices.size > max_cells:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_cells, replace=False)
    return np.sort(indices)


def load_matrix(adata, indices: np.ndarray, batch_size: int) -> np.ndarray:
    n_cells = indices.size
    n_genes = adata.shape[1]
    X = np.empty((n_cells, n_genes), dtype=np.float32)
    for start in range(0, n_cells, batch_size):
        batch_idx = indices[start : start + batch_size]
        X[start : start + batch_idx.size] = np.asarray(adata.X[batch_idx, :], dtype=np.float32)
    return X


def compute_gene_pca(
    X: np.ndarray,
    n_components: int,
    seed: int,
    device: str,
    oversample: int,
) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    X -= X.mean(axis=0, keepdims=True)
    max_rank = min(X.shape[0], X.shape[1])
    if n_components > max_rank:
        raise ValueError(f"n_components={n_components} exceeds max rank {max_rank}.")
    q = min(max_rank, n_components + oversample)
    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    Xt = torch.from_numpy(X).to(device)
    U, S, V = torch.pca_lowrank(Xt, q=q, center=False)
    order = torch.argsort(S, descending=True)
    S = S[order][:n_components]
    V = V[:, order][:, :n_components]
    gene_emb = V * S
    return gene_emb.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gene-aware action embeddings from training cells.")
    parser.add_argument("--dataset", required=True, help="Path to .h5ad dataset.")
    parser.add_argument("--split", required=True, help="Split JSON path (train groups only).")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--oversample", type=int, default=10)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    split_path = Path(args.split)
    out_path = Path(args.out)

    adata = ad.read_h5ad(dataset_path, backed="r")
    split = load_split(split_path)
    indices = select_training_indices(adata, split)
    indices = maybe_downsample(indices, args.max_cells, args.seed)

    X = load_matrix(adata, indices, args.batch_size)
    gene_emb = compute_gene_pca(X, args.n_components, args.seed, args.device, args.oversample)

    gene_names = [str(g) for g in adata.var_names]
    if gene_emb.shape[0] != len(gene_names):
        raise ValueError("Gene embedding count does not match number of genes.")

    payload = {gene: gene_emb[i].astype(float).tolist() for i, gene in enumerate(gene_names)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(payload)} genes (dim={gene_emb.shape[1]}).")


if __name__ == "__main__":
    main()
