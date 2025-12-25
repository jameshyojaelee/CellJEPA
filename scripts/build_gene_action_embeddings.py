#!/usr/bin/env python3
"""Build simple gene action embeddings from training split.

We compute gene embeddings as the top PCA components of the gene-by-cell matrix
using training cells only (split-safe). Outputs a JSON mapping gene -> vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import anndata as ad


def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def load_split(split_path: Path):
    split = json.loads(split_path.read_text())
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gene action embeddings from training split.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--max-cells", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    adata = ad.read_h5ad(args.dataset)
    split = load_split(Path(args.split))
    group_key = split["group_key"]
    train_groups = set(map(str, split["train_groups"]))

    mask = adata.obs[group_key].astype(str).isin(train_groups).values
    adata = adata[mask]

    X = _to_dense(adata.X).astype(np.float32)
    if X.shape[0] > args.max_cells:
        idx = rng.choice(X.shape[0], size=args.max_cells, replace=False)
        X = X[idx]

    # X is (cells, genes). We want gene embeddings.
    X = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(args.n_components, Vt.shape[0])
    gene_embed = Vt[:k].T  # (genes, k)

    gene_names = [str(g) for g in adata.var_names]
    out = {gene: gene_embed[i].tolist() for i, gene in enumerate(gene_names)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(out)} genes")


if __name__ == "__main__":
    main()
