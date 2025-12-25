#!/usr/bin/env python3
"""Subset a processed .h5ad to a provided gene list for M4 harmonization."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize dataset to a gene list.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--genes", required=True, help="Path to newline-delimited gene list.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    genes = [g.strip() for g in Path(args.genes).read_text().splitlines() if g.strip()]
    gene_set = set(genes)

    adata = ad.read_h5ad(args.dataset)
    var_names = [str(v) for v in adata.var_names]
    mask = [g in gene_set for g in var_names]
    if not any(mask):
        raise ValueError("No overlapping genes between dataset and gene list.")

    adata = adata[:, mask].copy()
    adata.uns["feature_set"] = Path(args.genes).stem
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    print(f"Wrote {args.out} with {adata.n_vars} genes")


if __name__ == "__main__":
    main()
