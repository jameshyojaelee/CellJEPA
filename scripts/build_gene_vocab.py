#!/usr/bin/env python3
"""Build a shared gene vocabulary from processed h5ad datasets.

Extracts var.index (gene symbols) from all processed h5ad files and
produces a deduplicated, sorted gene vocabulary file.

Usage:
    python scripts/build_gene_vocab.py \
        --data-dir data/processed \
        --out configs/gene_vocab_human_v1.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shared gene vocabulary.")
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "processed"),
        help="Directory containing processed h5ad files",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "configs" / "gene_vocab_human_v1.txt"),
        help="Output vocabulary file path",
    )
    parser.add_argument(
        "--min-datasets",
        type=int,
        default=1,
        help="Only include genes present in at least this many datasets (default: 1 = union)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    h5ad_files = sorted(data_dir.rglob("*.h5ad"))

    if not h5ad_files:
        print(f"ERROR: No h5ad files found under {data_dir}")
        sys.exit(1)

    print(f"Found {len(h5ad_files)} h5ad files under {data_dir}")

    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata required: pip install anndata")

    # Count gene occurrences across datasets
    from collections import Counter
    gene_counts: Counter = Counter()
    dataset_genes: dict[str, int] = {}

    for h5ad_path in h5ad_files:
        try:
            adata = ad.read_h5ad(h5ad_path, backed="r")
            genes = list(adata.var.index)
            gene_counts.update(genes)
            dataset_genes[h5ad_path.name] = len(genes)
            adata.file.close()
            print(f"  {h5ad_path.name}: {len(genes):,} genes")
        except Exception as e:
            print(f"  WARNING: skipping {h5ad_path.name}: {e}")
            continue

    # Filter by min_datasets
    if args.min_datasets > 1:
        vocab = sorted(g for g, c in gene_counts.items() if c >= args.min_datasets)
        print(f"\nFiltered to genes in ≥{args.min_datasets} datasets: {len(vocab):,}")
    else:
        vocab = sorted(gene_counts.keys())

    print(f"\nTotal unique genes (union): {len(gene_counts):,}")
    print(f"Vocabulary size: {len(vocab):,}")

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for gene in vocab:
            f.write(gene + "\n")

    print(f"Saved vocabulary to {out_path}")

    # Summary stats
    if dataset_genes:
        counts = list(dataset_genes.values())
        print(f"\nPer-dataset gene counts:")
        for name, count in sorted(dataset_genes.items()):
            print(f"  {name}: {count:,}")
        print(f"  Min: {min(counts):,}, Max: {max(counts):,}")


if __name__ == "__main__":
    main()
