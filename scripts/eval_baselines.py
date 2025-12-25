#!/usr/bin/env python3
"""Evaluate simple baselines on a processed dataset and split."""

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

from celljepa.data.validation import validate_or_raise
from celljepa.eval.baselines import build_pairs, evaluate_baselines
from celljepa.eval.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline predictors on a dataset split.")
    parser.add_argument("--dataset", required=True, help="Path to processed .h5ad file.")
    parser.add_argument("--split", required=True, help="Path to split JSON file.")
    parser.add_argument("--out", required=True, help="Output run directory.")
    parser.add_argument("--max-cells-per-group", type=int, default=None)
    parser.add_argument("--pca-components", type=int, default=50)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    split_path = Path(args.split)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(dataset_path)
    validate_or_raise(adata)

    split = json.loads(split_path.read_text())
    group_key = split["group_key"]
    train_groups = split["train_groups"]
    val_groups = split["val_groups"]
    test_groups = split["test_groups"]

    pairs = build_pairs(adata, max_cells_per_group=args.max_cells_per_group)
    results = evaluate_baselines(
        pairs,
        group_key=group_key,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        pca_components=args.pca_components,
    )

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = {
        "dataset": adata.uns.get("dataset_id", dataset_path.name),
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "split": split_path.name,
        "group_key": group_key,
    }
    write_report(out_dir / "report.md", summary, results)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()

