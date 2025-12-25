#!/usr/bin/env python3
"""Generate deterministic splits for a dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from celljepa.data import load_anndata, validate_or_raise
from celljepa.splits import build_split_spec, get_group_key, make_group_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create split files for CellJEPA datasets.")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID or path to .h5ad.")
    parser.add_argument("--split-name", required=True, help="Split name (e.g., S1_unseen_perturbation).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--out", required=True, help="Output JSON path.")
    args = parser.parse_args()

    adata = load_anndata(args.dataset_id)
    validate_or_raise(adata)

    group_key = get_group_key(args.split_name)
    groups = list(adata.obs[group_key].astype(str).unique())
    split = make_group_splits(groups, seed=args.seed, k_folds=args.k_folds, fold=args.fold)

    spec = build_split_spec(
        dataset_id=adata.uns.get("dataset_id", str(args.dataset_id)),
        split_name=args.split_name,
        seed=args.seed,
        fold=args.fold,
        train_groups=split["train_groups"],
        val_groups=split["val_groups"],
        test_groups=split["test_groups"],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Wrote split file to {out_path}")


if __name__ == "__main__":
    main()
