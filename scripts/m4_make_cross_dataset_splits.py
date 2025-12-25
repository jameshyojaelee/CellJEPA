#!/usr/bin/env python3
"""Create cross-dataset holdout split JSONs for M4."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Make cross-dataset holdout splits.")
    parser.add_argument("--train", nargs="+", required=True, help="Train dataset IDs")
    parser.add_argument("--test", nargs="+", required=True, help="Test dataset IDs")
    parser.add_argument("--gene-set-id", required=True)
    parser.add_argument("--gene-set-path", required=True)
    parser.add_argument("--dataset-paths", nargs="+", required=True, help="dataset_id=path mappings")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-name", required=True)
    args = parser.parse_args()

    dataset_paths = {}
    for item in args.dataset_paths:
        if "=" not in item:
            raise ValueError("dataset-paths must be in dataset_id=path form")
        k, v = item.split("=", 1)
        dataset_paths[k] = v

    split = {
        "split_name": args.split_name,
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "train_datasets": args.train,
        "test_datasets": args.test,
        "gene_set_id": args.gene_set_id,
        "gene_set_path": args.gene_set_path,
        "dataset_paths": dataset_paths,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
