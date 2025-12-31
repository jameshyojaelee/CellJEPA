#!/usr/bin/env python3
"""Create cross-dataset holdout split JSONs for M4."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np


def _load_noncontrol_perturbations(path: Path) -> set[str]:
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    if "is_control" not in obs or "perturbation_id" not in obs:
        raise ValueError(f"Missing required obs columns in {path} (need is_control + perturbation_id).")
    perts = obs.loc[~obs["is_control"], "perturbation_id"].astype(str).values
    out: set[str] = set()
    for pid in np.unique(perts):
        pid_str = str(pid).strip()
        if pid_str.lower() in {"nan", "none", ""}:
            continue
        out.add(pid_str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Make cross-dataset holdout splits.")
    parser.add_argument("--train", nargs="+", required=True, help="Train dataset IDs")
    parser.add_argument("--test", nargs="+", required=True, help="Test dataset IDs")
    parser.add_argument("--gene-set-id", required=True)
    parser.add_argument("--gene-set-path", required=True)
    parser.add_argument("--dataset-paths", nargs="+", required=True, help="dataset_id=path mappings")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument(
        "--require-min-action-overlap",
        type=int,
        default=None,
        help="If set, require at least this many shared non-control perturbations between train and each test dataset.",
    )
    parser.add_argument(
        "--require-min-action-overlap-frac",
        type=float,
        default=None,
        help="If set, require at least this fraction of test perturbations to be present in train.",
    )
    args = parser.parse_args()

    dataset_paths = {}
    for item in args.dataset_paths:
        if "=" not in item:
            raise ValueError("dataset-paths must be in dataset_id=path form")
        k, v = item.split("=", 1)
        dataset_paths[k] = v

    # Optional sanity check: ensure cross-dataset split has meaningful action overlap.
    overlap_summary = None
    if args.require_min_action_overlap is not None or args.require_min_action_overlap_frac is not None:
        all_ids = list(dict.fromkeys(args.train + args.test))
        perts_by_id = {did: _load_noncontrol_perturbations(Path(dataset_paths[did])) for did in all_ids}
        train_perts = set().union(*[perts_by_id[did] for did in args.train])
        per_test = {}
        for did in args.test:
            test_perts = perts_by_id[did]
            shared = train_perts & test_perts
            frac = float(len(shared) / max(len(test_perts), 1))
            per_test[did] = {"test_unique": len(test_perts), "shared": len(shared), "shared_frac": frac}
            if args.require_min_action_overlap is not None and len(shared) < args.require_min_action_overlap:
                raise ValueError(
                    f"Insufficient action overlap for {did}: shared={len(shared)} < {args.require_min_action_overlap}."
                )
            if args.require_min_action_overlap_frac is not None and frac < args.require_min_action_overlap_frac:
                raise ValueError(
                    f"Insufficient action overlap for {did}: shared_frac={frac:.4f} < {args.require_min_action_overlap_frac}."
                )
        overlap_summary = {
            "train_unique": len(train_perts),
            "per_test": per_test,
        }

    split = {
        "split_name": args.split_name,
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "train_datasets": args.train,
        "test_datasets": args.test,
        "gene_set_id": args.gene_set_id,
        "gene_set_path": args.gene_set_path,
        "dataset_paths": dataset_paths,
    }
    if overlap_summary is not None:
        split["action_overlap"] = overlap_summary

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
