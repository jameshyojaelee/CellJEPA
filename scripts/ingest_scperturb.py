#!/usr/bin/env python3
"""Ingest a scPerturb .h5ad file and standardize obs columns."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from celljepa.data.ingest_scperturb import IngestConfig, ingest_scperturb, _infer_default_perturbation_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest scPerturb h5ad and standardize metadata.")
    parser.add_argument("--input", required=True, help="Path to scPerturb .h5ad file.")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID to assign.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .h5ad path. Default: data/processed/<dataset_id>/<dataset_id>_v1.h5ad",
    )
    parser.add_argument("--preprocess-name", default="scperturb_passthrough")
    parser.add_argument("--preprocess-version", default="v1")
    parser.add_argument("--source", default="scPerturb Zenodo 13350497")
    parser.add_argument("--default-perturbation-type", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if args.out:
        output_path = Path(args.out)
    else:
        output_path = Path("data/processed") / args.dataset_id / f"{args.dataset_id}_{args.preprocess_version}.h5ad"

    default_type = args.default_perturbation_type or _infer_default_perturbation_type(args.dataset_id)
    cfg = IngestConfig(
        dataset_id=args.dataset_id,
        preprocess_name=args.preprocess_name,
        preprocess_version=args.preprocess_version,
        source=args.source,
        default_perturbation_type=default_type,
    )
    out = ingest_scperturb(input_path, output_path, cfg)
    print(f"Wrote processed dataset to {out}")


if __name__ == "__main__":
    main()
