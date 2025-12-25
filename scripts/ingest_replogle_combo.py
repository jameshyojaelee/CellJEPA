#!/usr/bin/env python3
"""Ingest and merge Replogle K562 essential + RPE1 datasets into a combined dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from celljepa.data.ingest_replogle_combo import ComboConfig, ingest_replogle_combo


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Replogle K562 essential + RPE1 into a combined dataset.")
    parser.add_argument("--k562", required=True, help="Path to Replogle K562 essential .h5ad")
    parser.add_argument("--rpe1", required=True, help="Path to Replogle RPE1 .h5ad")
    parser.add_argument("--dataset-id", default="replogle_k562_rpe1")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .h5ad path. Default: data/processed/<dataset_id>/<dataset_id>_v1.h5ad",
    )
    parser.add_argument("--preprocess-name", default="scperturb_passthrough")
    parser.add_argument("--preprocess-version", default="v1")
    parser.add_argument("--source", default="scPerturb Zenodo 13350497 (Replogle 2022)")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path("data/processed") / args.dataset_id / f"{args.dataset_id}_{args.preprocess_version}.h5ad"

    cfg = ComboConfig(
        dataset_id=args.dataset_id,
        preprocess_name=args.preprocess_name,
        preprocess_version=args.preprocess_version,
        source=args.source,
    )

    out_path = ingest_replogle_combo(
        k562_path=Path(args.k562),
        rpe1_path=Path(args.rpe1),
        output_path=out,
        config=cfg,
    )
    print(f"Wrote combined dataset to {out_path}")


if __name__ == "__main__":
    main()

