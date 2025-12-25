#!/usr/bin/env python3
"""Generate a tiny toy dataset that conforms to the CellJEPA data contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _require_anndata():
    try:
        import anndata as ad  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "anndata is required to write .h5ad files. "
            "Install it in your environment before proceeding."
        ) from exc
    return ad


def make_toy_dataset(
    dataset_id: str,
    out_path: Path,
    seed: int = 0,
    n_cells_per_group: int = 25,
    n_genes: int = 20,
) -> None:
    rng = np.random.default_rng(seed)

    contexts = ["ctx_A", "ctx_B"]
    perturbations = ["control", "gene:STAT1", "drug:DEX"]

    rows = []
    X_rows = []
    for ctx in contexts:
        for pert in perturbations:
            is_control = pert == "control"
            for _ in range(n_cells_per_group):
                base = rng.poisson(1.0, size=n_genes).astype(float)
                if not is_control:
                    base += rng.normal(loc=0.5, scale=0.2, size=n_genes)
                expr = np.log1p(np.clip(base, a_min=0.0, a_max=None))
                X_rows.append(expr)
                rows.append(
                    {
                        "perturbation_id": pert,
                        "is_control": is_control,
                        "context_id": ctx,
                        "perturbation_tokens": "control:CTRL" if is_control else pert,
                        "dose": 1.0 if not is_control else np.nan,
                        "time_hours": 24.0 if not is_control else np.nan,
                    }
                )

    X = np.vstack(X_rows)
    obs = pd.DataFrame(rows)
    var = pd.DataFrame(index=[f"ENSG{i:05d}" for i in range(n_genes)])

    ad = _require_anndata()
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["dataset_id"] = dataset_id
    adata.uns["preprocess_name"] = "libnorm_log1p_v1"
    adata.uns["preprocess_version"] = "v1"
    adata.uns["created_at"] = datetime.now(timezone.utc).isoformat()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a toy CellJEPA dataset (.h5ad).")
    parser.add_argument("--dataset-id", default="toy", help="Dataset ID to embed in the file.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .h5ad path. Default: data/processed/<dataset_id>/<dataset_id>_toy.h5ad",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cells-per-group", type=int, default=25)
    parser.add_argument("--n-genes", type=int, default=20)
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("data/processed") / args.dataset_id / f"{args.dataset_id}_toy.h5ad"

    make_toy_dataset(
        dataset_id=args.dataset_id,
        out_path=out_path,
        seed=args.seed,
        n_cells_per_group=args.n_cells_per_group,
        n_genes=args.n_genes,
    )
    print(f"Wrote toy dataset to {out_path}")


if __name__ == "__main__":
    main()

