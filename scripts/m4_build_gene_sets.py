#!/usr/bin/env python3
"""Build gene-set lists for M4 harmonization using intersections."""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py


def read_var_names(path: Path) -> set[str]:
    with h5py.File(path, "r") as f:
        var = f["var"]
        index_key = var.attrs.get("_index")
        if index_key is None:
            raise ValueError(f"Missing var index in {path}")
        names = var[index_key][:]
        return {n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names}


def write_list(path: Path, genes: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(genes)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intersection gene lists for M4.")
    parser.add_argument("--drug", nargs="+", required=True, help="Drug dataset .h5ad paths (sciplex).")
    parser.add_argument("--genetic", nargs="+", required=True, help="Genetic dataset .h5ad paths (replogle/norman).")
    parser.add_argument("--out-dir", required=True, help="Output directory for gene lists.")
    args = parser.parse_args()

    drug_sets = [read_var_names(Path(p)) for p in args.drug]
    genetic_sets = [read_var_names(Path(p)) for p in args.genetic]

    intersection_drug = set.intersection(*drug_sets)
    intersection_genetic = set.intersection(*genetic_sets)
    intersection_all = intersection_drug & intersection_genetic

    out_dir = Path(args.out_dir)
    write_list(out_dir / "genes_intersection_sciplex_v1.txt", intersection_drug)
    write_list(out_dir / "genes_intersection_genetic_v1.txt", intersection_genetic)
    write_list(out_dir / "genes_intersection_all_v1.txt", intersection_all)

    print(f"drug intersection: {len(intersection_drug)}")
    print(f"genetic intersection: {len(intersection_genetic)}")
    print(f"all intersection: {len(intersection_all)}")


if __name__ == "__main__":
    main()
