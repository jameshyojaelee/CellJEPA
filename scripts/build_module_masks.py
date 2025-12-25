#!/usr/bin/env python3
"""Build a module mask JSON file from one or more GMT gene set files.

This keeps dependencies minimal and expects gene identifiers to match adata.var_names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def read_gmt(path: Path) -> list[tuple[str, list[str]]]:
    modules = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if not name or not genes:
                continue
            modules.append((name, genes))
    return modules


def load_gene_universe(path: Path) -> set[str]:
    # If .h5ad, read var index from H5AD attrs
    if path.suffix == ".h5ad":
        import h5py
        with h5py.File(path, "r") as f:
            var = f["var"]
            index_key = var.attrs.get("_index")
            if index_key is None:
                raise ValueError("Could not locate var index in .h5ad file.")
            names = var[index_key][:]
            return {n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names}

    # Otherwise treat as newline-delimited gene list
    genes = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.add(g)
    return genes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build module masks from GMT files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more GMT files.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--gene-universe", required=True, help=".h5ad path or newline gene list.")
    parser.add_argument("--min-genes", type=int, default=15)
    parser.add_argument("--max-genes", type=int, default=500)
    parser.add_argument("--dedup", action="store_true", help="Drop duplicate gene sets (by sorted genes).")
    args = parser.parse_args()

    universe = load_gene_universe(Path(args.gene_universe))
    all_modules = []

    for gmt in args.inputs:
        gmt_path = Path(gmt)
        for name, genes in read_gmt(gmt_path):
            filtered = [g for g in genes if g in universe]
            if len(filtered) < args.min_genes or len(filtered) > args.max_genes:
                continue
            all_modules.append({"name": name, "genes": sorted(set(filtered))})

    if args.dedup:
        seen = set()
        deduped = []
        for mod in all_modules:
            key = tuple(mod["genes"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(mod)
        all_modules = deduped

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_modules, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(all_modules)} modules)")


if __name__ == "__main__":
    main()
