#!/usr/bin/env python3
"""Summarize gene-set overlap across processed datasets for M4 planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import h5py


def read_var_names(path: Path) -> tuple[list[str], str]:
    with h5py.File(path, "r") as f:
        var = f["var"]
        index_key = var.attrs.get("_index")
        if index_key is None:
            raise ValueError(f"Missing var index in {path}")
        names = var[index_key][:]
        names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]
        return names, str(index_key)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gene overlap across datasets.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    datasets = [Path(p) for p in args.datasets]
    entries = []
    for path in datasets:
        names, index_key = read_var_names(path)
        entries.append({"path": str(path), "index_key": index_key, "n_genes": len(names), "genes": set(names)})

    all_genes = set().union(*[e["genes"] for e in entries])
    intersection = set(entries[0]["genes"])
    for e in entries[1:]:
        intersection &= e["genes"]

    lines = ["# M4 Gene Overlap Summary", ""]
    lines.append("## Dataset gene counts")
    lines.append("")
    lines.append("| dataset | n_genes | index_key |")
    lines.append("|---|---|---|")
    for e in entries:
        lines.append(f"| {Path(e['path']).name} | {e['n_genes']} | {e['index_key']} |")

    lines.append("")
    lines.append("## Global overlap")
    lines.append("")
    lines.append(f"- union genes: {len(all_genes)}")
    lines.append(f"- intersection genes: {len(intersection)}")

    lines.append("")
    lines.append("## Pairwise Jaccard")
    lines.append("")
    header = "| dataset | " + " | ".join([Path(e["path"]).name for e in entries]) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(entries) + 1)) + "|")
    for i, e in enumerate(entries):
        row = [Path(e["path"]).name]
        for j, f in enumerate(entries):
            row.append(f"{jaccard(e['genes'], f['genes']):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.json:
        out = {
            "datasets": [
                {"path": e["path"], "index_key": e["index_key"], "n_genes": e["n_genes"]}
                for e in entries
            ],
            "union_genes": len(all_genes),
            "intersection_genes": len(intersection),
        }
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
