#!/usr/bin/env python3
"""Build a regulon database for regulon-aware masking.

Produces a JSON artifact mapping TF → [target genes].
Used by RegulonMask in celljepa.models.masking.

Data sources:
  - DoRothEA: TF → target gene regulons with confidence levels (A-E)
  - Custom: any JSON with {"TF_SYMBOL": ["TARGET1", "TARGET2", ...]} format

Usage:
  # From DoRothEA CSV:
  python scripts/build_regulon_masks.py \\
      --dorothea data/external/dorothea_hs.csv \\
      --confidence-levels A B C \\
      --gene-universe configs/harmonization/genes_intersection_genetic_v1.txt \\
      --min-targets 3 \\
      --out configs/regulons/dorothea_v1.json

  # Synthetic regulons for testing:
  python scripts/build_regulon_masks.py \\
      --gene-universe configs/harmonization/genes_intersection_genetic_v1.txt \\
      --synthetic --synthetic-n-regulons 50 --synthetic-targets-per-tf 20 \\
      --out configs/regulons/synthetic_test.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set


def load_gene_universe(path: Path) -> List[str]:
    """Load gene symbols from a newline-delimited text file."""
    genes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.append(g)
    return sorted(set(genes))


def load_dorothea(
    path: Path,
    confidence_levels: Set[str],
    gene_set: Set[str],
) -> Dict[str, List[str]]:
    """Load DoRothEA TF-target regulons.

    DoRothEA CSV format (typical columns):
      tf, confidence, target, mor (mode of regulation)

    We keep edges where both TF and target are in our gene universe
    and the confidence level is in the allowed set.
    """
    regulons: Dict[str, Set[str]] = defaultdict(set)

    with open(path, newline="", encoding="utf-8") as f:
        # Try to detect delimiter
        sample = f.read(4096)
        f.seek(0)
        if "\t" in sample:
            delimiter = "\t"
        else:
            delimiter = ","

        reader = csv.DictReader(f, delimiter=delimiter)
        # Normalize column names
        fieldnames = [c.strip().lower() for c in (reader.fieldnames or [])]

        # Find relevant columns
        tf_col = next((c for c in fieldnames if c in ("tf", "source", "regulator")), None)
        target_col = next((c for c in fieldnames if c in ("target", "gene")), None)
        conf_col = next((c for c in fieldnames if c in ("confidence", "confidence_level", "level", "mor_confidence")), None)

        if tf_col is None or target_col is None:
            raise ValueError(
                f"Could not identify TF/target columns. Found: {fieldnames}. "
                f"Expected 'tf'/'source' and 'target'/'gene' columns."
            )

        # Re-read with original fieldnames (DictReader uses originals)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        orig_fields = reader.fieldnames or []

        # Map normalized → original
        norm_to_orig = {c.strip().lower(): c for c in orig_fields}
        tf_key = norm_to_orig.get(tf_col, tf_col)
        target_key = norm_to_orig.get(target_col, target_col)
        conf_key = norm_to_orig.get(conf_col, conf_col) if conf_col else None

        for row in reader:
            tf = row.get(tf_key, "").strip()
            target = row.get(target_key, "").strip()

            if conf_key:
                conf = row.get(conf_key, "").strip().upper()
                if conf not in confidence_levels:
                    continue

            if tf in gene_set and target in gene_set and tf != target:
                regulons[tf].add(target)

    return {tf: sorted(targets) for tf, targets in regulons.items()}


def build_synthetic_regulons(
    genes: List[str],
    n_regulons: int = 50,
    targets_per_tf: int = 20,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Build synthetic regulons for testing.

    Randomly selects TFs and assigns random target genes.
    """
    random.seed(seed)
    tfs = random.sample(genes, min(n_regulons, len(genes)))
    remaining = [g for g in genes if g not in tfs]

    regulons = {}
    for tf in tfs:
        n_targets = min(targets_per_tf, len(remaining))
        targets = random.sample(remaining, n_targets)
        regulons[tf] = sorted(targets)

    return regulons


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regulon database for masking.")
    parser.add_argument("--dorothea", help="DoRothEA CSV/TSV file")
    parser.add_argument(
        "--confidence-levels",
        nargs="+",
        default=["A", "B", "C"],
        help="DoRothEA confidence levels to include (default: A B C)",
    )
    parser.add_argument("--custom-json", help="Custom TF→targets JSON file")
    parser.add_argument("--gene-universe", required=True, help="Newline-delimited gene symbols")
    parser.add_argument("--min-targets", type=int, default=3, help="Min targets per TF")
    parser.add_argument("--max-targets", type=int, default=500, help="Max targets per TF")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic regulons")
    parser.add_argument("--synthetic-n-regulons", type=int, default=50)
    parser.add_argument("--synthetic-targets-per-tf", type=int, default=20)
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    genes = load_gene_universe(Path(args.gene_universe))
    gene_set = set(genes)
    print(f"Gene universe: {len(genes)} genes")

    all_regulons: Dict[str, List[str]] = {}

    if args.synthetic:
        print(f"Building synthetic regulons (n={args.synthetic_n_regulons}, "
              f"targets_per_tf={args.synthetic_targets_per_tf})")
        all_regulons = build_synthetic_regulons(
            genes, args.synthetic_n_regulons,
            args.synthetic_targets_per_tf, args.synthetic_seed,
        )
    else:
        if args.dorothea:
            conf_set = {c.upper() for c in args.confidence_levels}
            print(f"Loading DoRothEA from {args.dorothea} (levels: {conf_set})")
            dorothea_regs = load_dorothea(Path(args.dorothea), conf_set, gene_set)
            all_regulons.update(dorothea_regs)
            print(f"  → {len(dorothea_regs)} TFs loaded from DoRothEA")

        if args.custom_json:
            print(f"Loading custom regulons from {args.custom_json}")
            with open(args.custom_json) as f:
                custom = json.load(f)
            for tf, targets in custom.items():
                filtered = [g for g in targets if g in gene_set and g != tf]
                if filtered:
                    existing = set(all_regulons.get(tf, []))
                    all_regulons[tf] = sorted(existing | set(filtered))
            print(f"  → merged, now {len(all_regulons)} TFs total")

    # Filter by target count
    filtered = {
        tf: targets
        for tf, targets in all_regulons.items()
        if args.min_targets <= len(targets) <= args.max_targets
    }
    removed = len(all_regulons) - len(filtered)
    if removed > 0:
        print(f"Filtered out {removed} TFs (target count outside [{args.min_targets}, {args.max_targets}])")

    # Statistics
    if filtered:
        target_counts = [len(t) for t in filtered.values()]
        print(f"Final: {len(filtered)} TFs, "
              f"targets per TF: min={min(target_counts)}, med={sorted(target_counts)[len(target_counts)//2]}, "
              f"max={max(target_counts)}, total unique edges={sum(target_counts)}")
    else:
        print("WARNING: No regulons passed filters. Use --synthetic for testing.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(filtered, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved regulon database to {out_path}")


if __name__ == "__main__":
    main()
