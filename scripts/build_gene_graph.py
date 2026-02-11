#!/usr/bin/env python3
"""Build a gene interaction graph (PPI + GO co-annotation) for GNN encoders.

Produces a PyTorch artifact containing:
  - edge_index: (2, n_edges) long tensor
  - gene_names: list of gene symbols (defines node ordering)
  - n_genes: int, number of nodes
  - metadata: dict with source info and build parameters

Data sources:
  - STRING-db protein links (experimental + database channels, score ≥ 400)
  - Gene Ontology co-annotation edges (optional)

Usage:
  # From downloaded STRING file:
  python scripts/build_gene_graph.py \\
      --string-links data/external/9606.protein.links.v12.0.txt.gz \\
      --string-aliases data/external/9606.protein.aliases.v12.0.txt.gz \\
      --gene-universe configs/harmonization/genes_intersection_genetic_v1.txt \\
      --min-score 400 \\
      --out configs/graphs/ppi_go_graph_v1.pt

  # Synthetic graph for testing (no external data needed):
  python scripts/build_gene_graph.py \\
      --gene-universe configs/harmonization/genes_intersection_genetic_v1.txt \\
      --synthetic --synthetic-density 0.001 \\
      --out configs/graphs/synthetic_test_graph.pt
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch


def load_gene_universe(path: Path) -> List[str]:
    """Load gene symbols from a newline-delimited text file."""
    genes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.append(g)
    return sorted(set(genes))


def load_string_aliases(path: Path) -> Dict[str, str]:
    """Load STRING protein aliases → gene symbol mapping.

    STRING uses Ensembl protein IDs (e.g. 9606.ENSP00000000233).
    The aliases file maps these to gene symbols.
    """
    alias_map: Dict[str, str] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[call-overload]
        for line in f:
            if line.startswith("#") or line.startswith("string_protein_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            string_id, alias, source = parts[0], parts[1], parts[2]
            # Prefer gene symbol sources
            if "BioMart_HUGO" in source or "Ensembl_HGNC" in source or "BLAST_UniProt_GN" in source:
                alias_map[string_id] = alias
    return alias_map


def load_string_links(
    path: Path,
    alias_map: Dict[str, str],
    gene_set: Set[str],
    min_score: int = 400,
) -> List[Tuple[str, str, int]]:
    """Load STRING protein-protein interactions, filtered to our gene set.

    Args:
        path: Path to STRING protein.links file (optionally gzipped).
        alias_map: STRING protein ID → gene symbol mapping.
        gene_set: Set of gene symbols we care about.
        min_score: Minimum combined score threshold.

    Returns:
        List of (gene_a, gene_b, score) tuples.
    """
    edges = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[call-overload]
        for line in f:
            if line.startswith("protein1") or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            p1, p2, score = parts[0], parts[1], int(parts[2])
            if score < min_score:
                continue
            g1 = alias_map.get(p1)
            g2 = alias_map.get(p2)
            if g1 and g2 and g1 in gene_set and g2 in gene_set and g1 != g2:
                edges.append((g1, g2, score))
    return edges


def load_go_coannot(
    path: Path,
    gene_set: Set[str],
    min_shared_terms: int = 3,
) -> List[Tuple[str, str]]:
    """Load Gene Ontology co-annotation edges.

    Two genes are connected if they share ≥ min_shared_terms GO terms.

    Expects JSON format: {"GENE_SYMBOL": ["GO:0001234", ...], ...}
    """
    with open(path) as f:
        go_annot = json.load(f)

    # Filter to our gene universe
    filtered = {g: set(terms) for g, terms in go_annot.items() if g in gene_set}
    genes = sorted(filtered.keys())

    edges = []
    for i, g1 in enumerate(genes):
        for g2 in genes[i + 1:]:
            shared = len(filtered[g1] & filtered[g2])
            if shared >= min_shared_terms:
                edges.append((g1, g2))

    return edges


def build_synthetic_graph(
    genes: List[str],
    density: float = 0.001,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Build a synthetic random graph for testing.

    Connects random pairs of genes with the given edge density
    (fraction of all possible edges).
    """
    random.seed(seed)
    n = len(genes)
    n_possible = n * (n - 1) // 2
    n_edges = max(1, int(n_possible * density))

    edges = set()
    while len(edges) < n_edges:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            pair = (min(i, j), max(i, j))
            edges.add(pair)

    return [(genes[i], genes[j]) for i, j in edges]


def edges_to_tensor(
    edges: List[Tuple[str, str]],
    gene_to_idx: Dict[str, int],
) -> torch.Tensor:
    """Convert edge list to bidirectional edge_index tensor."""
    src, dst = [], []
    for g1, g2 in edges:
        i, j = gene_to_idx.get(g1), gene_to_idx.get(g2)
        if i is not None and j is not None:
            # Bidirectional
            src.extend([i, j])
            dst.extend([j, i])
    return torch.tensor([src, dst], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gene interaction graph.")
    parser.add_argument("--string-links", help="STRING protein.links file (optionally .gz)")
    parser.add_argument("--string-aliases", help="STRING protein.aliases file (optionally .gz)")
    parser.add_argument("--go-annotations", help="GO gene annotations JSON")
    parser.add_argument("--go-min-shared", type=int, default=3, help="Min shared GO terms for edge")
    parser.add_argument("--gene-universe", required=True, help="Newline-delimited gene symbols")
    parser.add_argument("--min-score", type=int, default=400, help="Min STRING combined score")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic graph")
    parser.add_argument("--synthetic-density", type=float, default=0.001, help="Edge density for synthetic")
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output .pt path")
    args = parser.parse_args()

    genes = load_gene_universe(Path(args.gene_universe))
    gene_set = set(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    print(f"Gene universe: {len(genes)} genes")

    all_edge_pairs: List[Tuple[str, str]] = []
    source_info: Dict[str, str] = {}

    if args.synthetic:
        print(f"Building synthetic graph (density={args.synthetic_density}, seed={args.synthetic_seed})")
        pairs = build_synthetic_graph(genes, args.synthetic_density, args.synthetic_seed)
        all_edge_pairs.extend(pairs)
        source_info["type"] = "synthetic"
        source_info["density"] = str(args.synthetic_density)
        source_info["seed"] = str(args.synthetic_seed)
    else:
        if args.string_links and args.string_aliases:
            print(f"Loading STRING aliases from {args.string_aliases}")
            aliases = load_string_aliases(Path(args.string_aliases))
            print(f"  → {len(aliases)} alias mappings")

            print(f"Loading STRING links from {args.string_links} (min_score={args.min_score})")
            string_edges = load_string_links(
                Path(args.string_links), aliases, gene_set, args.min_score,
            )
            all_edge_pairs.extend([(g1, g2) for g1, g2, _ in string_edges])
            print(f"  → {len(string_edges)} STRING edges (in gene universe)")
            source_info["string"] = f"min_score={args.min_score}, n_edges={len(string_edges)}"

        if args.go_annotations:
            print(f"Loading GO co-annotations from {args.go_annotations}")
            go_edges = load_go_coannot(
                Path(args.go_annotations), gene_set, args.go_min_shared,
            )
            all_edge_pairs.extend(go_edges)
            print(f"  → {len(go_edges)} GO co-annotation edges")
            source_info["go"] = f"min_shared={args.go_min_shared}, n_edges={len(go_edges)}"

        if not all_edge_pairs:
            print("WARNING: No edges loaded. Use --synthetic for testing or provide data sources.")

    # Deduplicate
    unique_pairs = set()
    for g1, g2 in all_edge_pairs:
        pair = (min(g1, g2), max(g1, g2))
        unique_pairs.add(pair)
    all_edge_pairs = list(unique_pairs)

    edge_index = edges_to_tensor(all_edge_pairs, gene_to_idx)
    print(f"Final graph: {len(genes)} nodes, {edge_index.shape[1]} directed edges "
          f"({len(all_edge_pairs)} undirected)")

    # Count connected nodes
    if edge_index.shape[1] > 0:
        connected = torch.unique(edge_index).numel()
    else:
        connected = 0
    print(f"Connected nodes: {connected}/{len(genes)} ({connected/len(genes)*100:.1f}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "edge_index": edge_index,
        "gene_names": genes,
        "gene_to_idx": gene_to_idx,
        "n_genes": len(genes),
        "n_edges_undirected": len(all_edge_pairs),
        "metadata": {
            "sources": source_info,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gene_universe_path": str(args.gene_universe),
        },
    }, out_path)
    print(f"Saved graph to {out_path}")


if __name__ == "__main__":
    main()
