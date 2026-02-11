"""Gene-token–aware masking strategies for JEPA training.

All masking operates at the gene-token level: given a set of gene tokens,
return which tokens are visible (context) and which are masked (target).

Strategies:
- RandomGeneMask: uniformly random gene masking
- RegulonMask: mask a TF + its downstream targets together
- PathwayBlockMask: mask entire biological pathways
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class MaskResult:
    """Result of a masking operation.

    Attributes:
        visible_mask: (n_genes,) bool tensor, True for visible genes.
        target_mask: (n_genes,) bool tensor, True for masked/target genes.
        mask_ratio: float, fraction of genes masked.
        strategy: str, name of the masking strategy used.
        metadata: dict, optional metadata (e.g. which regulon was masked).
    """

    visible_mask: torch.Tensor
    target_mask: torch.Tensor
    mask_ratio: float
    strategy: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RandomGeneMask:
    """Randomly mask a fraction of gene tokens.

    This is the simplest strategy: each gene has an independent probability
    of being masked.
    """

    def __init__(self, mask_ratio: float = 0.25, min_context: int = 10):
        assert 0 < mask_ratio < 1, "mask_ratio must be in (0, 1)"
        self.mask_ratio = mask_ratio
        self.min_context = min_context

    def __call__(self, n_genes: int, device: torch.device = None) -> MaskResult:
        """Generate a random gene mask.

        Args:
            n_genes: number of genes to mask over.
            device: torch device for the mask tensors.

        Returns:
            MaskResult with random gene masking.
        """
        n_mask = max(1, min(int(n_genes * self.mask_ratio), n_genes - self.min_context))
        perm = torch.randperm(n_genes, device=device)
        target_mask = torch.zeros(n_genes, dtype=torch.bool, device=device)
        target_mask[perm[:n_mask]] = True
        visible_mask = ~target_mask

        return MaskResult(
            visible_mask=visible_mask,
            target_mask=target_mask,
            mask_ratio=n_mask / n_genes,
            strategy="random",
        )


class RegulonMask:
    """Mask a transcription factor and its known target genes together.

    This forces the model to predict the regulatory targets from the
    remaining context, learning TF-target relationships.

    Requires a regulon database (e.g. DoRothEA): TF -> [target genes].
    """

    def __init__(
        self,
        regulon_path: str | Path,
        gene_to_idx: Dict[str, int],
        min_targets: int = 3,
        max_mask_ratio: float = 0.4,
        fallback_random_ratio: float = 0.25,
    ):
        self.gene_to_idx = gene_to_idx
        self.min_targets = min_targets
        self.max_mask_ratio = max_mask_ratio
        self.fallback = RandomGeneMask(mask_ratio=fallback_random_ratio)

        # Load regulon database and filter to genes in our panel
        self.regulons = self._load_regulons(regulon_path)

    def _load_regulons(self, path: str | Path) -> List[Dict]:
        """Load regulon DB and filter to genes in our vocabulary."""
        with open(path) as f:
            raw = json.load(f)

        regulons = []
        for tf_name, targets in raw.items():
            tf_idx = self.gene_to_idx.get(tf_name)
            if tf_idx is None:
                continue
            target_idxs = [self.gene_to_idx[g] for g in targets if g in self.gene_to_idx]
            if len(target_idxs) >= self.min_targets:
                regulons.append({
                    "tf": tf_name,
                    "tf_idx": tf_idx,
                    "target_idxs": target_idxs,
                })

        return regulons

    def __call__(self, n_genes: int, device: torch.device = None) -> MaskResult:
        """Mask a randomly chosen regulon."""
        if not self.regulons:
            return self.fallback(n_genes, device)

        reg = random.choice(self.regulons)
        max_targets = int(n_genes * self.max_mask_ratio) - 1  # -1 for the TF itself

        target_mask = torch.zeros(n_genes, dtype=torch.bool, device=device)
        target_mask[reg["tf_idx"]] = True

        target_idxs = reg["target_idxs"]
        if len(target_idxs) > max_targets:
            target_idxs = random.sample(target_idxs, max_targets)
        for idx in target_idxs:
            if idx < n_genes:
                target_mask[idx] = True

        visible_mask = ~target_mask
        n_masked = target_mask.sum().item()

        return MaskResult(
            visible_mask=visible_mask,
            target_mask=target_mask,
            mask_ratio=n_masked / n_genes,
            strategy="regulon",
            metadata={"tf": reg["tf"], "n_targets": len(target_idxs)},
        )


class PathwayBlockMask:
    """Mask entire biological pathways (MSigDB, Reactome, GO).

    Similar to regulon masking but at the pathway level: mask all genes
    in a randomly selected pathway.
    """

    def __init__(
        self,
        pathway_file: str | Path,
        gene_to_idx: Dict[str, int],
        min_genes: int = 5,
        max_mask_ratio: float = 0.4,
        fallback_random_ratio: float = 0.25,
    ):
        self.gene_to_idx = gene_to_idx
        self.max_mask_ratio = max_mask_ratio
        self.fallback = RandomGeneMask(mask_ratio=fallback_random_ratio)

        self.pathways = self._load_pathways(pathway_file, min_genes)

    def _load_pathways(self, path: str | Path, min_genes: int) -> List[Dict]:
        """Load pathway gene sets."""
        with open(path) as f:
            raw = json.load(f)

        pathways = []
        for name, genes in raw.items():
            gene_idxs = [self.gene_to_idx[g] for g in genes if g in self.gene_to_idx]
            if len(gene_idxs) >= min_genes:
                pathways.append({"name": name, "gene_idxs": gene_idxs})

        return pathways

    def __call__(self, n_genes: int, device: torch.device = None) -> MaskResult:
        """Mask a randomly chosen pathway."""
        if not self.pathways:
            return self.fallback(n_genes, device)

        pw = random.choice(self.pathways)
        max_mask = int(n_genes * self.max_mask_ratio)

        target_mask = torch.zeros(n_genes, dtype=torch.bool, device=device)
        idxs = pw["gene_idxs"]
        if len(idxs) > max_mask:
            idxs = random.sample(idxs, max_mask)
        for idx in idxs:
            if idx < n_genes:
                target_mask[idx] = True

        visible_mask = ~target_mask
        n_masked = target_mask.sum().item()

        return MaskResult(
            visible_mask=visible_mask,
            target_mask=target_mask,
            mask_ratio=n_masked / n_genes,
            strategy="pathway",
            metadata={"pathway": pw["name"], "n_genes": len(idxs)},
        )


def batch_masks(
    mask_fn,
    batch_size: int,
    n_genes: int,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[MaskResult]]:
    """Generate masks for a batch, returning stacked tensors.

    For efficiency, all cells in a batch currently share the same mask
    (this can be extended to per-cell masks later).

    Args:
        mask_fn: callable that takes (n_genes, device) -> MaskResult
        batch_size: number of cells in the batch
        n_genes: number of genes
        device: torch device

    Returns:
        visible_mask: (batch, n_genes) bool tensor
        target_mask: (batch, n_genes) bool tensor
        mask_results: list of MaskResult (one per unique mask)
    """
    result = mask_fn(n_genes, device)
    visible = result.visible_mask.unsqueeze(0).expand(batch_size, -1)
    target = result.target_mask.unsqueeze(0).expand(batch_size, -1)
    return visible, target, [result]
