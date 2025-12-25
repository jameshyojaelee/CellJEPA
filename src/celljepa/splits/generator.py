"""Deterministic split generation for CellJEPA."""

from __future__ import annotations

import random
from typing import Dict, List


SPLIT_GROUP_KEYS = {
    "S1_unseen_perturbation": "perturbation_id",
    "S2_unseen_context": "context_id",
}


def get_group_key(split_name: str) -> str:
    if split_name not in SPLIT_GROUP_KEYS:
        raise KeyError(f"Unknown split_name: {split_name}")
    return SPLIT_GROUP_KEYS[split_name]


def make_group_splits(
    groups: List[str], seed: int, k_folds: int = 5, fold: int = 0
) -> Dict[str, List[str]]:
    """Deterministically split group IDs into train/val/test sets."""
    if not groups:
        raise ValueError("No groups provided for splitting.")

    unique_groups = sorted(set(groups))
    k_folds = min(k_folds, len(unique_groups))
    if k_folds < 2:
        raise ValueError("Need at least 2 groups to split.")

    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    folds: List[List[str]] = [[] for _ in range(k_folds)]
    for idx, group in enumerate(unique_groups):
        folds[idx % k_folds].append(group)

    fold = fold % k_folds
    test_groups = folds[fold]
    val_groups = folds[(fold + 1) % k_folds] if k_folds > 2 else []
    train_groups = [
        g for i, f in enumerate(folds) if i not in {fold, (fold + 1) % k_folds} for g in f
    ]

    if not train_groups:
        # If too few groups, fall back to using remaining folds as train.
        train_groups = [g for i, f in enumerate(folds) if i != fold for g in f]
        val_groups = []

    return {
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
    }


def build_split_spec(
    dataset_id: str,
    split_name: str,
    seed: int,
    fold: int,
    train_groups: List[str],
    val_groups: List[str],
    test_groups: List[str],
) -> Dict[str, object]:
    """Construct the JSON-serializable split spec."""
    return {
        "dataset_id": dataset_id,
        "split_name": split_name,
        "seed": seed,
        "fold": fold,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "test_groups": test_groups,
        "group_key": get_group_key(split_name),
    }

