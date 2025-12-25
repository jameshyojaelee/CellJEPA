#!/usr/bin/env python3
"""Create a validation split by partitioning train_groups from an existing split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def derive_split_name(split_name: str, val_frac: float) -> str:
    if "valsplit" in split_name.lower():
        return split_name
    suffix = f"valsplit{int(round(val_frac * 100))}"
    return f"{split_name}_{suffix}"


def split_train_groups(
    train_groups: list[str], val_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"--val-frac must be in (0, 1), got {val_frac}.")
    if not train_groups:
        raise ValueError("Split has no train_groups to partition.")

    groups = [str(g) for g in train_groups]
    if len(groups) < 2:
        return sorted(groups), []

    rng = random.Random(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_val = int(round(n_groups * val_frac))
    n_val = max(1, min(n_groups - 1, n_val))

    val_groups = sorted(groups[:n_val])
    train_groups = sorted(groups[n_val:])
    return train_groups, val_groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a validation split by partitioning train_groups from an existing split."
    )
    parser.add_argument("--in", dest="in_path", required=True, help="Input split JSON path.")
    parser.add_argument("--out", required=True, help="Output split JSON path.")
    parser.add_argument("--val-frac", type=float, required=True, help="Fraction of train_groups for val.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for train/val split.")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    split = json.loads(in_path.read_text(encoding="utf-8"))

    train_groups = split.get("train_groups")
    test_groups = split.get("test_groups")
    if train_groups is None or test_groups is None:
        raise KeyError("Input split JSON must define train_groups and test_groups.")

    new_train, new_val = split_train_groups(train_groups, args.val_frac, args.seed)

    split_name = split.get("split_name") or in_path.stem
    split_out = dict(split)
    split_out["split_name"] = derive_split_name(split_name, args.val_frac)
    split_out["seed"] = args.seed
    split_out["train_groups"] = new_train
    split_out["val_groups"] = new_val
    split_out["test_groups"] = list(test_groups)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split_out, indent=2), encoding="utf-8")
    if not new_val:
        print("Warning: val_groups is empty (not enough train_groups to split).")
    print(f"Wrote split file to {out_path}")


if __name__ == "__main__":
    main()
