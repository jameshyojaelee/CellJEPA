#!/usr/bin/env python3
"""Create a val split by carving from train groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a val split from existing split JSON.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split = json.loads(Path(args.in_path).read_text())
    train_groups = list(map(str, split.get("train_groups", [])))
    if not train_groups:
        raise ValueError("No train_groups found in input split.")

    rng = random.Random(args.seed)
    rng.shuffle(train_groups)
    n_val = max(1, int(len(train_groups) * args.val_frac))
    val_groups = train_groups[:n_val]
    new_train = train_groups[n_val:]

    split["train_groups"] = new_train
    split["val_groups"] = val_groups
    split["split_name"] = f"{split.get('split_name', 'split')}_valsplit{int(args.val_frac*100)}"
    split["seed_valsplit"] = args.seed
    split["val_frac"] = args.val_frac

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2))
    print(f"Wrote {out_path} (train={len(new_train)}, val={len(val_groups)})")


if __name__ == "__main__":
    main()
