#!/usr/bin/env python3
"""Train transition predictors in embedding space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import anndata as ad
import numpy as np
import torch

from celljepa.models.jepa import JEPA, JepaConfig
from celljepa.models.transition import PrototypePredictor, SetPredictor, TransitionConfig
from celljepa.train.transition_trainer import PairProto, PairSet, train_prototype, train_set, energy_distance_torch
from celljepa.eval.metrics import cosine_distance


def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def embed_cells(adata, checkpoint_path: Path, indices: np.ndarray, batch_size: int = 512, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = JepaConfig(**ckpt["config"])
    model = JEPA(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    X = adata.X
    indices = np.asarray(indices)
    out = np.zeros((indices.size, cfg.embed_dim), dtype=np.float32)

    for i in range(0, indices.size, batch_size):
        batch_idx = indices[i : i + batch_size]
        x = _to_dense(X[batch_idx]).astype(np.float32)
        with torch.no_grad():
            z = model.student(torch.from_numpy(x).to(device)).cpu().numpy()
        out[i : i + batch_idx.size] = z
    return out


def build_pairs(adata, max_cells_per_group: int | None = None):
    obs = adata.obs
    control_groups = {}
    pert_groups = {}
    for context_id, idx in obs[obs["is_control"]].groupby("context_id").indices.items():
        control_groups[context_id] = np.array(idx)
    for (context_id, perturbation_id), idx in obs[~obs["is_control"]].groupby(["context_id", "perturbation_id"]).indices.items():
        pert_groups[(context_id, perturbation_id)] = np.array(idx, dtype=np.int64)

    pairs_proto = []
    pairs_set = []
    for (context_id, perturbation_id), p_idx in pert_groups.items():
        c_idx = control_groups.get(context_id)
        if c_idx is None or p_idx.size == 0:
            continue
        if max_cells_per_group:
            c_idx = c_idx[: max_cells_per_group]
            p_idx = p_idx[: max_cells_per_group]
        pairs_set.append(
            PairSet(
                context_id=str(context_id),
                perturbation_id=str(perturbation_id),
                control_indices=np.array(c_idx, dtype=np.int64),
                pert_indices=np.array(p_idx, dtype=np.int64),
            )
        )
    return pairs_set


def split_pairs(pairs, group_key, split):
    train_groups = set(map(str, split["train_groups"]))
    val_groups = set(map(str, split["val_groups"]))
    test_groups = set(map(str, split["test_groups"]))

    train, val, test = [], [], []
    for p in pairs:
        group_val = p.perturbation_id if group_key == "perturbation_id" else p.context_id
        if group_val in test_groups:
            test.append(p)
        elif group_val in val_groups:
            val.append(p)
        else:
            train.append(p)
    return train, val, test


def eval_prototype(pairs: list[PairProto]) -> dict:
    mse = []
    cos = []
    skipped = 0
    for p in pairs:
        if not np.isfinite(p.control_proto).all() or not np.isfinite(p.pert_proto).all():
            skipped += 1
            continue
        diff = p.control_proto - p.pert_proto
        mse_val = float(np.mean(diff ** 2))
        cos_val = cosine_distance(p.control_proto, p.pert_proto)
        if not np.isfinite(mse_val) or not np.isfinite(cos_val):
            skipped += 1
            continue
        mse.append(mse_val)
        cos.append(cos_val)
    return {
        "mse_mean": float(np.mean(mse)) if mse else float("nan"),
        "cosine_mean": float(np.mean(cos)) if cos else float("nan"),
        "skipped_pairs": skipped,
        "n_eval": len(mse),
    }


def eval_set(
    model: SetPredictor,
    pairs: list[PairSet],
    embeddings: np.ndarray,
    pert_to_idx: dict[str, int],
    device: str,
    sample_size: int = 128,
) -> dict:
    rng = np.random.default_rng(0)
    losses = []
    skipped = 0
    for p in pairs:
        c_idx = p.control_indices
        t_idx = p.pert_indices
        if c_idx.size == 0 or t_idx.size == 0:
            skipped += 1
            continue
        c_sel = rng.choice(c_idx, size=min(sample_size, c_idx.size), replace=False)
        t_sel = rng.choice(t_idx, size=min(sample_size, t_idx.size), replace=False)

        c = torch.tensor(embeddings[c_sel], dtype=torch.float32, device=device)
        y = torch.tensor(embeddings[t_sel], dtype=torch.float32, device=device)
        if not torch.isfinite(c).all() or not torch.isfinite(y).all():
            skipped += 1
            continue
        idx = torch.tensor([pert_to_idx.get(p.perturbation_id, 0)], device=device, dtype=torch.long)
        with torch.no_grad():
            pred = model(c, idx)
            if not torch.isfinite(pred).all():
                skipped += 1
                continue
            loss = energy_distance_torch(pred, y)
        loss_val = float(loss.detach().cpu().numpy())
        if not np.isfinite(loss_val):
            skipped += 1
            continue
        losses.append(loss_val)

    return {
        "edist_mean": float(np.mean(losses)) if losses else float("nan"),
        "skipped_pairs": skipped,
        "n_eval": len(losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transition predictors (prototype or set).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["prototype", "set"], default="prototype")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--max-cells-per-group", type=int, default=5000)
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on number of condition pairs.")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.dataset)
    split = json.loads(Path(args.split).read_text())
    group_key = split["group_key"]

    pairs_set = build_pairs(adata, max_cells_per_group=args.max_cells_per_group)
    if args.max_pairs and len(pairs_set) > args.max_pairs:
        rng = np.random.default_rng(0)
        pairs_set = list(rng.choice(pairs_set, size=args.max_pairs, replace=False))
    train_pairs, val_pairs, test_pairs = split_pairs(pairs_set, group_key, split)

    # Build perturbation vocab from train only
    train_perturbations = sorted({p.perturbation_id for p in train_pairs})
    pert_to_idx = {"<UNK>": 0}
    for i, p in enumerate(train_perturbations, 1):
        pert_to_idx[p] = i

    # Collect indices to embed
    indices = np.unique(np.concatenate([p.control_indices for p in pairs_set] + [p.pert_indices for p in pairs_set]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = embed_cells(adata, Path(args.checkpoint), indices, device=device)
    idx_map = {idx: i for i, idx in enumerate(indices)}

    # remap to embedding indices
    for p in pairs_set:
        p.control_indices = np.array([idx_map[i] for i in p.control_indices if i in idx_map], dtype=np.int64)
        p.pert_indices = np.array([idx_map[i] for i in p.pert_indices if i in idx_map], dtype=np.int64)

    # drop empty pairs after remapping
    pairs_set = [p for p in pairs_set if p.control_indices.size > 0 and p.pert_indices.size > 0]

    cfg = TransitionConfig(embed_dim=args.embed_dim, perturbation_vocab=len(pert_to_idx), hidden_dim=args.hidden_dim)

    metrics = {"mode": args.mode, "n_train": len(train_pairs), "n_test": len(test_pairs)}
    if args.mode == "prototype":
        model = PrototypePredictor(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # build prototype pairs
        pairs_proto = []
        for p in train_pairs + test_pairs:
            c = embeddings[p.control_indices].mean(axis=0)
            y = embeddings[p.pert_indices].mean(axis=0)
            pairs_proto.append(PairProto(p.context_id, p.perturbation_id, c, y))
        train_proto = [p for p in pairs_proto if p.perturbation_id in train_perturbations]
        test_proto = [p for p in pairs_proto if p.perturbation_id not in train_perturbations]
        metrics.update(train_prototype(model, opt, train_proto, pert_to_idx, device, epochs=args.epochs, batch_size=args.batch_size))
        metrics["test"] = eval_prototype(test_proto) if test_proto else eval_prototype(pairs_proto)
        torch.save({"model": model.state_dict(), "pert_to_idx": pert_to_idx, "config": cfg.__dict__}, out_dir / "model.pt")
    else:
        model = SetPredictor(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        metrics.update(train_set(model, opt, train_pairs, embeddings, pert_to_idx, device, epochs=args.epochs, sample_size=args.sample_size))
        metrics["test"] = eval_set(model, test_pairs, embeddings, pert_to_idx, device, sample_size=args.sample_size)
        torch.save({"model": model.state_dict(), "pert_to_idx": pert_to_idx, "config": cfg.__dict__}, out_dir / "model.pt")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
