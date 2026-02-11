#!/usr/bin/env python3
"""V2 JEPA pretraining script — integrates P1-P4 modules.

Supports:
- JEPAv2 with pluggable encoder backends (Transformer, GNN, Perceiver)
- Gene tokenization via GeneTokenizer
- Streaming multi-dataset loading via CellDataset / MultiDatasetMixer
- Multi-GPU DDP training via torchrun/SLURM
- Checkpoint resume, gradient accumulation, LR warmup + cosine decay
- Gene-token masking strategies (random, regulon, pathway)

Usage (single GPU):
    python3 scripts/pretrain_jepa_v2.py \\
        --datasets data/processed/replogle/processed.h5ad \\
        --out runs/pretrain_v2 \\
        --encoder transformer

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 scripts/pretrain_jepa_v2.py \\
        --datasets data/processed/replogle/processed.h5ad \\
        --out runs/pretrain_v2_multi \\
        --encoder transformer

Usage (SLURM):
    srun --ntasks-per-node=4 python3 scripts/pretrain_jepa_v2.py \\
        --datasets data/processed/replogle/processed.h5ad \\
        --out runs/pretrain_v2_slurm \\
        --encoder transformer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from celljepa.models.jepa import JEPAv2, JEPAv2Config, variance_covariance_loss
from celljepa.models.gene_tokenizer import GeneTokenizerConfig
from celljepa.models.masking import RandomGeneMask
from celljepa.data.streaming_dataset import CellDataset, CellDatasetConfig, MultiDatasetMixer
from celljepa.train.distributed import (
    DistributedTrainConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_ddp_model,
    get_distributed_sampler,
    effective_batch_size,
    get_lr_scheduler,
)


def collate_variable_genes(batch):
    """Collate cells with variable numbers of expressed genes.

    Pads to the maximum gene count in the batch.

    Returns:
        expression: (B, max_genes) padded expression values
        gene_ids: (B, max_genes) padded gene IDs (0 for padding)
        mask: (B, max_genes) bool mask (True = real gene, False = padding)
    """
    exprs, gids = zip(*batch)
    max_len = max(e.shape[0] for e in exprs)
    batch_size = len(batch)

    expression = torch.zeros(batch_size, max_len, dtype=torch.float32)
    gene_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (e, g) in enumerate(zip(exprs, gids)):
        n = e.shape[0]
        expression[i, :n] = e
        gene_ids[i, :n] = g
        mask[i, :n] = True

    return expression, gene_ids, mask


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    out_dir: Path,
    cfg_dict: dict,
) -> Path:
    """Save training checkpoint with resume capability."""
    # Unwrap DDP
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    ckpt = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "config": cfg_dict,
    }
    path = out_dir / f"checkpoint_epoch{epoch:04d}.pt"
    torch.save(ckpt, path)
    # Symlink latest
    latest = out_dir / "checkpoint_latest.pt"
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.name)
    return path


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loader: DataLoader,
    device: torch.device,
    cfg: JEPAv2Config,
    grad_accum: int = 1,
    max_grad_norm: float = 1.0,
    mask_ratio: float = 0.25,
) -> dict:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_repr = 0.0
    total_var = 0.0
    total_cov = 0.0
    n_steps = 0

    masker = RandomGeneMask(mask_ratio=mask_ratio)

    for batch_idx, (expression, gene_ids, valid_mask) in enumerate(loader):
        expression = expression.to(device)
        gene_ids = gene_ids.to(device)

        # Get number of genes in this batch
        n_genes = gene_ids.shape[1]

        # Generate mask (using first sample's gene count)
        mask_result = masker(n_genes)

        # Forward pass
        unwrapped = model.module if hasattr(model, "module") else model
        outputs = unwrapped(expression, gene_ids[0], mask_result)

        # Compute loss
        loss_dict = unwrapped.compute_loss(outputs)
        loss = loss_dict["loss"] / grad_accum

        # Backward
        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()

            # EMA update
            unwrapped.update_teacher()

        total_loss += loss_dict["loss"].item()
        total_repr += loss_dict["repr_loss"].item()
        total_var += loss_dict["var_loss"].item()
        total_cov += loss_dict["cov_loss"].item()
        n_steps += 1

    return {
        "loss": total_loss / max(n_steps, 1),
        "repr_loss": total_repr / max(n_steps, 1),
        "var_loss": total_var / max(n_steps, 1),
        "cov_loss": total_cov / max(n_steps, 1),
        "n_steps": n_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 JEPA Pretraining (P1-P4)")

    # Data
    parser.add_argument("--datasets", nargs="+", required=True, help="h5ad file path(s)")
    parser.add_argument("--gene-vocab", type=str, default=None, help="Shared gene vocabulary file")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Dataset mixing weights")
    parser.add_argument("--max-genes", type=int, default=5000, help="Max genes per cell")

    # Model
    parser.add_argument("--encoder", choices=["transformer", "gnn", "perceiver"], default="transformer")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--loss", choices=["mse", "smooth_l1", "cosine"], default="smooth_l1")

    # Training
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Setup distributed
    setup_distributed()
    dist_cfg = DistributedTrainConfig.from_env(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )

    set_seed(args.seed + dist_cfg.rank)
    device = torch.device(f"cuda:{dist_cfg.local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    ds_config = CellDatasetConfig(
        gene_vocab_path=args.gene_vocab,
        max_genes_per_cell=args.max_genes,
    )

    datasets = []
    for path in args.datasets:
        ds = CellDataset(path, ds_config)
        datasets.append(ds)
        if is_main_process():
            print(f"  Dataset: {path} ({ds.n_cells:,} cells, {ds.mapped_gene_count:,} mapped genes)")

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = MultiDatasetMixer(datasets, weights=args.weights, seed=args.seed)

    # DataLoader
    sampler = get_distributed_sampler(dataset, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        collate_fn=collate_variable_genes,
        pin_memory=True,
        drop_last=True,
    )

    # Build model
    encoder_kwargs = {
        "embed_dim": args.embed_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
    }
    jepa_cfg = JEPAv2Config(
        tokenizer=GeneTokenizerConfig(embed_dim=args.embed_dim),
        encoder_type=args.encoder,
        encoder_kwargs=encoder_kwargs,
        loss_type=args.loss,
    )
    model = JEPAv2(jepa_cfg).to(device)
    model = get_ddp_model(model, dist_cfg.local_rank)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(loader) * args.epochs // args.grad_accum
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") and scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    if is_main_process():
        eff_bs = effective_batch_size(args.batch_size, args.grad_accum, dist_cfg.world_size)
        print(f"\nPretraining: encoder={args.encoder}, epochs={args.epochs}, "
              f"eff_batch_size={eff_bs}, lr={args.lr}")
        print(f"Device: {device}, world_size={dist_cfg.world_size}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)

        t0 = time.time()
        metrics = train_epoch(
            model, optimizer, scheduler, loader,
            device=device, cfg=jepa_cfg,
            grad_accum=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            mask_ratio=args.mask_ratio,
        )
        elapsed = time.time() - t0
        metrics["epoch"] = epoch
        metrics["time_s"] = elapsed
        history.append(metrics)

        if is_main_process():
            print(f"epoch {epoch:3d} | loss={metrics['loss']:.4f} repr={metrics['repr_loss']:.4f} "
                  f"var={metrics['var_loss']:.4f} cov={metrics['cov_loss']:.4f} | "
                  f"{elapsed:.1f}s ({metrics['n_steps']} steps)")

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, metrics["n_steps"],
                    out_dir, {"jepa": jepa_cfg.__dict__, "args": vars(args)},
                )

    # Save final logs
    if is_main_process():
        (out_dir / "metrics.json").write_text(
            json.dumps({"history": history}, indent=2, default=str), encoding="utf-8"
        )
        (out_dir / "config.json").write_text(
            json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
        )
        print(f"\nDone. Output: {out_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
