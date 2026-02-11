#!/usr/bin/env python3
"""Smoke test for P4 data infrastructure.

Tests streaming dataset, multi-dataset mixer, dataset catalog, and
distributed training utilities. No external data required — uses
synthetic h5ad files created in a temp directory.

Usage:
    python3 scripts/test_data_infrastructure.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers: create synthetic h5ad + gene vocab for testing
# ---------------------------------------------------------------------------

def _create_synthetic_h5ad(path: Path, n_cells: int = 100, n_genes: int = 50) -> list:
    """Create a minimal synthetic h5ad file."""
    try:
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp
    except ImportError:
        raise ImportError("anndata, pandas, scipy required for smoke test")

    rng = np.random.default_rng(42)
    X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=rng)
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    obs = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path)
    return gene_names


def _create_gene_vocab(path: Path, genes: list) -> None:
    """Write a gene vocabulary file."""
    with open(path, "w") as f:
        for g in genes:
            f.write(g + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cell_dataset() -> bool:
    print("--- CellDataset ---")
    from celljepa.data.streaming_dataset import CellDataset, CellDatasetConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        h5ad_path = Path(tmpdir) / "test.h5ad"
        vocab_path = Path(tmpdir) / "genes.txt"
        genes = _create_synthetic_h5ad(h5ad_path, n_cells=50, n_genes=30)
        _create_gene_vocab(vocab_path, genes[:25])  # vocab covers 25/30 genes

        cfg = CellDatasetConfig(gene_vocab_path=str(vocab_path))
        ds = CellDataset(h5ad_path, cfg)

        assert len(ds) == 50, f"Expected 50 cells, got {len(ds)}"
        print(f"  Length: {len(ds)} ✓")
        print(f"  Mapped genes: {ds.mapped_gene_count} / {len(ds.dataset_genes)} ✓")
        assert ds.mapped_gene_count == 25

        expr, gids = ds[0]
        assert expr.dtype == torch.float32
        assert gids.dtype == torch.long
        assert expr.shape == gids.shape
        print(f"  Item shapes: expr={expr.shape}, gene_ids={gids.shape} ✓")

        # Gene IDs should be in [1, 25] (1-indexed, within vocab)
        assert gids.min() >= 0 and gids.max() <= 25
        print(f"  Gene ID range: [{gids.min()}, {gids.max()}] ✓")

    return True


def test_cell_dataset_no_vocab() -> bool:
    print("--- CellDataset (no vocab) ---")
    from celljepa.data.streaming_dataset import CellDataset, CellDatasetConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        h5ad_path = Path(tmpdir) / "test.h5ad"
        _create_synthetic_h5ad(h5ad_path, n_cells=20, n_genes=40)

        cfg = CellDatasetConfig()
        ds = CellDataset(h5ad_path, cfg)

        assert len(ds) == 20
        assert ds.mapped_gene_count == 40
        print(f"  No vocab: all {ds.mapped_gene_count} genes mapped ✓")

        expr, gids = ds[0]
        assert expr.shape[0] > 0
        print(f"  Item: {expr.shape} genes expressed ✓")

    return True


def test_multi_dataset_mixer() -> bool:
    print("--- MultiDatasetMixer ---")
    from celljepa.data.streaming_dataset import CellDataset, CellDatasetConfig, MultiDatasetMixer

    with tempfile.TemporaryDirectory() as tmpdir:
        genes_all = [f"GENE_{i}" for i in range(50)]
        vocab_path = Path(tmpdir) / "genes.txt"
        _create_gene_vocab(vocab_path, genes_all)
        cfg = CellDatasetConfig(gene_vocab_path=str(vocab_path))

        # Create two datasets of different sizes
        h5_a = Path(tmpdir) / "ds_a.h5ad"
        h5_b = Path(tmpdir) / "ds_b.h5ad"
        _create_synthetic_h5ad(h5_a, n_cells=80, n_genes=50)
        _create_synthetic_h5ad(h5_b, n_cells=20, n_genes=50)

        ds_a = CellDataset(h5_a, cfg)
        ds_b = CellDataset(h5_b, cfg)

        mixer = MultiDatasetMixer([ds_a, ds_b], weights=[0.7, 0.3], seed=42)
        assert len(mixer) > 0
        print(f"  Mixed size: {len(mixer)} ✓")

        expr, gids = mixer[0]
        assert expr.dtype == torch.float32
        print(f"  Item from mixer: {expr.shape} ✓")

        # Test reshuffling
        mixer.reshuffle(seed=123)
        expr2, gids2 = mixer[0]
        print(f"  After reshuffle: {expr2.shape} ✓")

        # Test weights property
        w = mixer.weights
        assert abs(w.sum() - 1.0) < 1e-6
        print(f"  Weights sum: {w.sum():.4f} ✓")

    return True


def test_dataset_catalog() -> bool:
    print("--- Dataset Catalog ---")
    from celljepa.data.dataset_catalog import (
        CATALOG, get_perturbation_datasets, get_pretraining_datasets,
        get_datasets_by_type,
    )

    assert len(CATALOG) >= 12, f"Expected ≥12 datasets, got {len(CATALOG)}"
    print(f"  Total datasets: {len(CATALOG)} ✓")

    pert = get_perturbation_datasets()
    assert len(pert) >= 10
    print(f"  Perturbation datasets: {len(pert)} ✓")

    pretrain = get_pretraining_datasets()
    assert len(pretrain) >= 2
    print(f"  Pretraining datasets: {len(pretrain)} ✓")

    genetic = get_datasets_by_type("genetic")
    drug = get_datasets_by_type("drug")
    print(f"  Genetic: {len(genetic)}, Drug: {len(drug)} ✓")

    # Check all required fields are populated
    for ds_id, info in CATALOG.items():
        assert info.dataset_id, f"Missing dataset_id for {ds_id}"
        assert info.name, f"Missing name for {ds_id}"
        assert info.organism in ("human", "mouse"), f"Bad organism for {ds_id}: {info.organism}"
        assert info.n_cells_approx > 0, f"Zero cells for {ds_id}"
    print(f"  All metadata fields valid ✓")

    return True


def test_distributed_config() -> bool:
    print("--- DistributedTrainConfig ---")
    from celljepa.train.distributed import (
        DistributedTrainConfig, is_main_process, effective_batch_size,
        get_lr_scheduler,
    )

    cfg = DistributedTrainConfig(batch_size=64, gradient_accumulation_steps=4)
    assert cfg.batch_size == 64
    print(f"  Config: batch={cfg.batch_size}, accum={cfg.gradient_accumulation_steps} ✓")

    assert is_main_process() == True, "Should be main process in single-GPU mode"
    print(f"  is_main_process: True ✓")

    eff_bs = effective_batch_size(64, 4, 2)
    assert eff_bs == 512, f"Expected 512, got {eff_bs}"
    print(f"  Effective batch size: {eff_bs} ✓")

    # Test LR scheduler
    model = torch.nn.Linear(10, 10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = get_lr_scheduler(opt, warmup_steps=10, total_steps=100)
    # Step through warmup
    for _ in range(10):
        sched.step()
    lr_at_warmup = opt.param_groups[0]["lr"]
    assert abs(lr_at_warmup - 1e-3) < 1e-6, f"LR at warmup end: {lr_at_warmup}"
    print(f"  LR scheduler warmup: {lr_at_warmup:.6f} ✓")

    # Step past warmup into decay
    for _ in range(45):
        sched.step()
    lr_mid = opt.param_groups[0]["lr"]
    assert lr_mid < lr_at_warmup
    print(f"  LR scheduler decay: {lr_mid:.6f} ✓")

    return True


def test_collate_fn() -> bool:
    print("--- collate_variable_genes ---")
    # Simulate the import and test
    sys.path.insert(0, str(ROOT / "scripts"))
    from pretrain_jepa_v2 import collate_variable_genes

    batch = [
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5, 10, 15])),
        (torch.tensor([4.0, 5.0]), torch.tensor([3, 7])),
    ]
    expression, gene_ids, mask = collate_variable_genes(batch)
    assert expression.shape == (2, 3)
    assert gene_ids.shape == (2, 3)
    assert mask.shape == (2, 3)
    assert mask[0].all()  # first sample has 3 genes, no padding
    assert mask[1, :2].all() and not mask[1, 2]  # second has 2 real + 1 pad
    print(f"  Expression: {expression.shape} ✓")
    print(f"  Gene IDs: {gene_ids.shape} ✓")
    print(f"  Padding mask: {mask.tolist()} ✓")

    return True


def test_gene_vocabulary() -> bool:
    print("--- Gene vocabulary ---")
    from celljepa.data.streaming_dataset import load_gene_vocabulary

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "genes.txt"
        genes = ["TP53", "BRCA1", "EGFR", "MYC"]
        _create_gene_vocab(vocab_path, genes)

        names, to_idx = load_gene_vocabulary(vocab_path)
        assert names == genes
        assert to_idx["TP53"] == 1  # 1-indexed
        assert to_idx["MYC"] == 4
        print(f"  Loaded {len(names)} genes ✓")
        print(f"  1-indexed: TP53={to_idx['TP53']}, MYC={to_idx['MYC']} ✓")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("CellDataset", test_cell_dataset),
        ("CellDataset (no vocab)", test_cell_dataset_no_vocab),
        ("MultiDatasetMixer", test_multi_dataset_mixer),
        ("Dataset Catalog", test_dataset_catalog),
        ("DistributedTrainConfig", test_distributed_config),
        ("collate_variable_genes", test_collate_fn),
        ("Gene vocabulary", test_gene_vocabulary),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))
        print()

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")
        if result != "PASS":
            all_pass = False

    if all_pass:
        print(f"\nAll {len(results)} tests PASSED.")
    else:
        n_fail = sum(1 for _, r in results if r != "PASS")
        print(f"\n{n_fail}/{len(results)} tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
