#!/usr/bin/env python3
"""Smoke test for P2 perturbation encoders.

Tests all five encoder classes: instantiation, forward pass shape,
and gradient flow. No external data or dependencies required.

Usage:
    python3 scripts/test_perturbation_encoders.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from celljepa.models.perturbation_encoders import (
    PerturbationEncoderConfig,
    GeneIdentityPerturbationEncoder,
    GeneGraphPerturbationEncoder,
    ChemicalFingerprintEncoder,
    CombinatorialPerturbationEncoder,
    DoseTimeEncoder,
    build_perturbation_encoder,
)
from celljepa.models.gene_tokenizer import GeneIdentityEmbedding


BATCH = 8
EMBED_DIM = 256
N_GENES = 200  # small for testing
GENE_EMBED_DIM = 128


def _base_cfg(**overrides) -> PerturbationEncoderConfig:
    defaults = dict(
        embed_dim=EMBED_DIM,
        n_genes=N_GENES,
        gene_embed_dim=GENE_EMBED_DIM,
    )
    defaults.update(overrides)
    return PerturbationEncoderConfig(**defaults)


def _make_synthetic_edge_index(n_nodes: int, n_edges: int = 50) -> torch.Tensor:
    """Create a small random edge_index for graph tests."""
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    # Make bidirectional
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    return torch.stack([src_all, dst_all], dim=0)


def _check_grad(encoder: torch.nn.Module, output: torch.Tensor, name: str) -> bool:
    """Check that gradients flow back through the encoder."""
    loss = output.sum()
    loss.backward()
    has_grad = False
    for p in encoder.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    return has_grad


def test_gene_identity() -> bool:
    print("--- GeneIdentityPerturbationEncoder ---")
    cfg = _base_cfg(encoder_type="gene_identity")
    encoder = GeneIdentityPerturbationEncoder(cfg)
    gene_idx = torch.randint(1, N_GENES, (BATCH,))
    out = encoder(gene_idx=gene_idx)
    assert out.shape == (BATCH, EMBED_DIM), f"Shape mismatch: {out.shape}"
    assert _check_grad(encoder, out, "GeneIdentity"), "No gradients"
    print(f"  Output shape: {out.shape} ✓")
    print(f"  Gradient flow: ✓")
    return True


def test_gene_identity_shared() -> bool:
    print("--- GeneIdentityPerturbationEncoder (shared weights) ---")
    shared = GeneIdentityEmbedding(n_genes=N_GENES, embed_dim=GENE_EMBED_DIM)
    cfg = _base_cfg(encoder_type="gene_identity")
    encoder = GeneIdentityPerturbationEncoder(cfg, shared_embedding=shared)
    gene_idx = torch.randint(1, N_GENES, (BATCH,))
    out = encoder(gene_idx=gene_idx)
    assert out.shape == (BATCH, EMBED_DIM), f"Shape mismatch: {out.shape}"
    # Check weight sharing
    assert encoder.gene_embedding is shared, "Weights not shared"
    print(f"  Output shape: {out.shape} ✓")
    print(f"  Weight sharing: ✓")
    return True


def test_gene_graph() -> bool:
    print("--- GeneGraphPerturbationEncoder ---")
    cfg = _base_cfg(encoder_type="gene_graph", gnn_layers=2, gnn_heads=4)
    encoder = GeneGraphPerturbationEncoder(cfg)
    gene_idx = torch.randint(1, N_GENES, (BATCH,))
    edge_index = _make_synthetic_edge_index(N_GENES)
    gene_ids = torch.arange(1, N_GENES + 1)
    out = encoder(gene_idx=gene_idx, edge_index=edge_index, gene_ids=gene_ids)
    assert out.shape == (BATCH, EMBED_DIM), f"Shape mismatch: {out.shape}"
    assert _check_grad(encoder, out, "GeneGraph"), "No gradients"
    print(f"  Output shape: {out.shape} ✓")
    print(f"  Gradient flow: ✓")
    return True


def test_chemical_fingerprint() -> bool:
    print("--- ChemicalFingerprintEncoder ---")
    cfg = _base_cfg(encoder_type="chemical_fingerprint", fingerprint_dim=2048)
    encoder = ChemicalFingerprintEncoder(cfg)
    fp = torch.randint(0, 2, (BATCH, 2048)).float()
    out = encoder(fingerprint=fp)
    assert out.shape == (BATCH, EMBED_DIM), f"Shape mismatch: {out.shape}"
    assert _check_grad(encoder, out, "ChemicalFingerprint"), "No gradients"
    print(f"  Output shape: {out.shape} ✓")
    print(f"  Gradient flow: ✓")
    return True


def test_combinatorial() -> bool:
    print("--- CombinatorialPerturbationEncoder ---")
    cfg = _base_cfg(encoder_type="combinatorial", combo_n_heads=4)
    encoder = CombinatorialPerturbationEncoder(cfg)

    for n_perts in [1, 2, 5]:
        pert_embs = torch.randn(BATCH, n_perts, EMBED_DIM)
        mask = torch.ones(BATCH, n_perts, dtype=torch.bool)
        out = encoder(pert_embeddings=pert_embs, pert_mask=mask)
        assert out.shape == (BATCH, EMBED_DIM), f"Shape mismatch for n_perts={n_perts}: {out.shape}"
        print(f"  n_perts={n_perts}: shape {out.shape} ✓")

    # Test with padding
    pert_embs = torch.randn(BATCH, 5, EMBED_DIM)
    mask = torch.ones(BATCH, 5, dtype=torch.bool)
    mask[:, 3:] = False  # last 2 are padding
    out = encoder(pert_embeddings=pert_embs, pert_mask=mask)
    assert out.shape == (BATCH, EMBED_DIM), f"Shape mismatch with padding: {out.shape}"
    assert _check_grad(encoder, out, "Combinatorial"), "No gradients"
    print(f"  With padding: shape {out.shape} ✓")
    print(f"  Gradient flow: ✓")
    return True


def test_dose_time() -> bool:
    print("--- DoseTimeEncoder ---")
    cfg = _base_cfg(encoder_type="dose_time", n_covariates=2)
    encoder = DoseTimeEncoder(cfg)
    # log(dose), log(time_hours)
    covariates = torch.randn(BATCH, 2)
    out = encoder(covariates=covariates)
    assert out.shape == (BATCH, EMBED_DIM), f"Shape mismatch: {out.shape}"
    assert _check_grad(encoder, out, "DoseTime"), "No gradients"
    print(f"  Output shape: {out.shape} ✓")
    print(f"  Gradient flow: ✓")
    return True


def test_factory() -> bool:
    print("--- build_perturbation_encoder factory ---")
    for etype in ["gene_identity", "gene_graph", "chemical_fingerprint", "combinatorial", "dose_time"]:
        cfg = _base_cfg(encoder_type=etype)
        encoder = build_perturbation_encoder(cfg)
        assert encoder.embed_dim == EMBED_DIM, f"embed_dim mismatch for {etype}"
        print(f"  {etype}: instantiated ✓")

    # Invalid type
    try:
        cfg = _base_cfg(encoder_type="invalid_type")
        build_perturbation_encoder(cfg)
        assert False, "Should have raised ValueError"
    except ValueError:
        print(f"  invalid_type: ValueError raised ✓")

    return True


def main() -> None:
    tests = [
        ("GeneIdentityPerturbationEncoder", test_gene_identity),
        ("GeneIdentityPerturbationEncoder (shared)", test_gene_identity_shared),
        ("GeneGraphPerturbationEncoder", test_gene_graph),
        ("ChemicalFingerprintEncoder", test_chemical_fingerprint),
        ("CombinatorialPerturbationEncoder", test_combinatorial),
        ("DoseTimeEncoder", test_dose_time),
        ("Factory", test_factory),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"  ERROR: {e}")
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
