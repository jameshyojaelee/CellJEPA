#!/usr/bin/env python3
"""Smoke test for P6 multi-modal extension.

Tests ATAC/protein tokenizers, all 3 fusion encoders, and cross-modal JEPA
using synthetic data. No external dependencies required.

Usage:
    python3 scripts/test_multimodal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_atac_tokenizer() -> bool:
    print("--- ATACPeakTokenizer ---")
    from celljepa.models.tokenizer_atac import ATACPeakTokenizer, ATACPeakTokenizerConfig

    cfg = ATACPeakTokenizerConfig(n_peaks=100, token_dim=64, peak_embed_dim=32, n_fourier_features=32)
    tokenizer = ATACPeakTokenizer(cfg)

    batch, n_peaks = 4, 100
    accessibility = torch.rand(batch, n_peaks)
    peak_ids = torch.arange(1, n_peaks + 1)

    tokens = tokenizer(accessibility, peak_ids)
    assert tokens.shape == (batch, n_peaks, 64), f"Bad shape: {tokens.shape}"
    print(f"  Output shape: {tokens.shape} ✓")
    assert tokenizer.token_dim == 64
    print(f"  token_dim: {tokenizer.token_dim} ✓")

    # Check gradient flow
    loss = tokens.sum()
    loss.backward()
    grad_count = sum(1 for p in tokenizer.parameters() if p.grad is not None)
    assert grad_count > 0
    print(f"  Gradient flow: {grad_count} params with grad ✓")

    return True


def test_protein_tokenizer() -> bool:
    print("--- ProteinTokenizer ---")
    from celljepa.models.tokenizer_protein import ProteinTokenizer, ProteinTokenizerConfig

    cfg = ProteinTokenizerConfig(n_proteins=50, token_dim=64, protein_embed_dim=32, n_fourier_features=32)
    tokenizer = ProteinTokenizer(cfg)

    batch, n_proteins = 4, 50
    abundance = torch.rand(batch, n_proteins) * 100  # ADT counts
    protein_ids = torch.arange(1, n_proteins + 1)

    tokens = tokenizer(abundance, protein_ids)
    assert tokens.shape == (batch, n_proteins, 64), f"Bad shape: {tokens.shape}"
    print(f"  Output shape: {tokens.shape} ✓")

    # Check log normalization
    assert tokenizer.cfg.log_normalize is True
    print(f"  Log normalization: enabled ✓")

    return True


def test_early_fusion() -> bool:
    print("--- EarlyFusionEncoder ---")
    from celljepa.models.encoder_multimodal import MultiModalEncoderConfig, build_multimodal_encoder

    cfg = MultiModalEncoderConfig(
        token_dim=64, n_layers=2, n_heads=4, ff_dim=128,
        fusion="early", modalities=["rna", "atac"],
    )
    encoder = build_multimodal_encoder(cfg)

    modality_tokens = {
        "rna": torch.randn(2, 20, 64),
        "atac": torch.randn(2, 30, 64),
    }
    out = encoder(modality_tokens)

    assert "cell_embedding" in out
    assert out["cell_embedding"].shape == (2, 64), f"Bad cell_emb: {out['cell_embedding'].shape}"
    assert "rna_embeddings" in out
    assert out["rna_embeddings"].shape == (2, 20, 64)
    assert "atac_embeddings" in out
    assert out["atac_embeddings"].shape == (2, 30, 64)
    print(f"  cell_embedding: {out['cell_embedding'].shape} ✓")
    print(f"  rna_embeddings: {out['rna_embeddings'].shape} ✓")
    print(f"  atac_embeddings: {out['atac_embeddings'].shape} ✓")

    return True


def test_cross_modal_encoder() -> bool:
    print("--- CrossModalEncoder ---")
    from celljepa.models.encoder_multimodal import MultiModalEncoderConfig, build_multimodal_encoder

    cfg = MultiModalEncoderConfig(
        token_dim=64, n_layers=2, n_heads=4, ff_dim=128,
        fusion="cross_modal", modalities=["rna", "atac"],
        n_cross_attn_layers=1,
    )
    encoder = build_multimodal_encoder(cfg)

    modality_tokens = {
        "rna": torch.randn(2, 20, 64),
        "atac": torch.randn(2, 30, 64),
    }
    out = encoder(modality_tokens)

    assert out["cell_embedding"].shape == (2, 64)
    assert out["rna_embeddings"].shape == (2, 20, 64)
    assert out["atac_embeddings"].shape == (2, 30, 64)
    print(f"  cell_embedding: {out['cell_embedding'].shape} ✓")
    print(f"  Cross-attention applied ✓")

    # Gradient flow through cross-attention
    loss = out["cell_embedding"].sum()
    loss.backward()
    print(f"  Backward pass: OK ✓")

    return True


def test_late_fusion() -> bool:
    print("--- LateFusionEncoder ---")
    from celljepa.models.encoder_multimodal import MultiModalEncoderConfig, build_multimodal_encoder

    for method in ["attention", "mlp"]:
        cfg = MultiModalEncoderConfig(
            token_dim=64, n_layers=2, n_heads=4, ff_dim=128,
            fusion="late", modalities=["rna", "atac"],
            late_fusion_method=method,
        )
        encoder = build_multimodal_encoder(cfg)

        modality_tokens = {
            "rna": torch.randn(2, 20, 64),
            "atac": torch.randn(2, 30, 64),
        }
        out = encoder(modality_tokens)

        assert out["cell_embedding"].shape == (2, 64)
        print(f"  {method} fusion: cell_embedding {out['cell_embedding'].shape} ✓")

    return True


def test_cross_modal_jepa() -> bool:
    print("--- MultiModalJEPA ---")
    from celljepa.models.jepa_multimodal import MultiModalJEPAConfig, MultiModalJEPA

    cfg = MultiModalJEPAConfig(
        modalities=["rna", "atac"],
        fusion="cross_modal",
        token_dim=64,
        n_layers=2,
        n_heads=4,
        ff_dim=128,
        predictor_hidden=64,
        predictor_layers=1,
        mask_ratio=0.3,
        vicreg_weight=0.01,
    )
    model = MultiModalJEPA(cfg)

    modality_tokens = {
        "rna": torch.randn(4, 20, 64),
        "atac": torch.randn(4, 30, 64),
    }

    out = model(modality_tokens)

    assert "loss" in out
    assert "losses" in out
    assert "student_output" in out
    assert "teacher_output" in out

    loss = out["loss"]
    assert loss.requires_grad
    print(f"  Total loss: {loss.item():.4f} ✓")
    print(f"  Sub-losses: {list(out['losses'].keys())} ✓")

    # Backward
    loss.backward()
    student_grads = sum(1 for p in model.student.parameters() if p.grad is not None)
    teacher_grads = sum(1 for p in model.teacher.parameters() if p.grad is not None)
    assert student_grads > 0, "Student should have gradients"
    assert teacher_grads == 0, "Teacher should NOT have gradients"
    print(f"  Student gradients: {student_grads} params ✓")
    print(f"  Teacher gradients: {teacher_grads} (frozen) ✓")

    # EMA update
    model.update_teacher()
    print(f"  EMA update: OK ✓")

    return True


def test_three_modality() -> bool:
    print("--- Three-modality (RNA + ATAC + protein) ---")
    from celljepa.models.encoder_multimodal import MultiModalEncoderConfig, build_multimodal_encoder

    cfg = MultiModalEncoderConfig(
        token_dim=64, n_layers=2, n_heads=4, ff_dim=128,
        fusion="early", modalities=["rna", "atac", "protein"],
    )
    encoder = build_multimodal_encoder(cfg)

    modality_tokens = {
        "rna": torch.randn(2, 20, 64),
        "atac": torch.randn(2, 30, 64),
        "protein": torch.randn(2, 10, 64),
    }
    out = encoder(modality_tokens)

    assert out["cell_embedding"].shape == (2, 64)
    assert "rna_embeddings" in out
    assert "atac_embeddings" in out
    assert "protein_embeddings" in out
    print(f"  3-modality early fusion: {out['cell_embedding'].shape} ✓")
    print(f"  All modality embeddings present ✓")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("ATACPeakTokenizer", test_atac_tokenizer),
        ("ProteinTokenizer", test_protein_tokenizer),
        ("EarlyFusionEncoder", test_early_fusion),
        ("CrossModalEncoder", test_cross_modal_encoder),
        ("LateFusionEncoder", test_late_fusion),
        ("MultiModalJEPA", test_cross_modal_jepa),
        ("Three-modality", test_three_modality),
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
