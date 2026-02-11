#!/usr/bin/env python3
"""Smoke test for P3 world models, gene-level decoder, and metrics.

Tests all three world model architectures, the GeneLevelDecoder (both modes),
and the gene-level metric functions. No external data required.

Usage:
    python3 scripts/test_world_model_decoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from celljepa.models.world_model import (
    WorldModelV2Config,
    AttentionWorldModel,
    GraphConditionedWorldModel,
    DisentangledWorldModel,
    build_world_model,
)
from celljepa.models.decoder import DecoderConfig, GeneLevelDecoder
from celljepa.eval.metrics import (
    lfc_pearson_correlation,
    top_k_deg_recall,
    direction_accuracy,
)


BATCH = 4
EMBED_DIM = 128
N_GENES = 50
HIDDEN_DIM = 64


def _base_cfg(**overrides) -> WorldModelV2Config:
    defaults = dict(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    defaults.update(overrides)
    return WorldModelV2Config(**defaults)


def _make_synthetic_edge_index(n_nodes: int, n_edges: int = 30) -> torch.Tensor:
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def _check_grad(model: torch.nn.Module, output: torch.Tensor) -> bool:
    loss = output.sum()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# World Model tests
# ---------------------------------------------------------------------------

def test_attention_world_model() -> bool:
    print("--- AttentionWorldModel ---")
    cfg = _base_cfg(model_type="attention", attn_n_heads=4, attn_n_layers=2)
    model = AttentionWorldModel(cfg)

    cell_emb = torch.randn(BATCH, EMBED_DIM)
    pert_emb = torch.randn(BATCH, EMBED_DIM)
    gene_embs = torch.randn(BATCH, N_GENES, EMBED_DIM)

    out = model(cell_embedding=cell_emb, perturbation_embedding=pert_emb, gene_embeddings=gene_embs)
    assert out["cell_embedding"].shape == (BATCH, EMBED_DIM), f"cell shape: {out['cell_embedding'].shape}"
    assert out["gene_deltas"].shape == (BATCH, N_GENES, EMBED_DIM), f"delta shape: {out['gene_deltas'].shape}"
    print(f"  cell_embedding: {out['cell_embedding'].shape} ✓")
    print(f"  gene_deltas: {out['gene_deltas'].shape} ✓")

    # With covariates
    cov_emb = torch.randn(BATCH, EMBED_DIM)
    out2 = model(cell_embedding=cell_emb, perturbation_embedding=pert_emb,
                 gene_embeddings=gene_embs, covariate_embedding=cov_emb)
    assert out2["cell_embedding"].shape == (BATCH, EMBED_DIM)
    print(f"  With covariates: ✓")

    assert _check_grad(model, out["cell_embedding"]), "No gradients"
    print(f"  Gradient flow: ✓")
    return True


def test_graph_world_model() -> bool:
    print("--- GraphConditionedWorldModel ---")
    cfg = _base_cfg(model_type="graph", gnn_layers=2, gnn_heads=4)
    model = GraphConditionedWorldModel(cfg)

    cell_emb = torch.randn(BATCH, EMBED_DIM)
    pert_emb = torch.randn(BATCH, EMBED_DIM)
    gene_embs = torch.randn(BATCH, N_GENES, EMBED_DIM)
    edge_index = _make_synthetic_edge_index(N_GENES)
    pert_gene_idx = torch.randint(0, N_GENES, (BATCH,))

    out = model(
        cell_embedding=cell_emb, perturbation_embedding=pert_emb,
        gene_embeddings=gene_embs, edge_index=edge_index,
        perturbed_gene_idx=pert_gene_idx,
    )
    assert out["cell_embedding"].shape == (BATCH, EMBED_DIM), f"cell shape: {out['cell_embedding'].shape}"
    assert out["gene_deltas"].shape == (BATCH, N_GENES, EMBED_DIM), f"delta shape: {out['gene_deltas'].shape}"
    print(f"  cell_embedding: {out['cell_embedding'].shape} ✓")
    print(f"  gene_deltas: {out['gene_deltas'].shape} ✓")

    assert _check_grad(model, out["cell_embedding"]), "No gradients"
    print(f"  Gradient flow: ✓")
    return True


def test_disentangled_world_model() -> bool:
    print("--- DisentangledWorldModel ---")
    cfg = _base_cfg(model_type="disentangled", factor_dim=64)
    model = DisentangledWorldModel(cfg)

    cell_emb = torch.randn(BATCH, EMBED_DIM)
    pert_emb = torch.randn(BATCH, EMBED_DIM)

    # Without covariates
    out = model(cell_embedding=cell_emb, perturbation_embedding=pert_emb)
    assert out["cell_embedding"].shape == (BATCH, EMBED_DIM)
    assert "factors" in out
    assert out["factors"]["cell"].shape[1] == 64
    print(f"  cell_embedding: {out['cell_embedding'].shape} ✓")
    print(f"  factors: cell={out['factors']['cell'].shape}, pert={out['factors']['perturbation'].shape} ✓")

    # With covariates
    cov_emb = torch.randn(BATCH, EMBED_DIM)
    out2 = model(cell_embedding=cell_emb, perturbation_embedding=pert_emb, covariate_embedding=cov_emb)
    assert out2["cell_embedding"].shape == (BATCH, EMBED_DIM)
    print(f"  With covariates: ✓")

    assert _check_grad(model, out["cell_embedding"]), "No gradients"
    print(f"  Gradient flow: ✓")
    return True


def test_safety() -> bool:
    print("--- Safety-by-construction ---")
    # Test delta clamping
    cfg = _base_cfg(model_type="disentangled", max_delta_norm=0.1, alpha=0.5)
    model = DisentangledWorldModel(cfg)

    cell_emb = torch.randn(BATCH, EMBED_DIM)
    pert_emb = torch.randn(BATCH, EMBED_DIM) * 10  # large perturbation

    out = model(cell_embedding=cell_emb, perturbation_embedding=pert_emb)
    diff = (out["cell_embedding"] - cell_emb).norm(dim=-1)
    # With alpha=0.5 and max_delta_norm=0.1, max diff should be <= 0.05 + epsilon
    assert diff.max().item() < 0.1, f"Delta too large: {diff.max().item()}"
    print(f"  Delta clamping: max_diff={diff.max().item():.4f} ✓")
    return True


def test_factory() -> bool:
    print("--- build_world_model factory ---")
    for mtype in ["attention", "graph", "disentangled"]:
        cfg = _base_cfg(model_type=mtype)
        model = build_world_model(cfg)
        assert model.embed_dim == EMBED_DIM
        print(f"  {mtype}: instantiated ✓")

    try:
        cfg = _base_cfg(model_type="invalid")
        build_world_model(cfg)
        assert False, "Should have raised"
    except ValueError:
        print(f"  invalid: ValueError raised ✓")
    return True


# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------

def test_decoder_cell_level() -> bool:
    print("--- GeneLevelDecoder (cell-level) ---")
    cfg = DecoderConfig(embed_dim=EMBED_DIM, n_genes=N_GENES, hidden_dim=HIDDEN_DIM, n_layers=2)
    decoder = GeneLevelDecoder(cfg)

    z_pert = torch.randn(BATCH, EMBED_DIM)
    z_ctrl = torch.randn(BATCH, EMBED_DIM)

    lfc = decoder(z_pert=z_pert, z_ctrl=z_ctrl)
    assert lfc.shape == (BATCH, N_GENES), f"Shape mismatch: {lfc.shape}"
    print(f"  Output shape: {lfc.shape} ✓")

    assert _check_grad(decoder, lfc), "No gradients"
    print(f"  Gradient flow: ✓")
    return True


def test_decoder_gene_level() -> bool:
    print("--- GeneLevelDecoder (gene-level) ---")
    cfg = DecoderConfig(embed_dim=EMBED_DIM, n_genes=N_GENES, hidden_dim=HIDDEN_DIM)
    decoder = GeneLevelDecoder(cfg)

    gene_deltas = torch.randn(BATCH, N_GENES, EMBED_DIM)
    lfc = decoder.from_gene_deltas(gene_deltas)
    assert lfc.shape == (BATCH, N_GENES), f"Shape mismatch: {lfc.shape}"
    print(f"  Output shape: {lfc.shape} ✓")

    assert _check_grad(decoder, lfc), "No gradients"
    print(f"  Gradient flow: ✓")
    return True


def test_decoder_single_layer() -> bool:
    print("--- GeneLevelDecoder (1-layer) ---")
    cfg = DecoderConfig(embed_dim=EMBED_DIM, n_genes=N_GENES, n_layers=1)
    decoder = GeneLevelDecoder(cfg)

    z_pert = torch.randn(BATCH, EMBED_DIM)
    z_ctrl = torch.randn(BATCH, EMBED_DIM)
    lfc = decoder(z_pert=z_pert, z_ctrl=z_ctrl)
    assert lfc.shape == (BATCH, N_GENES)
    print(f"  Output shape: {lfc.shape} ✓")
    return True


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

def test_metrics() -> bool:
    print("--- Gene-level metrics ---")

    # Perfect prediction
    obs = np.array([1.5, -2.0, 0.5, -0.3, 3.0, -1.2, 0.8, -0.1, 2.0, -0.5])
    pred_perfect = obs.copy()

    r = lfc_pearson_correlation(pred_perfect, obs)
    assert abs(r - 1.0) < 1e-6, f"Perfect pearson should be 1.0, got {r}"
    print(f"  Perfect Pearson: {r:.4f} ✓")

    recall = top_k_deg_recall(pred_perfect, obs, k=5)
    assert recall == 1.0, f"Perfect recall should be 1.0, got {recall}"
    print(f"  Perfect top-5 recall: {recall:.2f} ✓")

    acc = direction_accuracy(pred_perfect, obs)
    assert acc == 1.0, f"Perfect direction should be 1.0, got {acc}"
    print(f"  Perfect direction accuracy: {acc:.2f} ✓")

    # Random prediction
    rng = np.random.default_rng(42)
    pred_random = rng.normal(0, 1, 100)
    obs_random = rng.normal(0, 1, 100)

    r_rand = lfc_pearson_correlation(pred_random, obs_random)
    assert abs(r_rand) < 0.5, f"Random pearson should be near 0, got {r_rand}"
    print(f"  Random Pearson: {r_rand:.4f} ✓")

    # Inverted prediction → direction accuracy should be 0
    pred_inv = -obs
    acc_inv = direction_accuracy(pred_inv, obs)
    assert acc_inv == 0.0, f"Inverted direction should be 0.0, got {acc_inv}"
    print(f"  Inverted direction accuracy: {acc_inv:.2f} ✓")

    # Edge: zero variance
    r_zero = lfc_pearson_correlation(np.zeros(10), obs)
    assert np.isnan(r_zero), "Zero variance should give NaN"
    print(f"  Zero variance → NaN: ✓")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("AttentionWorldModel", test_attention_world_model),
        ("GraphConditionedWorldModel", test_graph_world_model),
        ("DisentangledWorldModel", test_disentangled_world_model),
        ("Safety-by-construction", test_safety),
        ("Factory", test_factory),
        ("GeneLevelDecoder (cell)", test_decoder_cell_level),
        ("GeneLevelDecoder (gene)", test_decoder_gene_level),
        ("GeneLevelDecoder (1-layer)", test_decoder_single_layer),
        ("Gene-level metrics", test_metrics),
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
