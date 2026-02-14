#!/usr/bin/env python3
"""Evaluate pretrained CellJEPA checkpoints.

Loads a checkpoint, extracts cell embeddings from the teacher encoder,
and computes:
  1. kNN retrieval accuracy (perturbation labels)
  2. Silhouette score (perturbation clustering)
  3. Batch mixing (ASW on batch labels, lower = better mixing)
  4. UMAP visualizations colored by perturbation and batch

Usage:
    python3 scripts/eval_pretrain.py \
        --checkpoint runs/p4_pretrain_transformer_.../checkpoint_latest.pt \
        --dataset data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad \
        --gene-vocab configs/gene_vocab_human_v1.txt \
        --out results/eval_pretrain/transformer/ \
        --max-cells 20000 \
        --encoder transformer

    # Compare all 3 encoders:
    python3 scripts/eval_pretrain.py \
        --checkpoint runs/p4_pretrain_transformer_.../checkpoint_latest.pt \
                     runs/p4_pretrain_perceiver_.../checkpoint_latest.pt \
                     runs/p4_pretrain_gnn_.../checkpoint_latest.pt \
        --encoder transformer perceiver gnn \
        --dataset data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad \
        --gene-vocab configs/gene_vocab_human_v1.txt \
        --out results/eval_pretrain/comparison/ \
        --max-cells 20000
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
from torch.utils.data import DataLoader, Subset

# ── CellJEPA imports ────────────────────────────────────────────────
from celljepa.models.jepa import JEPAv2, JEPAv2Config
from celljepa.models.gene_tokenizer import GeneTokenizerConfig
from celljepa.data.streaming_dataset import CellDataset, CellDatasetConfig


# ── Helpers ─────────────────────────────────────────────────────────

def collate_variable_genes(batch):
    """Same collate as pretrain — pad to max gene count in batch."""
    exprs, gids = zip(*batch)
    max_len = max(e.shape[0] for e in exprs)
    batch_size = len(batch)

    expression = torch.zeros(batch_size, max_len, dtype=torch.float32)
    gene_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, (e, g) in enumerate(zip(exprs, gids)):
        n = e.shape[0]
        expression[i, :n] = e
        gene_ids[i, :n] = g

    return expression, gene_ids


def build_model_from_checkpoint(
    ckpt_path: str,
    encoder_type: str,
    gene_graph_path: str | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[JEPAv2, dict]:
    """Load a checkpoint and reconstruct the JEPAv2 model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]

    # Reconstruct the config
    jepa_cfg = JEPAv2Config(
        tokenizer=GeneTokenizerConfig(**cfg_dict["jepa"]["tokenizer"]),
        encoder_type=cfg_dict["jepa"]["encoder_type"],
        encoder_kwargs=cfg_dict["jepa"]["encoder_kwargs"],
        loss_type=cfg_dict["jepa"].get("loss_type", "smooth_l1"),
    )

    model = JEPAv2(jepa_cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    return model, ckpt


@torch.no_grad()
def extract_embeddings(
    model: JEPAv2,
    loader: DataLoader,
    device: torch.device,
    edge_index: torch.Tensor | None = None,
    max_batches: int = 0,
) -> np.ndarray:
    """Extract cell embeddings from the teacher encoder.

    Uses the teacher (EMA-averaged) encoder for more stable representations.
    """
    all_embeddings = []
    for i, (expression, gene_ids) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        expression = expression.to(device)
        gene_ids = gene_ids.to(device)

        # Tokenize
        tokens = model.tokenizer(expression, gene_ids)

        # Prepare edge_index for GNN if needed
        batch_edge = None
        if edge_index is not None:
            from pretrain_jepa_v2 import extract_subgraph
            batch_edge = extract_subgraph(edge_index, gene_ids[0])

        # Encode with teacher (EMA-averaged, more stable)
        out = model._encode(model.teacher, tokens, mask=None, edge_index=batch_edge)
        cell_embs = out["cell_embedding"].cpu().numpy()
        all_embeddings.append(cell_embs)

        if (i + 1) % 50 == 0:
            print(f"  Extracted {sum(e.shape[0] for e in all_embeddings):,} embeddings...")

    return np.concatenate(all_embeddings, axis=0)


def compute_metrics(
    embeddings: np.ndarray,
    perturbation_labels: np.ndarray,
    batch_labels: np.ndarray,
    is_control: np.ndarray,
    k: int = 10,
) -> dict:
    """Compute embedding quality metrics."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    results = {}
    n_cells = embeddings.shape[0]
    print(f"\n  Computing metrics on {n_cells:,} cells, {embeddings.shape[1]}d embeddings...")

    # 1. kNN perturbation retrieval (perturbed cells only)
    pert_mask = ~is_control
    pert_embs = embeddings[pert_mask]
    pert_labels = perturbation_labels[pert_mask]

    # Filter to perturbations with enough cells for kNN
    unique, counts = np.unique(pert_labels, return_counts=True)
    valid_perts = unique[counts >= k]
    valid_mask = np.isin(pert_labels, valid_perts)

    if valid_mask.sum() > 100:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        scores = cross_val_score(knn, pert_embs[valid_mask], pert_labels[valid_mask], cv=3)
        results["knn_perturbation_acc"] = float(np.mean(scores))
        results["knn_perturbation_std"] = float(np.std(scores))
        results["n_valid_perturbations"] = int(len(valid_perts))
        print(f"  kNN perturbation accuracy: {results['knn_perturbation_acc']:.4f} "
              f"(±{results['knn_perturbation_std']:.4f}), {len(valid_perts)} perturbations")
    else:
        results["knn_perturbation_acc"] = float("nan")
        print("  kNN: too few valid perturbation cells")

    # 2. Silhouette score on perturbation labels (subsample for speed)
    max_sil = min(10000, len(pert_embs))
    if max_sil > 500 and len(np.unique(pert_labels[:max_sil])) > 1:
        idx = np.random.RandomState(42).choice(len(pert_embs), max_sil, replace=False)
        sil = silhouette_score(pert_embs[idx], pert_labels[idx], metric="cosine", sample_size=None)
        results["silhouette_perturbation"] = float(sil)
        print(f"  Silhouette (perturbation): {sil:.4f}")
    else:
        results["silhouette_perturbation"] = float("nan")

    # 3. Batch mixing score (ASW on batch labels — lower = better mixing)
    max_batch_sil = min(10000, n_cells)
    if max_batch_sil > 500 and len(np.unique(batch_labels[:max_batch_sil])) > 1:
        idx = np.random.RandomState(42).choice(n_cells, max_batch_sil, replace=False)
        batch_sil = silhouette_score(embeddings[idx], batch_labels[idx], metric="cosine")
        results["batch_asw"] = float(batch_sil)
        print(f"  Batch ASW (lower=better mixing): {batch_sil:.4f}")
    else:
        results["batch_asw"] = float("nan")

    # 4. Control vs perturbed separation
    ctrl_embs = embeddings[is_control]
    pert_embs_all = embeddings[~is_control]
    if len(ctrl_embs) > 10 and len(pert_embs_all) > 10:
        ctrl_mean = ctrl_embs.mean(axis=0)
        pert_mean = pert_embs_all.mean(axis=0)
        cos_sep = 1.0 - np.dot(ctrl_mean, pert_mean) / (
            np.linalg.norm(ctrl_mean) * np.linalg.norm(pert_mean) + 1e-8
        )
        results["control_pert_cosine_sep"] = float(cos_sep)
        print(f"  Control-Perturbed cosine sep: {cos_sep:.4f}")

    # 5. Embedding stats
    results["mean_norm"] = float(np.linalg.norm(embeddings, axis=1).mean())
    results["std_norm"] = float(np.linalg.norm(embeddings, axis=1).std())
    var_per_dim = embeddings.var(axis=0)
    results["mean_dim_variance"] = float(var_per_dim.mean())
    collapsed_dims = (var_per_dim < 1e-6).sum()
    results["collapsed_dims"] = int(collapsed_dims)
    results["embed_dim"] = int(embeddings.shape[1])
    print(f"  Embedding norm: {results['mean_norm']:.2f}±{results['std_norm']:.2f}, "
          f"collapsed dims: {collapsed_dims}/{embeddings.shape[1]}")

    return results


def make_umap_plots(
    embeddings: np.ndarray,
    perturbation_labels: np.ndarray,
    batch_labels: np.ndarray,
    is_control: np.ndarray,
    out_dir: Path,
    encoder_name: str,
) -> list[Path]:
    """Generate UMAP visualizations."""
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  UMAP/matplotlib not available, skipping plots")
        return []

    print(f"\n  Computing UMAP for {encoder_name}...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    xy = reducer.fit_transform(embeddings)

    paths = []

    # 1. UMAP colored by control vs perturbed
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    colors = np.where(is_control, "#2ecc71", "#e74c3c")
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=1, alpha=0.3, rasterized=True)
    ax.set_title(f"{encoder_name} — Control (green) vs Perturbed (red)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    path = out_dir / f"umap_{encoder_name}_ctrl_vs_pert.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    print(f"  Saved: {path}")

    # 2. UMAP colored by batch
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    unique_batches = np.unique(batch_labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_batches))
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
    batch_colors = [cmap(batch_to_idx[b]) for b in batch_labels]
    ax.scatter(xy[:, 0], xy[:, 1], c=batch_colors, s=1, alpha=0.3, rasterized=True)
    ax.set_title(f"{encoder_name} — Colored by Batch ({len(unique_batches)} batches)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    path = out_dir / f"umap_{encoder_name}_by_batch.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    print(f"  Saved: {path}")

    # 3. UMAP colored by top-20 most frequent perturbations (for readability)
    pert_mask = ~is_control
    unique_perts, pert_counts = np.unique(perturbation_labels[pert_mask], return_counts=True)
    top20 = unique_perts[np.argsort(-pert_counts)[:20]]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=150)
    # Background: all cells in grey
    ax.scatter(xy[:, 0], xy[:, 1], c="#dddddd", s=0.5, alpha=0.2, rasterized=True)
    # Overlay top-20 perturbations
    cmap20 = plt.cm.get_cmap("tab20", 20)
    for i, pert in enumerate(top20):
        mask = perturbation_labels == pert
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[cmap20(i)], s=3, alpha=0.7,
                   label=pert, rasterized=True)
    ax.legend(fontsize=6, ncol=2, loc="best", markerscale=3)
    ax.set_title(f"{encoder_name} — Top 20 Perturbations")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    path = out_dir / f"umap_{encoder_name}_top20_perts.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    print(f"  Saved: {path}")

    return paths


# ── Main ────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_path: str,
    encoder_type: str,
    dataset_path: str,
    gene_vocab: str,
    out_dir: Path,
    max_cells: int = 20000,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    gene_graph: str | None = None,
) -> dict:
    """Full evaluation pipeline for a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {encoder_type}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # Load model
    t0 = time.time()
    model, ckpt = build_model_from_checkpoint(ckpt_path, encoder_type, device=device)
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded model (epoch {epoch}) in {time.time()-t0:.1f}s")

    # Load dataset
    ds_config = CellDatasetConfig(gene_vocab_path=gene_vocab, max_genes_per_cell=5000)
    ds = CellDataset(dataset_path, ds_config)
    print(f"  Dataset: {ds.n_cells:,} cells, {ds.mapped_gene_count:,} mapped genes")

    # Subsample if needed
    if max_cells > 0 and ds.n_cells > max_cells:
        indices = np.random.RandomState(42).choice(ds.n_cells, max_cells, replace=False)
        indices.sort()
        subset = Subset(ds, indices)
        print(f"  Subsampled to {max_cells:,} cells")
    else:
        subset = ds
        indices = np.arange(ds.n_cells)

    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_variable_genes, num_workers=2, pin_memory=False,
    )

    # Load edge_index for GNN
    edge_index = None
    if encoder_type == "gnn" and gene_graph:
        graph_data = torch.load(gene_graph, weights_only=False)
        edge_index = graph_data["edge_index"].to(device)
        print(f"  Gene graph: {edge_index.shape[1]:,} edges")

    # Extract embeddings
    t0 = time.time()
    embeddings = extract_embeddings(model, loader, device, edge_index=edge_index)
    print(f"  Extracted {embeddings.shape[0]:,} embeddings in {time.time()-t0:.1f}s")

    # Load metadata for the same cells
    import anndata as ad
    adata = ad.read_h5ad(dataset_path, backed="r")
    obs = adata.obs.iloc[indices].copy()
    adata.file.close()

    perturbation_labels = obs["gene"].values.astype(str)
    batch_labels = obs["batch"].values.astype(str)
    is_control = obs["is_control"].values.astype(bool)

    # Compute metrics
    metrics = compute_metrics(embeddings, perturbation_labels, batch_labels, is_control)
    metrics["encoder_type"] = encoder_type
    metrics["checkpoint"] = str(ckpt_path)
    metrics["epoch"] = epoch
    metrics["n_cells_evaluated"] = int(embeddings.shape[0])

    # Save metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{encoder_type}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\n  Metrics saved: {metrics_path}")

    # Generate UMAP plots
    make_umap_plots(embeddings, perturbation_labels, batch_labels, is_control,
                    out_dir, encoder_type)

    # Save embeddings for downstream use
    emb_path = out_dir / f"embeddings_{encoder_type}.npz"
    np.savez_compressed(emb_path, embeddings=embeddings, indices=indices)
    print(f"  Embeddings saved: {emb_path}")

    return metrics


def print_comparison_table(all_metrics: list[dict]) -> None:
    """Print a comparison table across encoders."""
    print(f"\n{'='*70}")
    print("ENCODER COMPARISON")
    print(f"{'='*70}")

    metric_keys = [
        ("knn_perturbation_acc", "kNN Pert. Acc", True),
        ("silhouette_perturbation", "Silhouette (pert)", True),
        ("batch_asw", "Batch ASW (↓better)", False),
        ("control_pert_cosine_sep", "Ctrl-Pert Sep", True),
        ("mean_dim_variance", "Mean Dim Var", True),
        ("collapsed_dims", "Collapsed Dims", False),
    ]

    # Header
    encoders = [m["encoder_type"] for m in all_metrics]
    header = f"{'Metric':<25s}" + "".join(f"{e:>15s}" for e in encoders)
    print(header)
    print("-" * len(header))

    for key, label, higher_better in metric_keys:
        values = [m.get(key, float("nan")) for m in all_metrics]
        row = f"{label:<25s}"
        best_idx = np.nanargmax(values) if higher_better else np.nanargmin(values)
        for i, v in enumerate(values):
            marker = " *" if i == best_idx and not np.isnan(v) else "  "
            if isinstance(v, int):
                row += f"{v:>13d}{marker}"
            else:
                row += f"{v:>13.4f}{marker}"
        print(row)

    print(f"\n* = best for that metric")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained CellJEPA checkpoints")
    parser.add_argument("--checkpoint", nargs="+", required=True, help="Checkpoint path(s)")
    parser.add_argument("--encoder", nargs="+", required=True, help="Encoder type(s)")
    parser.add_argument("--dataset", required=True, help="h5ad dataset path")
    parser.add_argument("--gene-vocab", required=True, help="Gene vocabulary file")
    parser.add_argument("--gene-graph", default=None, help="Gene graph .pt (for GNN)")
    parser.add_argument("--out", default="results/eval_pretrain/", help="Output directory")
    parser.add_argument("--max-cells", type=int, default=20000, help="Max cells to evaluate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda:N)")
    args = parser.parse_args()

    assert len(args.checkpoint) == len(args.encoder), \
        f"Must provide same number of checkpoints and encoders ({len(args.checkpoint)} vs {len(args.encoder)})"

    out_dir = Path(args.out)
    device = torch.device(args.device)

    all_metrics = []
    for ckpt_path, enc_type in zip(args.checkpoint, args.encoder):
        metrics = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            encoder_type=enc_type,
            dataset_path=args.dataset,
            gene_vocab=args.gene_vocab,
            out_dir=out_dir,
            max_cells=args.max_cells,
            batch_size=args.batch_size,
            device=device,
            gene_graph=args.gene_graph if enc_type == "gnn" else None,
        )
        all_metrics.append(metrics)

    # Comparison table
    if len(all_metrics) > 1:
        print_comparison_table(all_metrics)

    # Save combined results
    combined_path = out_dir / "comparison.json"
    combined_path.write_text(json.dumps(all_metrics, indent=2, default=str))
    print(f"\nCombined results: {combined_path}")
    print("Done!")


if __name__ == "__main__":
    main()
