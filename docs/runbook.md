# CellJEPA — Runbook (Operational Spec)

Date: 2026-02-11  
Status: v2 (updated for gene-token architecture and gene-level metrics)

This file is the **operational spec** for running CellJEPA: dataset schema, gene tokenization, split protocols, metrics, baseline fairness, and HPC execution.

## Doc map (what to read)

Canonical docs (keep these consistent):
- `docs/plan.md` — roadmap + gates ("what we're building and why")
- `docs/runbook.md` — operational spec ("how we run it") *(this file)*
- `docs/DECISIONS.md` — user-confirmed turning-point decisions
- `docs/PROJECT_STATE.md` — current snapshot + run inventory

Conflict resolution:
1) `docs/DECISIONS.md`
2) `docs/plan.md`
3) `docs/runbook.md`
4) `docs/archive/`

---

## Quickstart (toy run)

Smallest end-to-end path (works without real data):

```bash
python3 -m compileall src scripts

python3 scripts/make_toy_dataset.py --out data/processed/toy/toy_v1.h5ad

python3 scripts/make_splits.py \
  --dataset-id data/processed/toy/toy_v1.h5ad \
  --split-name S1_unseen_perturbation \
  --seed 0 --fold 0 \
  --out runs/m0_splits/s1.json

python3 scripts/make_splits.py \
  --dataset-id data/processed/toy/toy_v1.h5ad \
  --split-name S2_unseen_context \
  --seed 0 --fold 0 \
  --out runs/m0_splits/s2.json

python3 scripts/eval_baselines.py \
  --dataset data/processed/toy/toy_v1.h5ad \
  --split runs/m0_splits/s2.json \
  --out runs/m0_toy_baselines/
```

Expected: `runs/m0_toy_baselines/metrics.json` + `report.md`.

---

## Data contract (v2)

### Processed dataset artifact

Format: `anndata` stored as `.h5ad`.

Required fields:
- `X`: numeric matrix `(n_cells, n_genes)` (lib-size normalize → log1p)
- `var.index`: gene identifiers (unique)
- `obs` columns:
  - `perturbation_id` (string)
  - `is_control` (bool)
  - `context_id` (string)
  - `perturbation_tokens` (string; deterministic serialization)
- `uns` provenance keys:
  - `dataset_id`, `preprocess_name`, `preprocess_version`, `created_at`

Recommended (when available):
- `obs["cell_type"]`, `obs["batch"]`
- `obs["dose"]` (float), `obs["time_hours"]` (float)
- `uns["source"]` (URL/citation), `uns["feature_set"]`

### Gene identifiers
- `var.index` must be **unique**
- v2 default: **gene symbols**
- Store `var["ensembl_id"]` when available

### Gene tokenization (v2, new)

All encoder backends consume tokenized gene input:
- `GeneTokenizer` maps each expressed gene → `(gene_id_embedding, expression_fourier_features)`
- Gene identity embedding: learned d-dimensional vector per gene symbol (~20K genes)
- Expression encoding: Fourier features for continuous values (no binning)
- Variable gene count per cell supported (sparse input)

### Gene interaction graph (v2, new)

- `configs/graphs/ppi_go_graph_v1.pt`: STRING-db PPI + Gene Ontology co-annotation
- Build via: `python3 scripts/build_gene_graph.py --out configs/graphs/ppi_go_graph_v1.pt`
- Used by: GNN encoder, gene perturbation encoder

### Regulon database (v2, new)

- `configs/regulons/dorothea_v1.json`: DoRothEA TF → target gene mappings
- Build via: `python3 scripts/build_regulon_masks.py --out configs/regulons/dorothea_v1.json`
- Used by: regulon-aware masking strategy

### Action / perturbation token serialization

Same as v1:
- `|`-separated tokens, prefixed: `gene:STAT1`, `drug:dexamethasone`, `control:CTRL`
- Combos: `gene:STAT1|gene:IRF9` (deterministic order)

---

## Dataset suite (v2)

### Pretraining datasets (P4, no perturbation labels needed)
- CellxGene Census (human, ~50M cells)
- Tahoe-100M (if accessible)
- Tabula Sapiens (~500K cells, diverse primary human tissues)

### Perturbation datasets (for fine-tuning + evaluation)
See `docs/plan.md` §3 for the full target list. Current:
- scPerturb v1.4 (Sci-Plex2/3/4)
- Replogle 2022 (K562 + RPE1)
- NormanWeissman2019 (K562)
- NadigOConner2024 (HepG2, Jurkat) — in progress

Planned additions:
- Dixit 2016 (K562, bone marrow DCs)
- CROP-seq Datlinger 2017 (T cells)
- Perturb-CITE-seq Frangieh 2021 (T cells, multi-modal)
- McFarland 2020 (diverse cancer, dose-response)

---

## Split protocols (v2)

Identical to v1 — see `docs/plan.md` §4 for details.

**S1 — Unseen perturbation:** group key `perturbation_id`  
**S2 — Unseen context:** group key `context_id`

Defaults: `k_folds=5`, training seeds `0, 1, 2`.

### Leakage prevention ("no peeking")
- HVG selection, scaling, batch correction → train fold only
- Store fold-specific artifacts under `runs/<run_id>/artifacts/`

---

## Metrics (v2)

Unit of evaluation: per **(context_id, perturbation_id)** condition pair, aggregated across pairs.

### Primary metrics (gene-level decoded, new in v2)
- **LFC Pearson correlation**: predicted vs observed log-fold changes across genes
- **Top-20 DEG recall**: true top-20 DEGs found in predicted top-20
- **Direction accuracy**: binary up/down classification per gene
- **PerturBench rank metric**: perturbation ranking quality (0 = perfect)

### Secondary metrics (embedding-level, retained from v1)
- **Energy distance (E-distance)**: primary set metric
- **Prototype metrics**: cosine distance / MSE between prototypes
- **Retrieval**: kNN accuracy / MRR
- **Calibrated E-distance**: Oct 2025 variant

### Confidence intervals
- Bootstrap over condition pairs (sample pairs with replacement, recompute aggregate)

### Reporting requirements (minimum per run)
- `runs/<run_id>/metrics.json` (machine-readable; includes CIs)
- `runs/<run_id>/config.json` (CLI args / config snapshot)
- `runs/<run_id>/report.md` (human-readable)

---

## Baselines and fairness protocol

### Required baselines (every report)
- **no-change** (predict control as post-perturbation)
- **mean-shift** (per-perturbation shift, train contexts only for S2)
- **ridge** (in PCA or embedding space, train contexts only for S2)

### SOTA baselines (P5 head-to-head)
- **GEARS** (GNN + Gene Ontology)
- **scGPT** (Transformer + pretraining)
- **CPA** (disentangled VAE)
- Run on identical splits with comparable tuning budgets

### Fairness rules
- Hyperparameters selected on **validation** folds only
- Report search space + best config
- Comparable tuning budget across all methods

---

## Cross-dataset evaluation

- Always compute and log action overlap
- Main table: **shared-action only**
- Control-based embedding calibration (z-score) **enabled by default**
- Log `<UNK>`/unseen-action as diagnostics only

---

## Running jobs (v2 entry points)

Common scripts:
- Ingestion: `scripts/ingest_scperturb.py`, `scripts/ingest_replogle_combo.py`
- Splits: `scripts/make_splits.py`
- Baselines: `scripts/eval_baselines.py`
- JEPA: `scripts/train_jepa.py` (supports `--encoder {transformer,gnn,perceiver}`)
- World model: `scripts/train_world_model.py`
- Large-scale pretraining: `scripts/pretrain_large.py` (P4)
- Benchmark suite: `scripts/run_benchmark_suite.py` (P5)
- Graph artifacts: `scripts/build_gene_graph.py`
- Regulon masks: `scripts/build_regulon_masks.py`

Run artifact convention: `runs/<run_id>/` with at least `metrics.json`.

---

## HPC / Slurm

Policy:
- prefer `sbatch` for anything beyond quick checks
- default walltime > 24 hours (recommend `--time=48:00:00` or higher)
- multi-GPU support for large-scale pretraining (DDP/FSDP via `scripts/pretrain_large.py`)
- logs under `logs/` (gitignored), artifacts under `runs/<run_id>/`

Minimal Slurm template:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=celljepa
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Optional GPU:
#SBATCH --gres=gpu:1
#SBATCH --partition=<YOUR_GPU_PARTITION>

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /gpfs/commons/home/jameslee/CellJEPA
python3 -m compileall src scripts
# python3 scripts/train_jepa.py --encoder transformer --dataset ... --split ... --out runs/<run_id>/
```

### Downloads (Slurm arrays)

Manifests under `configs/download/` (TSV: `<url>\t<relative_path>\t<optional_checksum>`).

```bash
scripts/download/submit_downloads.sh configs/download/scperturb_v1_4_urls.tsv scperturb_v1_4 10
```
