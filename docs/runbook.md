# CellJEPA — Runbook (Operational Spec)

Date: 2026-01-13  
Status: canonical (replaces several older “how-to” docs)

This file is the **operational spec** for running CellJEPA: dataset schema, dataset suite, split protocols, metrics, baseline fairness, and HPC execution. It intentionally avoids being a second project plan.

## Doc map (what to read)

Canonical docs (keep these consistent):
- `docs/plan.md` — roadmap + gates (“what we’re building and why”)
- `docs/runbook.md` — operational spec (“how we run it”) *(this file)*
- `docs/DECISIONS.md` — user-confirmed turning-point decisions
- `docs/PROJECT_STATE.md` — current snapshot + run inventory

Historical docs live in `docs/archive/` and are **not** authoritative.

Conflict resolution:
1) `docs/DECISIONS.md`
2) `docs/plan.md`
3) `docs/runbook.md`
4) `docs/archive/`

---

## Quickstart (toy run)

This is the smallest end-to-end path that should work without real data:

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

Expected artifacts:
- `runs/m0_toy_baselines/metrics.json`
- `runs/m0_toy_baselines/report.md`

---

## Data contract (v1)

CellJEPA is a world-model framing where **actions induce transitions**. In v1, actions are perturbations, so the schema uses `perturbation_id` naming.

### Processed dataset artifact

Preferred v1 format: `anndata` stored as a single `.h5ad` per dataset and preprocessing version.

Required fields:
- `X`: numeric matrix `(n_cells, n_genes)` (v1 default is lib-size normalize → log1p)
- `var.index`: gene identifiers (see “Gene identifiers” below)
- `obs` (per-cell metadata) columns:
  - `perturbation_id` (string)
  - `is_control` (bool)
  - `context_id` (string)
  - `perturbation_tokens` (string; deterministic serialization, see below)
- `uns` provenance keys:
  - `dataset_id` (string)
  - `preprocess_name` (string)
  - `preprocess_version` (string or hash)
  - `created_at` (ISO timestamp)

Recommended (when available):
- `obs["cell_type"]`, `obs["batch"]`
- `obs["dose"]` (float), `obs["time_hours"]` (float)
- `uns["source"]` (URL/citation), `uns["feature_set"]` (gene harmonization tag)

### Gene identifiers (practical v1 policy)

Current v1 reality in this repo: most ingested/harmonized artifacts use **gene symbols** as `var.index` (this matches the scPerturb harmonized `.h5ad` convention and the checked-in harmonization lists under `configs/harmonization/`).

Hard requirements regardless of ID space:
- `var.index` must be **unique** (no duplicates)
- the ID space must be **explicit** (record it in `uns` and/or `var` columns)
- if you remap IDs, store the mapping provenance and version

Recommended if you stay on gene symbols:
- also store an `Ensembl` column when available (e.g., `var["ensembl_id"]`) to reduce ambiguity later

### Action / perturbation token serialization

We store a per-cell token string in `obs["perturbation_tokens"]` so downstream code doesn’t depend on dataset-specific categorical columns.

Conventions:
- Token list is serialized with `|` as a separator.
- Tokens should be prefixed when possible:
  - genes: `gene:STAT1`
  - drugs: `drug:dexamethasone`
- Control must be canonical: `control:CTRL`
- Combos are multiple tokens (order must be deterministic):
  - `gene:STAT1|gene:IRF9`

---

## Dataset suite (v1)

Primary ingestion targets (current repo state):
- scPerturb v1.4 harmonized `.h5ad` files (includes Sci-Plex2/3/4)
- Sci-Plex3 as the default drug benchmark for within-dataset context holdout
- NormanWeissman2019 (filtered) + Replogle 2022 merged (K562 + RPE1) as genetic perturbation benchmarks
- Perturb-CITE-seq (RNA+protein) is reserved for M5

Download manifests live under `configs/download/` (see `configs/download/README.md`).

---

## Split protocols (v1)

Principles:
- Splits operate on **groups** (contexts or perturbations), not random per-cell splits.
- Split generation must be deterministic given `(dataset_id, split_name, seed, fold)`.
- Split files are artifacts: reuse them across runs for comparability.

### Stage A main-table splits

**S1 — Unseen perturbation (perturbation holdout)**  
Goal: generalize to held-out perturbations.
- group key: `perturbation_id`
- contexts may overlap between train/test (isolates perturbation holdout)

**S2 — Unseen context (context holdout)**  
Goal: generalize to held-out contexts (donors/cell lines/etc).
- group key: `context_id`
- perturbations may overlap between train/test (isolates context holdout)

Defaults:
- `k_folds = 5` where feasible; reduce for small datasets but avoid `< 3` without marking results “pilot”
- training seeds per fold: `0, 1, 2`

### Split file format

The canonical JSON schema matches `scripts/make_splits.py`:

```json
{
  "dataset_id": "…",
  "split_name": "S1_unseen_perturbation",
  "seed": 0,
  "fold": 0,
  "train_groups": ["…"],
  "val_groups": ["…"],
  "test_groups": ["…"],
  "group_key": "perturbation_id"
}
```

### Leakage prevention checklist (“no peeking”)

Anything distribution- or label-dependent must be computed on **training folds only**:
- HVG selection
- scaling parameters
- batch correction transforms
- learned masks/probes using perturbation labels

Store fold-specific artifacts under `runs/<run_id>/artifacts/` (or a split-keyed preprocessing cache) and reference them in reports.

---

## Metrics (v1)

Unit of evaluation:
- compute metrics per **(context_id, perturbation_id)** condition pair
- aggregate across condition pairs (not across cells) to avoid cell-count weighting artifacts

Primary v1 set metric:
- **energy distance (E-distance)** between predicted vs observed embedding sets

Implementation notes:
- equalize set sizes by sampling `n` cells per set (cap `n` for compute)
- average over multiple resamples for stability (record `n` and number of resamples)

Also report:
- prototype metrics: cosine distance / MSE between predicted and observed prototypes
- retrieval metrics: kNN accuracy / MRR for retrieving the correct condition

Confidence intervals:
- bootstrap over condition pairs (sample pairs with replacement, recompute aggregate metric)

Reporting requirements (minimum):
- dataset stats (cells/genes/conditions)
- split definition and counts after filtering
- baseline table + tuning details
- main results + ablations
- compute summary

---

## Baselines and fairness protocol

Required baselines for every report:
- **no-change** (predict control state as post-perturbation state)
- **mean-shift** (per-perturbation shift, fit on training contexts only for S2)
- **ridge** mapping in a fixed representation space (PCA or the chosen embedding space), fit on training contexts only for S2

Fairness rules:
- select hyperparameters on **validation** folds only
- report the search space + best config
- keep the tuning budget comparable across methods

Suggested default tuning budget (Stage A):
- ridge `alpha` grid: `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`
- PCA dimension grid (if used): `[32, 64, 128]` (or keep a single fixed choice and justify)

Safety baseline (recommended when models can blow up under shift):
- shrinkage residual form with `α ∈ [0, 1]`, including `α=0` as an explicit fallback

---

## Cross-dataset evaluation (M4 safety rules)

Cross-dataset evaluation is misleading by default because it mixes **domain shift** with **action vocabulary shift**.

Hard requirements:
- always compute and log action overlap: `|A_train ∩ A_test|` and overlap fractions
- define the main cross-dataset table on **shared-action only**
- log `<UNK>`/unseen-action as diagnostics only (no pooling)
- enable **control-based embedding calibration (z-score on controls)** by default for cross-dataset (ablate off explicitly)

---

## Running jobs (entry points)

Common scripts:
- Ingestion: `scripts/ingest_scperturb.py`, `scripts/ingest_replogle_combo.py`
- Splits: `scripts/make_splits.py`
- Baselines: `scripts/eval_baselines.py`
- JEPA: `scripts/train_jepa.py`, `scripts/eval_embeddings.py`
- Transition predictor: `scripts/train_transition.py`
- World model: `scripts/train_world_model.py`
- Cross-dataset helpers: `scripts/m4_*`

Run artifact convention:
- every run writes under `runs/<run_id>/` with at least `metrics.json` and `report.md`

---

## HPC / Slurm (default execution mode)

Policy:
- prefer `sbatch` for anything beyond quick checks
- default walltime **> 24 hours** (recommend `--time=48:00:00` or higher)
- logs go under `logs/` (gitignored)
- run artifacts go under `runs/<run_id>/…`

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
# python3 scripts/train_jepa.py --dataset ... --split ... --out runs/<run_id>/
```

### Downloads (Slurm arrays)

Manifests live under `configs/download/` and are TSV with:

```text
<url>\t<relative_path>\t<optional_checksum>
```

Submit a download array job:

```bash
scripts/download/submit_downloads.sh configs/download/scperturb_v1_4_urls.tsv scperturb_v1_4 10
```

Useful env vars:
- `PARTITION` (default tries `io` then `cpu`)
- `TIME` (default `48:00:00`)
- `DEST_ROOT` (default `data/raw`)
- `CURL_INSECURE=1` (cluster-specific workaround; only if needed)

Monitor:

```bash
squeue -u $USER
```

