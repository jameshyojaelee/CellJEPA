# CellJEPA — Data Contract (v1)

This document defines the **minimum required schema** for any dataset used by CellJEPA so that ingestion, splits, training, and evaluation remain reproducible and comparable across datasets.

Note on terminology:
- CellJEPA is framed as a cell-centric world model where “actions” induce state transitions.
- In v1, **perturbations** are the primary action/transition regime, so the schema uses `perturbation_id` naming.
- Future transition datasets can typically reuse this contract by mapping their action label into `perturbation_id` (and using `is_control` to denote the baseline condition).

## 1) Processed Dataset Artifact

Preferred v1 format: `anndata` stored as `.h5ad` (single file per dataset + preprocessing version).

Minimum required fields:

### 1.1 Expression
- `X`: float matrix shaped `(n_cells, n_genes)`
  - v1 default: library-size normalize → log1p
  - do **not** store raw counts in `X` unless explicitly versioned; if present, store raw counts in a separate layer.

### 1.2 Genes (`var`)
- `var.index`: stable gene identifier (preferred: Ensembl gene ID)
- `var["gene_symbol"]`: optional, if available

Gene identity must be stable across datasets. If you use dataset-specific gene IDs, you must provide a mapping step and document it.

### 1.3 Cell metadata (`obs`)
Required columns:
- `perturbation_id` (string): canonical perturbation label (single ID for a condition; combos should be encoded deterministically)
- `is_control` (bool): true for control cells
- `context_id` (string): donor / cell line / other context grouping key

Recommended columns (when available):
- `cell_type` (string)
- `batch` (string)
- `dose` (float)
- `time_hours` (float)

### 1.4 Perturbation tokens (for portable encoding)
Store a per-cell representation of perturbations:
- `obs["perturbation_tokens"]`: a *deterministic* serialization of a token list.
  - Example encoding: `gene:STAT1|gene:IRF9` for a 2-gene combo; a single token for non-combos.
  - If control: use a canonical token like `control:CTRL`.

Implementation note: even if the model consumes structured lists, serializing deterministically makes caching and comparisons easier.

### 1.5 Provenance (`uns`)
Required keys:
- `uns["dataset_id"]`: string
- `uns["preprocess_name"]`: string (e.g., `"libnorm_log1p_v1"`)
- `uns["preprocess_version"]`: string or hash
- `uns["created_at"]`: ISO timestamp

Recommended keys:
- `uns["source"]`: URL or citation
- `uns["feature_set"]`: `"intersection"` / `"foundation_set_v1"` / etc

## 2) Split-Safe Rule (“No Peeking”)

Any statistic that depends on the dataset distribution (and especially on labels) must be computed on **training folds only**.

Examples:
- HVG selection
- scaling parameters
- batch correction transforms
- “adversarial masks” learned using perturbation labels

If such steps exist, store fold-specific artifacts (e.g., in `runs/<run_id>/artifacts/` or a dedicated preprocessing cache keyed by split ID).

## 3) Validation Checks (minimum)

For every processed dataset, validate:
- required columns exist and have no invalid types,
- each `(context_id, perturbation_id)` has enough cells for evaluation (or is flagged),
- controls exist for each context (or document exceptions),
- `var.index` has no duplicates.
