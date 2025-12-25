# CellJEPA — Revised Project Plan (Execution-Ready)

Date: 2025-12-25  
Status: v2 (after self-critique)  
Canonical goal: Build a **cell-centric world model** for single-cell omics where JEPA learns a general **latent cell state representation**; evaluate primarily on **perturbation-driven state transitions** (perturbation prediction) in v1.

## 0) One-Page Summary

### Core thesis (testable)
Learning a JEPA-style representation of cell state and predicting **post-perturbation state in embedding space** yields **better OOD generalization** and **more stable/usable** perturbation prediction than objectives centered on reconstructing noisy measurements.

### Project framing (what “CellJEPA” means)
- a cell-centric “world model”: the cell’s latent state is the primary object
- JEPA is treated as a general representation learner for state (not a task-specific trick)
- perturbations are one instantiation of **state transitions** (action → next state)
- perturbation prediction is the primary evaluation regime in v1, not the only conceivable one

### Product deliverable (what exists at the end)
1. A reproducible **benchmark harness** that:
   - downloads/prepares a curated subset of perturbation datasets,
   - generates split files (OOD protocols),
   - trains simple + strong baselines,
   - trains CellJEPA models,
   - produces a fixed report with tables/plots and machine-readable metrics.
2. A **CellJEPA model** package with:
   - JEPA encoder pretraining,
   - perturbation-transition predictor(s) operating in embedding space,
   - ablations to isolate what matters (masking, teacher EMA, regularization).

### Non-goals (v1)
- Not a foundation model for all scRNA-seq.
- Not “best possible” count reconstruction.
- Not diffusion/LLM hybrids unless gated in as “stretch” after Stage A results land.
- Not morphology/spatial integration in v1.
- Not claiming perturbations are the only state-transition regime; they are the v1 evaluation focus.

## 0.1 Default Decisions (to unblock implementation)

These are *defaults*, not permanent commitments. We start here to avoid paralysis and to ensure runs are comparable.

- **Stage A main-table splits:** `S1_unseen_perturbation` and `S2_unseen_context` (defined in §4).
- **v1 preprocessing:** library-size normalize → log1p; no batch correction.
- **JEPA backbone (Stage A):** choose the simplest stable implementation first, then ablate alternatives later.

## 1) Definitions (so experiments are unambiguous)

- **Cell state embedding**: `z = fθ(x)` where `x` is a cell’s expression vector (after a fixed preprocessing contract).
- **Action / perturbation condition**: metadata `a` describing an intervention (gene KO, drug, dose, time). In v1, actions are perturbations.
- **State transition**: mapping from baseline/control state distribution to post-action state distribution within a context.
- **Baseline context**: the *control* distribution for a context (donor/cell line/cell type/batch), used as the “pre-perturbation” reference.
- **Prediction target**: the **post-perturbation embedding distribution** for condition `a` within a context.

We explicitly support two prediction granularities:
- **Prototype-level** (debug-first): predict condition mean/robust-mean embedding.
- **Set-level** (core): predict a set/distribution of embeddings and compare to the empirical perturbed set via set metrics.

## 2) Milestones and Acceptance Criteria (Gated Execution)

### M0 — Repo + Contracts (must be fast)
Deliverables:
- `docs/plan.md` (this), `docs/data_contract.md`, `docs/splits.md`, `docs/metrics.md`.
- `docs/datasets.md` with the initial dataset shortlist + rubric scores.
- A minimal repo layout (`src/`, `scripts/`, `configs/`, `data/`, `runs/`) with conventions.
- A split generator that produces deterministic split files from raw metadata.

Acceptance:
- Split generator runs end-to-end on a tiny toy dataset.
- All outputs are deterministic given a seed.

### M1 — Data ingestion + baseline harness (before any JEPA)
Deliverables:
- Implement dataset registry + ingestion for **one** chosen dataset first.
- Produce a processed artifact (e.g., `h5ad`) that conforms to the data contract.
- Implement **simple baselines** + evaluation:
  - no-change baseline,
  - mean-shift per perturbation (and per cell type if available),
  - linear mapping in PCA/latent space (ridge regression).

Acceptance:
- “Golden run” produces a report with dataset stats + baseline table.
- Baselines run on fixed OOD splits with confidence intervals.

### M2 — JEPA encoder pretraining (cell-level)
Deliverables:
- Implement cell-level JEPA pretraining:
  - teacher EMA encoder,
  - predictor head,
  - anti-collapse regularization + collapse diagnostics.
- Implement at least two masking strategies:
  - random gene masking,
  - module/pathway masking (if module mapping is feasible early; otherwise schedule for M3).

Acceptance:
- Training is stable (no collapse) across ≥3 seeds on a small dataset slice.
- Embeddings show non-trivial structure (e.g., kNN cell-type retrieval above chance if labels exist).

### M3 — Perturbation transition model (prototype → set)
Deliverables:
- Prototype transition predictor `h(μ_control, a) → μ_perturbed`.
- Set-level predictor and set-level loss/evaluation:
  - deterministic per-cell mapping with a set-level distance objective,
  - report set metrics (e.g., energy distance / E-distance, MMD; pick one primary).

Acceptance:
- On at least one OOD protocol, CellJEPA beats **at least one** strong simple baseline (e.g., ridge) on the primary metric with CIs.
- Full ablation matrix runs (teacher on/off; mask type; reg on/off).

### M4 — Multi-dataset robustness (core claim)
Deliverables:
- Extend ingestion to 2–4 datasets (curated list).
- Add cross-dataset holdout evaluation.
- Consolidated reporting across datasets.

Acceptance:
- Results table spans multiple datasets and demonstrates where CellJEPA helps and where it does not.

### M5 (Stretch) — Multi-modal RNA+protein (only after M3 success)
Deliverables:
- Ingest Perturb-CITE-seq (or equivalent), produce aligned multi-view representation.
- Evaluate whether multi-view JEPA improves robustness and reduces ambiguity.

Gate:
- Only start after M3 acceptance is met.

## 3) Data Plan (Concrete, Split-Safe)

### 3.1 Dataset selection rubric (used to pick the initial 2–4)
Score each candidate on:
- perturbation type diversity (genetic vs chemical),
- availability of controls matched by context,
- metadata completeness (perturbation ID, dose, time, donor/cell line),
- size (enough cells per condition for set metrics),
- minimal licensing / access friction.

### 3.2 Data contract (implementation requirement)
Every processed dataset must provide:
- `X`: numeric expression matrix (fixed preprocessing),
- `var`: gene identifiers and mapping,
- `obs`: per-cell metadata including:
  - `perturbation_id`, `is_control`,
  - `context_id` (donor/cell line),
  - optional: `cell_type`, `batch`.

Also store:
- `dataset_id`, preprocessing version hash, and split-safe statistics.

### 3.3 Preprocessing (start conservative; minimize degrees of freedom)
Initial v1 preprocessing target:
- library-size normalize → log1p (or another single fixed transform),
- fixed gene identifier standardization,
- no batch correction in v1 unless strictly split-safe and justified.

Rules:
- any statistics used by preprocessing that depend on the data distribution must be computed on the training fold only and saved per split.

### 3.4 Perturbation metadata schema (portable across datasets)

Define a canonical representation for perturbations:
- `perturbation_tokens`: list of string tokens (e.g., `["gene:STAT1"]`, `["drug:dexamethasone"]`, combos as multiple tokens)
- `dose`: numeric (optional; NaN if unknown)
- `time_hours`: numeric (optional; NaN if unknown)
- `is_control`: boolean

Default encoding strategy:
- token embedding lookup summed/pooled across tokens,
- numeric features (dose/time) passed through a small MLP and concatenated.

## 4) Split Protocols (OOD-first, enforced as code)

We define splits at the **condition level** (and optionally at the context level), then sample cells within condition/context groups.

### 4.1 Stage A main-table splits (defaults)

We standardize around two split families. Each split produces deterministic `train/val/test` condition lists and per-cell indices.

**S1 — Unseen perturbation (condition OOD)**  
Goal: generalize to perturbations not seen during training.
- Split key: `perturbation_id`
- Grouping rule: all cells with the same `perturbation_id` are assigned to the same fold.
- Context handling: contexts are allowed to appear in both train and test, but perturbations are disjoint.

**S2 — Unseen context (context OOD)**  
Goal: generalize to new donors/cell lines (or other context definition).
- Split key: `context_id`
- Grouping rule: all cells with the same `context_id` are assigned to the same fold.
- Perturbation handling: perturbations may overlap between train and test, but contexts are disjoint.

Defaults:
- folds: 5 (or fewer if the dataset is too small; never <3 without calling it “pilot only”)
- training seeds per fold: 3 (e.g., 0/1/2)

### 4.2 Cross-dataset holdout (Stage M4)
Hold out entire dataset(s) after harmonizing to a shared gene set (e.g., intersection or a documented “foundation set”).

Hard rules:
- No preprocessing statistics may use test-fold cells.
- Any learned modules/probes (e.g., adversarial masks) must be fit on training folds only.
- Any distribution-dependent choices (e.g., HVGs, scaling parameters) must be computed per split (train fold only) and stored as artifacts.

## 5) Modeling Plan

### 5.1 Cell-level JEPA (Stage A baseline)
Inputs:
- expression vector with gene identities (explicitly represented, not implicit index-only).

Core components:
- online encoder `fθ`,
- teacher encoder `fθ̄` (EMA),
- predictor `gφ(context_repr, target_pointer) → target_repr`.

Loss:
- representation regression (cosine/L2) between predicted target and stop-grad teacher target.

Anti-collapse:
- explicit regularization (variance/covariance style) and/or normalization constraints.
- log collapse diagnostics per step.

### 5.2 Masking strategies (first-class ablation)
Minimum ablations:
- random gene mask blocks,
- biologically coherent masks (pathways/modules/regulons) once mapping is stable.

Report:
- mask fraction,
- module size distribution,
- overlap handling.

### 5.3 Perturbation transition predictors (embedding space)
v0: Prototype predictor:
- input: control prototype embedding + perturbation metadata,
- output: predicted perturbed prototype embedding.

v1: Set-level predictor:
- input: a set of control embeddings + perturbation metadata,
- output: predicted perturbed embeddings (deterministic mapping first).

Training objective:
- set distance between predicted set and observed perturbed set, computed per condition/context.

Set-level training recipe (default):
- For each step, sample a `(context_id, perturbation_id)` pair with sufficient control + perturbed cells.
- Sample `n` control cells and `n` perturbed cells (equalized set size; `n` fixed or capped).
- Map each sampled control embedding through the predictor to form the predicted perturbed set.
- Compute the set loss between predicted set and observed perturbed set (using the primary set metric).

This avoids pseudo-pairing while still training against heterogeneity.

## 6) Baselines (mandatory and non-negotiable)

Simple, strong controls:
- no-change,
- mean-shift per perturbation (optionally conditioned on cell type/context),
- ridge regression mapping in PCA / baseline embedding space,
- additive baseline for combination perturbations (if applicable).

Stretch baselines (only if feasible after harness is stable):
- established perturbation predictors (e.g., scGen/CPA/GEARS),
- a generative transformer baseline (e.g., scGPT-style) for comparison to reconstruction-heavy objectives.

Baseline fairness protocol:
- fixed tuning budget and standardized early stopping across methods,
- report compute and hyperparameters,
- publish the search space and best config selected on validation folds.

Suggested default tuning budgets (Stage A):
- ridge regression: grid over regularization strengths; select on validation set.
- PCA dimension: small fixed set (e.g., 32/64/128) validated on the same folds.
- non-parametric baselines: no tuning (report as-is).

## 7) Metrics and Reporting

### 7.1 Primary metrics (embedding-native)
- prototype error: cosine distance / MSE between predicted and observed prototypes,
- retrieval: kNN accuracy / mean reciprocal rank for retrieving the correct perturbation condition,
- set distance: pick one primary (energy distance / MMD / sliced Wasserstein) with justification.

### 7.2 Secondary (if/when decoding is added)
- DE correlation, pathway enrichment agreement, calibration for stochastic predictors.

### 7.3 Reporting artifacts
Each run outputs:
- `config.yaml`, `metrics.json`, split IDs, and a reproducible report page.

Recommended run directory layout:
- `runs/<run_id>/config.yaml`
- `runs/<run_id>/splits.json` (or references to versioned split files)
- `runs/<run_id>/metrics.json`
- `runs/<run_id>/checkpoints/`
- `runs/<run_id>/artifacts/` (plots, embeddings, cached predictions)
- `runs/<run_id>/report.md` (or HTML)

## 8) Engineering Plan (Reproducibility-First)

Repo conventions (to be enforced):
- config-driven runs (`configs/`), deterministic split files, seeded training.
- “Golden run” that completes on a small subset quickly.
- avoid hidden state: no ad-hoc notebooks as the primary execution path.

## 9) Risk Register + Mitigations

- **Collapse / shortcut learning:** log collapse metrics; enforce anti-collapse; add module masks; sanity-check against library-size predictors.
- **No wins vs linear baselines:** treat as an outcome; focus on characterizing *where* JEPA helps (OOD axes) and possibly pivot to condition-level JEPA or better masking.
- **Evaluation unconvincing:** ensure at least one downstream task (retrieval/ranking) is a primary result.
- **Engineering sprawl:** strict milestone gates; optional branches only after M3 success.

## 10) Stretch Goals (Explicitly Gated)

Only consider after M3 acceptance:
- condition-level JEPA over sets of cells,
- OT-based pseudo-pairing,
- diffusion/LLM hybrids in latent space,
- morphology/spatial integration branches.
