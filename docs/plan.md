# CellJEPA — Revised Project Plan (Execution-Ready)

Date: 2026-01-05  
Status: v4 (phase roadmap + repo hygiene)  
Canonical goal: Build a **cell-centric world model** for single-cell omics where JEPA learns a general **latent cell state representation**; evaluate primarily on **perturbation-driven state transitions** (perturbation prediction) in v1.

Operational details (data contract, splits, metrics, HPC defaults) live in `docs/runbook.md`.

## 0) One-Page Summary

### Core thesis (testable)
Learning a JEPA-style representation of cell state and predicting **post-perturbation state in embedding space** yields **better generalization under distribution shift** and **more stable/usable** perturbation prediction than objectives centered on reconstructing noisy measurements.

### Project framing (what “CellJEPA” means)
- a cell-centric “world model”: the cell’s latent state is the primary object
- JEPA is treated as a general representation learner for state (not a task-specific trick)
- perturbations are one instantiation of **state transitions** (action → next state)
- perturbation prediction is the primary evaluation regime in v1, not the only conceivable one

### Product deliverable (what exists at the end)
1. A reproducible **benchmark harness** that:
   - downloads/prepares a curated subset of perturbation datasets,
   - generates split files (holdout protocols),
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
- **v1 primary set metric:** energy distance (E-distance) over embedding sets (see §7).
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

## 2) Phased Roadmap (0–5)

### Phase 0 — Repo + reproducibility baseline (immediate)
Deliverables:
- Un-ignore and commit `docs/` (keep large PDFs ignored by default).
- Add a minimal environment spec for baseline tooling (even a small `requirements.txt` is fine).
- “Toy” end-to-end path documented in README: `make_toy_dataset → make_splits → train_jepa → train_transition`.

Gate:
- Docs are tracked, and the toy run produces `metrics.json` + `report.md` deterministically.

### Phase 1 — Data correctness + gene identity (highest leverage)
Deliverables:
- Pick and enforce a **canonical gene ID space** (Ensembl IDs or gene symbols; must be explicit).
- Add ingestion-time validation that flags mismatched ID spaces and abnormal gene universes.
- Version gene harmonization artifacts and reference them in split metadata.

Gate:
- All datasets pass the data contract + gene ID validation; harmonized datasets are reproducible.

### Phase 2 — Evaluation + safety unification (no accidental blow-ups)
Deliverables:
- Make bounded residual heads the default in **all** training/eval scripts.
- Ensure `α=0` (no-change) is included in every alpha grid and is logged.
- Enforce action overlap thresholds for cross-dataset splits (hard-fail if overlap is zero).
- Default to control-based embedding calibration; log embedding scale diagnostics.

Gate:
- Cross-dataset runs do not explode; overlap stats + no-change baselines are always reported.

### Phase 3 — Omics-appropriate JEPA (remove shortcut invariances)
Deliverables:
- Replace **zero-as-mask** with explicit mask channels or learned mask tokens.
- Teacher sees full/augmented context; predictor conditions on **mask identity**, not just ratio.
- Train from sparse data without densifying full matrices.

Gate:
- Stable training across ≥3 seeds; collapse diagnostics logged; embeddings are non-degenerate.

### Phase 4 — Cross-dataset done right (shared-action only)
Deliverables:
- Ingest at least one dataset pair with meaningful action overlap; enforce overlap at split creation.
- Keep main cross-dataset tables **shared-action only**; unseen-action results are diagnostic.
- Evaluate with safety head + calibration and report pair counts after filtering.

Gate:
- Cross-dataset runs are valid (non-zero shared-action pairs) and stable.

### Phase 5 — Showcase deliverable (the “public artifact”)
Deliverables:
- A narrative report with dataset cards, split definitions, headroom audits, baselines + CIs, and JEPA ablations.
- Explicit effect-size stratification and honest reporting of negative results.

Gate:
- Reproducible report artifact (configs + metrics + plots) suitable for public sharing.

## 3) Data Plan (Concrete, Split-Safe)

### 3.1 Dataset selection rubric (used to pick the initial 2–4)
Score each candidate on:
- perturbation type diversity (genetic vs chemical),
- availability of controls matched by context,
- metadata completeness (perturbation ID, dose, time, donor/cell line),
- size (enough cells per condition for set metrics),
- cross-dataset compatibility (gene IDs, annotation quality),
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

Gene identity policy (must be explicit and enforced):
- Choose a **canonical ID space** for v1 (Ensembl IDs or gene symbols).
- If mapping is required, store the mapping artifact + version in `uns` and record the source.
- Log the `var` index key + ID space in reports; fail fast on mixed/unknown ID spaces.

### 3.3 Preprocessing (start conservative; minimize degrees of freedom)
Initial v1 preprocessing target:
- library-size normalize → log1p (or another single fixed transform),
- fixed gene identifier standardization,
- no batch correction in v1 unless strictly split-safe and justified.

Rules:
- any statistics used by preprocessing that depend on the data distribution must be computed on the training fold only and saved per split.

### 3.4 Action/perturbation metadata schema (portable across datasets)

Define a canonical representation for actions (perturbations in v1):
- `perturbation_tokens`: list of string tokens (e.g., `["gene:STAT1"]`, `["drug:dexamethasone"]`, combos as multiple tokens)
- `dose`: numeric (optional; NaN if unknown)
- `time_hours`: numeric (optional; NaN if unknown)
- `is_control`: boolean

Default encoding strategy:
- token embedding lookup summed/pooled across tokens,
- numeric features (dose/time) passed through a small MLP and concatenated.

## 4) Split Protocols (holdout-first, enforced as code)

We define splits at the **condition level** (and optionally at the context level), then sample cells within condition/context groups.

### 4.1 Stage A main-table splits (defaults)

We standardize around two split families. Each split produces deterministic `train/val/test` condition lists and per-cell indices.

**S1 — Unseen perturbation (perturbation holdout)**  
Goal: generalize to perturbations not seen during training.
- Split key: `perturbation_id`
- Grouping rule: all cells with the same `perturbation_id` are assigned to the same fold.
- Context handling: contexts are allowed to appear in both train and test, but perturbations are disjoint.

**S2 — Unseen context (context holdout)**  
Goal: generalize to new donors/cell lines (or other context definition).
- Split key: `context_id`
- Grouping rule: all cells with the same `context_id` are assigned to the same fold.
- Perturbation handling: perturbations may overlap between train and test, but contexts are disjoint.

Defaults:
- folds: 5 (or fewer if the dataset is too small; never <3 without calling it “pilot only”)
- training seeds per fold: 3 (e.g., 0/1/2)

### 4.2 Cross-dataset holdout (Phase 4 / M4)
Hold out entire dataset(s) after harmonizing to a shared gene set (e.g., intersection or a documented “foundation set”).

Critical note (learned from initial M4 runs):
- Cross-dataset holdout mixes **domain shift** *and* often **action/perturbation vocabulary shift** (very low overlap between perturbation vocabularies across studies).
- Therefore, every cross-dataset report must include:
  - action/perturbation overlap: `|A_train ∩ A_test|`, overlap fraction(s),
  - evaluation pair counts stratified by **shared-action** vs `<UNK>`/unseen-action.
- For v1, the main cross-dataset tables are **shared-action only**. Unseen-action cross-dataset is deferred until semantic action embeddings exist.

Hard rules:
- No preprocessing statistics may use test-fold cells.
- Any learned modules/probes (e.g., adversarial masks) must be fit on training folds only.
- Any distribution-dependent choices (e.g., HVGs, scaling parameters) must be computed per split (train fold only) and stored as artifacts.

## 5) Modeling Plan

### 5.1 Cell-level JEPA (Phase 3 baseline)
Inputs:
- expression vector with gene identities (explicitly represented, not implicit index-only).
- masking must be **explicit** (mask channel or learned mask tokens), not “zeros-as-mask”.

Core components:
- online encoder `fθ`,
- teacher encoder `fθ̄` (EMA),
- predictor `gφ(context_repr, target_pointer) → target_repr`.

Loss:
- representation regression (cosine/L2) between predicted target and stop-grad teacher target.
  - teacher target should come from a **full/augmented view**, not target-only zeroed inputs.

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
- whether mask identity was provided explicitly (mask channel/tokens vs implicit zeros).

### 5.3 Perturbation transition predictors (embedding space)
v0: Prototype predictor:
- input: control prototype embedding + perturbation metadata,
- output: predicted perturbed prototype embedding.

v1: Set-level predictor:
- input: a set of control embeddings + perturbation metadata,
- output: predicted perturbed embeddings (deterministic mapping first).

Safety-by-construction (required for holdout splits and cross-dataset):
- Prefer a **residual, shrinkage-parameterized** predictor:
  - `z_hat = z_ctrl + α · Δ(z_ctrl, a)` with `α ∈ [0, 1]` (grid-tuned on validation or learned with explicit regularization),
  - include `α=0` as an explicit no-change fallback,
  - clamp/normalize `Δ` to prevent runaway updates under shift.

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
- optional safety baselines: **shrinkage** versions of mean-shift / ridge with `α ∈ [0,1]` tuned on validation,
- additive baseline for combination perturbations (if applicable).

Baseline tooling:
- Standard single-cell toolkits (scanpy/pertpy/scvi-tools) are **allowed for baselines only** if they strengthen credibility.
- Core CellJEPA training remains dependency-minimal; new runtime deps still require user approval.

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
- set distance (primary v1): **energy distance (E-distance)** over embedding sets.
  - Acceptable alternatives (explicitly justified): MMD / sliced Wasserstein.

### 7.2 Secondary (if/when decoding is added)
- DE correlation, pathway enrichment agreement, calibration for stochastic predictors.

### 7.3 Reporting artifacts
Each run outputs:
- `config.yaml`, `metrics.json`, split IDs, and a reproducible report page.

Cross-dataset runs must additionally log:
- action overlap stats (including `%<UNK>` in evaluation pairs),
- embedding scale diagnostics (e.g., norms/variance by dataset; control-vs-control sanity),
- metrics stratified by shared-action vs unseen-action (no pooling).

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
- track `docs/` in git; keep plan + reports versioned (PDFs can stay ignored).
- minimal environment spec committed for baseline tooling.

### HPC / Slurm execution context

We develop and run CellJEPA primarily on an HPC cluster using Slurm. See `docs/runbook.md`.

Defaults:
- prefer `sbatch` for training/evaluation jobs
- default walltime **> 24 hours** (recommended: `--time=48:00:00`) to avoid timeouts
- write run artifacts to `runs/<run_id>/…` and logs to `logs/`

## 9) Risk Register + Mitigations

- **Collapse / shortcut learning:** log collapse metrics; enforce anti-collapse; add module masks; sanity-check against library-size predictors.
- **No wins vs linear baselines:** treat as an outcome; focus on characterizing *where* JEPA helps (generalization axes) and possibly pivot to condition-level JEPA or better masking.
- **Cross-dataset evaluation is misleading by default:** low perturbation overlap turns “domain shift” into “unseen-action + domain shift”; always report overlap and stratify seen vs `<UNK>`.
- **Catastrophic distribution-shift blow-ups:** require residual + α-gating (including α=0 fallback) and bounded updates; treat “α→0” as an honest outcome, not a failure of reporting.
- **Evaluation unconvincing:** ensure at least one downstream task (retrieval/ranking) is a primary result.
- **Engineering sprawl:** strict milestone gates; optional branches only after M3 success.

## 10) Stretch Goals (Explicitly Gated)

Only consider after M3 acceptance:
- condition-level JEPA over sets of cells,
- OT-based pseudo-pairing,
- diffusion/LLM hybrids in latent space,
- morphology/spatial integration branches.
