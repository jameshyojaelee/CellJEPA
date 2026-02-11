# CellJEPA — Project Plan (Execution-Ready)

Date: 2026-01-14  
Status: v5 (milestones + gates + contracts aligned)  
Canonical goal: Build a **cell-centric world model** for single-cell omics where JEPA learns a general **latent cell state representation**; evaluate primarily on **perturbation-driven state transitions** (perturbation prediction) in v1.

This file is the **canonical roadmap**: what we are building and the **gates** that determine whether we proceed.
Operational contracts (schemas, CLI expectations, metrics, HPC defaults) live in `docs/runbook.md`.

Doc precedence (if anything conflicts):
1) `docs/DECISIONS.md`
2) `docs/plan.md`
3) `docs/runbook.md`
4) `docs/archive/`

Current progress and run inventory: `docs/PROJECT_STATE.md`.

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

These are *defaults*, not permanent commitments. They exist to keep runs comparable and to avoid silent drift.
If a default changes, record it (with date) in `docs/DECISIONS.md` and update this section.

- **Stage A main-table splits:** `S1_unseen_perturbation` and `S2_unseen_context` (defined in §4).
- **v1 preprocessing:** library-size normalize → log1p; no batch correction.
- **v1 gene IDs:** gene symbols in `var.index` (store `var["ensembl_id"]` when available).
- **v1 action tokens:** `obs["perturbation_tokens"]` is a deterministic `|`-separated string; control token is `control:CTRL` (see `docs/runbook.md`).
- **v1 evaluation filters:** drop `perturbation_id` in `{nan, none, ""}`; default `min_cells_per_condition=30`.
- **v1 primary set metric:** energy distance (E-distance) over embedding sets (see §7).
- **Safety-by-construction:** residual/shrinkage heads with `α ∈ [0,1]`, always include `α=0` (no-change), and clamp updates.
- **Cross-dataset (M4):** shared-action only; control z-score calibration enabled by default; gene harmonization uses intersection sets (see `configs/harmonization/README.md`).
- **Baselines are mandatory:** always report `no_change`, `mean_shift`, and `ridge` (plus shrinkage variants when relevant).
- **JEPA backbone (Stage A):** choose the simplest stable implementation first, then ablate alternatives later.

## 0.2 Stop-the-Line Invariants (do not proceed if violated)

These are non-negotiable because violations create misleading results and waste time.

- **Data contract passes:** every `.h5ad` used in any run passes `celljepa.data.validation.validate_or_raise` (see `docs/runbook.md`).
- **Split artifacts are real artifacts:** train/val/test are group-based, deterministic, saved as JSON, and reused across methods.
- **No leakage:** any distribution- or label-dependent preprocessing is fit on **train only** and saved per split; never “peek” at test perturbed cells to choose features/params.
- **Evaluation unit is condition pairs:** aggregate metrics over `(context_id, perturbation_id)` pairs (not cell-weighted averages).
- **Baselines cannot be optional:** `no_change` must always be reported; if the model does not beat `no_change` anywhere, we do not claim “useful world model”.
- **Cross-dataset must report overlap:** always report action overlap and restrict main tables to **shared-action** evaluation; hard-fail cross-dataset if overlap is zero.
- **No non-finite values:** embeddings/predictions/metrics must be finite; any NaNs/Infs are a “stop and fix” event.

## 1) Definitions (so experiments are unambiguous)

- **Cell state embedding**: `z = fθ(x)` where `x` is a cell’s expression vector (after a fixed preprocessing contract).
- **Action / perturbation condition**: metadata `a` describing an intervention (gene KO, drug, dose, time). In v1, actions are perturbations.
- **State transition**: mapping from baseline/control state distribution to post-action state distribution within a context.
- **Baseline context**: the *control* distribution for a context (donor/cell line/cell type/batch), used as the “pre-perturbation” reference.
- **Prediction target**: the **post-perturbation embedding distribution** for condition `a` within a context.

We explicitly support two prediction granularities:
- **Prototype-level** (debug-first): predict condition mean/robust-mean embedding.
- **Set-level** (core): predict a set/distribution of embeddings and compare to the empirical perturbed set via set metrics.

## 2) Milestone Roadmap (M0–M5)

Naming note:
- “M0…M5” correspond to the `runs/m0_*`, `runs/m1_*`, … run prefixes and are the primary way we refer to progress.
- Older notes may say “Phase”; treat “Phase” as synonymous with “Milestone”.

### M0 — Repo + reproducibility baseline (immediate)
Deliverables:
- Track `docs/` in git (keep large PDFs ignored by default).
- Keep the toy quickstart in `docs/runbook.md` running end-to-end and producing artifacts under `runs/`.
- Add a minimal environment spec (at least one of: `requirements.txt`, `environment.yml`, or `pyproject.toml`).
- Maintain a “golden” smoke check: `python3 -m compileall src scripts`.

Gate:
- The toy quickstart produces `metrics.json` + `report.md` under `runs/` and can be rerun without errors.

### M1 — Data correctness + baseline harness (highest leverage)
Deliverables:
- Enforce the v1 data contract at ingestion time (required `obs`/`uns` keys; unique genes; see §3.2 and `docs/runbook.md`).
- Lock the v1 gene identity policy (gene symbols) and version harmonization artifacts.
- Ingest the v1 dataset suite and write processed `.h5ad` artifacts (see `docs/runbook.md`).
- Generate split files for `S1_unseen_perturbation` and `S2_unseen_context` and store them as reusable artifacts.
- Run the baseline harness (`no_change`, `mean_shift`, `ridge`) and produce reports/tables.
- Run a headroom audit before spending weeks optimizing a baseline-saturated benchmark.

Gate:
- Every v1 dataset passes contract validation; baseline runs complete for S1/S2; at least one dataset/split shows headroom beyond baselines (or we record “baseline-saturated” and change targets in `docs/DECISIONS.md`).

### M2 — Omics-appropriate JEPA pretraining (representation substrate)
Deliverables:
- Replace “zeros-as-mask” with explicit mask channels/tokens; teacher targets come from full/augmented views.
- Predictor conditions on mask identity (not only mask ratio).
- Split-safe pretraining: pretraining can be restricted to train (or train+val) via split artifacts.
- Anti-collapse: variance/covariance or normalization constraints with logged diagnostics.
- Memory safety: train without densifying full matrices (or make densification an explicit, bounded debug mode).

Gate:
- Stable training across ≥3 seeds; non-degenerate embeddings; collapse diagnostics logged; no OOM on intended dataset scale.

### M3 — Transition/world-model training + safety + within-dataset acceptance
Deliverables:
- Train prototype and set-level transition predictors in embedding space.
- Safety-by-construction is the default: residual/shrinkage heads with alpha grids that always include `α=0`.
- Evaluation produces bootstrap CIs and always includes `no_change`, `mean_shift`, `ridge` baselines (ideally computed in the same run to ensure identical filtering/resampling).
- Filtering defaults applied (`min_cells_per_condition=30`, drop nan perturbations) and recorded in metrics.
- M3 acceptance is allowed on `S2_unseen_context` only (per `docs/DECISIONS.md`) unless explicitly tightened.

Acceptance rule (“win with CI”, default):
- For a lower-is-better metric (e.g., E-distance, MSE): `model_ci95_hi < baseline_ci95_lo`.
- Require the win to hold in ≥2 of 3 seeds for the target setting.

Gate:
- **Baseline win:** beat `ridge` with CI on the chosen acceptance setting (default target lives in `docs/DECISIONS.md`).
- **Usefulness win:** beat `no_change` with CI on at least one within-dataset setting (any dataset/split), or we do not claim “useful world model”.
- No catastrophic blow-ups: residual alpha selection never “hides” divergence (alpha=0 is logged and may win).

### M4 — Cross-dataset done right (shared-action only)
Deliverables:
- Harmonize by intersection gene sets for the first pass (`configs/harmonization/README.md`).
- Create cross-dataset splits that enforce meaningful action overlap (hard-fail if overlap is below threshold; use `scripts/m4_make_cross_dataset_splits.py --require-min-action-overlap` and/or `--require-min-action-overlap-frac`).
- Evaluate shared-action only for the main cross-dataset tables; unseen-action is diagnostic.
- Enable control-based embedding z-score calibration by default, with an ablation to disable; always report calibration stats.
- Report overlap and evaluation pair counts after filtering; never pool across shared/unseen action.

Gate:
- Cross-dataset runs are valid (non-zero shared-action pairs), stable (no NaNs/Infs), and include `no_change` baselines + overlap stats in every report.

### M5 — Public artifact + multimodal extension (gated)
Deliverables:
- A narrative report with dataset cards, split definitions, headroom audits, baselines + CIs, and JEPA ablations.
- Optional: extend to RNA+protein (Perturb-CITE-seq) with an explicit, separate acceptance gate.

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

Current repo state (what we actually run) is documented in `docs/runbook.md` and `docs/PROJECT_STATE.md`.

### 3.2 Data contract (implementation requirement)
Operational details (including the exact schema) live in `docs/runbook.md`. This section is a *summary*.

Hard requirement: every processed dataset artifact must pass the repo’s validator (`celljepa.data.validation.validate_or_raise`).

Required fields (v1):
- `X`: numeric expression matrix (fixed preprocessing).
- `var.index`: gene identifiers (must be unique).
- `obs` columns:
  - `perturbation_id` (string)
  - `is_control` (bool)
  - `context_id` (string)
  - `perturbation_tokens` (string; deterministic serialization, see §3.4)
- `uns` keys (provenance):
  - `dataset_id`
  - `preprocess_name`
  - `preprocess_version`
  - `created_at`

Recommended `obs` columns when available:
- `cell_type`, `batch`, `dose`, `time_hours`.

Gene identity policy (v1, must be explicit and enforced):
- Canonical v1 policy is **gene symbols in `var.index`**.
- Store an Ensembl mapping column when available (e.g., `var["ensembl_id"]`) to reduce ambiguity.
- If you remap IDs, store mapping provenance + version in `uns` and/or alongside the artifact; fail fast on mixed/unknown ID spaces.

### 3.3 Preprocessing (start conservative; minimize degrees of freedom)
Initial v1 preprocessing target:
- library-size normalize → log1p (or another single fixed transform),
- fixed gene identifier standardization,
- no batch correction in v1 unless strictly split-safe and justified.

Rules:
- any statistics used by preprocessing that depend on the data distribution must be computed on the training fold only and saved per split.

### 3.4 Action/perturbation metadata schema (portable across datasets)

Define a canonical representation for actions (perturbations in v1):
- `obs["perturbation_tokens"]`: a deterministic `|`-separated string of tokens:
  - genes: `gene:STAT1`
  - drugs: `drug:dexamethasone`
  - controls: `control:CTRL`
  - combos: multiple tokens in deterministic order, e.g. `gene:STAT1|gene:IRF9`
- Optional numeric features stored per cell: `obs["dose"]`, `obs["time_hours"]` (NaN if unknown)
- `obs["is_control"]` remains the canonical control flag

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

Important: S1 is **not meaningful** if the transition model encodes perturbations as categorical IDs (everything is `<UNK>` at test).
To make S1 meaningful, use an action representation that generalizes beyond the training perturbation vocabulary (e.g., token embeddings or split-safe gene embeddings; see `configs/actions/` and `docs/DECISIONS.md`).

**S2 — Unseen context (context holdout)**  
Goal: generalize to new donors/cell lines (or other context definition).
- Split key: `context_id`
- Grouping rule: all cells with the same `context_id` are assigned to the same fold.
- Perturbation handling: perturbations may overlap between train and test, but contexts are disjoint.

Defaults:
- folds: 5 (or fewer if the dataset is too small; never <3 without calling it “pilot only”)
- training seeds per fold: 3 (e.g., 0/1/2)

### 4.2 Cross-dataset holdout (M4)
Hold out entire dataset(s) after harmonizing to a shared gene set (v1 default: intersection sets; see `configs/harmonization/README.md`).

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

### 5.1 Cell-level JEPA (M2 baseline)
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
- biologically coherent masks (pathways/modules/regulons) once mapping is stable (see `configs/modules/README.md` and `scripts/build_module_masks.py`).

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

### 7.3 Reporting + artifact contract

Minimum per run (non-negotiable):
- `runs/<run_id>/metrics.json` (machine-readable; includes dataset/split IDs, filters, and CIs when applicable).

Recommended (and required for M5 “public artifact” reproducibility):
- `runs/<run_id>/config.json` capturing CLI args / config snapshot (or embed the equivalent under `metrics["config"]`).
- `runs/<run_id>/report.md` as a human-readable summary (produced either by the run script or by a report generator under `scripts/`).

Current script behavior (so we don’t lie to ourselves):
- `scripts/eval_baselines.py`: writes `metrics.json` + `report.md`.
- `scripts/train_jepa.py`: writes `checkpoint.pt`, `metrics.json`, `config.json` (and may write `embedding_metrics.json`).
- `scripts/train_transition.py` / `scripts/train_world_model.py`: write `model.pt` + `metrics.json` (reports are generated separately).
- `scripts/m4_cross_dataset_eval.py`: writes `metrics.json` (reports are generated separately).

Cross-dataset runs must additionally log:
- action overlap stats (including `%<UNK>` in evaluation pairs),
- embedding scale diagnostics (e.g., norms/variance by dataset; control-vs-control sanity),
- metrics stratified by shared-action vs unseen-action (no pooling).

Recommended run directory layout:
- `runs/<run_id>/metrics.json`
- `runs/<run_id>/config.json` (when available)
- `runs/<run_id>/checkpoint.pt` or `runs/<run_id>/model.pt`
- `runs/<run_id>/artifacts/` (plots, embeddings, cached predictions)
- `runs/<run_id>/report.md` (when generated)

## 8) Engineering Plan (Reproducibility-First)

Repo conventions (to be enforced):
- config-snapshotted runs (`configs/` + per-run `config.json` where applicable), deterministic split files, seeded training.
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
- **Beating ridge is not enough:** if we beat `ridge` but still lose to `no_change`, we treat it as “not yet useful”; focus on characterizing *where* JEPA helps (generalization axes) and tighten the usefulness gate before claiming success.
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
