# CellJEPA — Mega-Prompts (Run the Plan)

These prompts are designed to be copied into an agentic coding environment (Codex CLI / ChatGPT w/ repo access) to execute `docs/plan.md` milestone-by-milestone.

Guidance:
- Run prompts in order (M0 → M5). Do not skip baselines/splits.
- Keep changes small and verifiable; prefer “one milestone per prompt.”
- Always follow `AGENTS.md`.
- Keep APIs general for state transitions; perturbations are the v1 action type and primary evaluation regime.

Execution context:
- We primarily develop/run on HPC using **Slurm**. See `docs/HPC.md`.
- For any run that may take more than a few minutes, prefer generating an `sbatch` script.
- Default to **walltime > 24 hours** (recommended: `--time=48:00:00`) to avoid timeouts.

## Status Tracker (update this as you go)

Use this checklist so a new Codex session can safely resume without redoing work.

- [x] **M0** Contracts + split skeleton (completed 2025-12-25; artifacts: `scripts/make_toy_dataset.py`, `scripts/make_splits.py`, `runs/m0_splits/`)
- [x] **M1** Ingest real dataset(s) + baselines + golden report (Sci-Plex2/3/4 + Norman2019 completed)
- [x] **M2** JEPA pretraining + embedding export + diagnostics (fast-dev run completed on Sci‑Plex3)
- [x] **M3** Transition predictor (prototype + set-level) + ablations (**accepted** on Replogle S2 via ridge win with CIs; no-change still best; see `reports/parallel_m3_progress.md`)
- [ ] **M4** Multi-dataset + cross-dataset holdout (**in progress**; gene overlap + harmonization + initial cross-dataset evals ran. Key blocker: shared non-control perturbation overlap is ~0 for Sci‑Plex2/4→Sci‑Plex3 and 8 genes for Norman→Replogle, so M4‑v2 defines cross‑dataset on shared actions only + control z‑score calibration by default; see `reports/m4_gene_overlap.md`, `reports/m4_cross_dataset_results.md`)
- [ ] **M5** (Stretch) Multi-modal RNA+protein

## Decision Gates (ask the user before proceeding)

Do **not** silently decide these in a new session; ask the user to confirm:

1) **Dataset choice (M1):** which dataset to ingest first, where it comes from, and any access constraints (manual download, credentials, internal mirrors).
2) **Primary set-distance metric (M3):** **E-distance** chosen for v1 (2025-12-25).
3) **Preprocessing scope (M1):** log1p-only vs HVGs vs other transforms; any batch correction is a separate, explicitly approved decision due to leakage risk.
4) **Backbone choice (M2):** “simplest stable” implementation details (e.g., MLP/Transformer tokenization choices); confirm if we should match a specific prior architecture.
5) **Module mask sources (M2/M3):** which gene set collections/regulon sources to use (versioned), if enabling module masks.
6) **Cross-dataset gene harmonization (M4):** intersection vs “foundation set” definition; this affects comparability.
7) **New runtime dependencies:** always ask before adding.
8) **M3 acceptance target (if Sci-Plex3 is baseline-saturated):** confirm which dataset/split will be treated as the M3 “acceptance” target to unblock M4.
9) **M3/M4 usefulness gate:** ✅ confirmed (2025-12-31): “success” requires beating **no-change** somewhere (not just ridge).
10) **Cross-dataset evaluation definition (M4):** ✅ confirmed (2025-12-31): **shared-action cross-dataset only** for the main table; enable **control-based embedding calibration (z-score on controls)** by default (ablate off).

---

## Prompt M0 — Contracts + Split Generator Skeleton

Status label: ✅ DONE (2025-12-25)

```text
You are working in the `CellJEPA` repository.

Goal (Milestone M0):
1) Make the repo “execution-ready” by implementing the minimal *contracts* and a *split generator skeleton*.
2) Do not implement JEPA yet.

Read and obey:
- `AGENTS.md`
- `docs/plan.md`
- `docs/data_contract.md`
- `docs/splits.md`
- `docs/metrics.md`

Deliverables (required):
1) Create a Python module layout under `src/celljepa/`:
   - `src/celljepa/data/` (dataset registry + validation helpers)
   - `src/celljepa/splits/` (split generation)
   - `src/celljepa/eval/` (metric stubs + report skeleton)
2) Add a deterministic split generator CLI:
   - `scripts/make_splits.py` with `--dataset-id`, `--split-name`, `--seed`, `--out`
   - Output a JSON split file in the recommended format from `docs/splits.md`.
3) Add data-contract validation utilities:
   - A function that validates an `anndata.AnnData` object conforms to `docs/data_contract.md` (check required `obs` columns, `var.index`, etc.).
4) Add a tiny *toy dataset* generator so M0 can be verified without real data:
   - `scripts/make_toy_dataset.py` writes a minimal `.h5ad` (or equivalent) with control + perturbed cells and a couple contexts.

Constraints:
- Ask before adding runtime dependencies.
- Do not download real datasets yet.
- Ensure deterministic behavior given seeds.

Verification (must run):
- `python3 -m compileall src`
- Run `scripts/make_toy_dataset.py` and then `scripts/make_splits.py` on it for `S1_unseen_perturbation` and `S2_unseen_context`.

Output:
- Make actual file changes.
- Summarize files touched and commands run.
- If something is ambiguous, ask a single targeted question.
```

---

## Prompt M1 — Ingest One Real Dataset + Baseline Harness + Golden Report

Status label: ✅ DONE (2025-12-25)

```text
You are working in the `CellJEPA` repository.

Goal (Milestone M1):
Ingest ONE real dataset (as chosen in `docs/datasets.md`), produce split files, run simple baselines, and generate a “golden run” report artifact.

Read and obey:
- `AGENTS.md`
- `docs/plan.md`
- `docs/data_contract.md`
- `docs/splits.md`
- `docs/metrics.md`
- `docs/datasets.md`

Before coding:
1) Identify the exact dataset to ingest (fill in `docs/datasets.md` with the chosen `dataset_id` and notes).
2) Confirm what format/source is expected (if not specified, ask a targeted question).
3) Confirm preprocessing scope (default: libnorm→log1p only; no batch correction) and stop if the user wants something else.

Deliverables (required):
1) Implement ingestion for the chosen dataset under `src/celljepa/data/`:
   - Must write a processed `.h5ad` into `data/processed/<dataset_id>/<preprocess_version>.h5ad`
   - Must populate required `obs` fields: `perturbation_id`, `context_id`, `is_control`, `perturbation_tokens`
   - Must populate `uns` provenance fields from the data contract.
2) Implement baseline models:
   - no-change baseline
   - mean-shift baseline (per perturbation; optionally stratified by cell_type if present)
   - ridge regression baseline in PCA space (with a small validation-selected alpha grid)
3) Implement evaluation runner:
   - `scripts/eval_baselines.py --dataset <...> --split <...> --out runs/<run_id>/`
   - Writes `runs/<run_id>/metrics.json` and `runs/<run_id>/report.md`
4) The report must include:
   - dataset summary table
   - split definition
   - baseline results table with confidence intervals (bootstrap over condition pairs)

Constraints:
- No data leakage: any statistics beyond fixed log1p must be fit on train only.
- Keep dependencies minimal; ask before adding runtime deps.
- Do not add JEPA yet.

Verification (must run):
- `python3 -m compileall src`
- Run the full golden path on the ingested dataset for S1 and S2 splits:
  - generate splits
  - run baselines
  - produce reports

Output:
- Summarize files touched, commands run, and where the report lives.
- If ingestion requires credentials or manual download, specify exact steps and stop.
```

---

## Prompt M2 — Implement Cell-Level JEPA Pretraining (Stable, With Diagnostics)

Status label: ✅ DONE (2025-12-25)

```text
You are working in the `CellJEPA` repository.

Goal (Milestone M2):
Implement stable JEPA pretraining to produce cell embeddings, with explicit collapse diagnostics and at least two masking strategies.

Read and obey:
- `AGENTS.md`
- `docs/plan.md`
- `docs/data_contract.md`
- `docs/splits.md`
- `docs/metrics.md`

Deliverables (required):
1) Implement JEPA components under `src/celljepa/models/`:
   - online encoder, teacher encoder (EMA), predictor head
   - anti-collapse regularization and logging of collapse metrics
2) Implement masking:
   - random gene masking (required)
   - module masks (optional if module mapping is not ready; otherwise implement)
3) Add training script:
   - `scripts/train_jepa.py --dataset ... --split ... --config ... --out runs/<run_id>/`
   - Writes: config snapshot, checkpoints, metrics, embeddings export
4) Add an embedding extraction/eval script:
   - computes simple embedding sanity checks (variance, kNN retrieval if labels exist)
   - logs results to `runs/<run_id>/metrics.json`

Constraints:
- Start with the simplest backbone that trains reliably.
- Deterministic seeding: runs should be comparable.
- Keep memory and compute reasonable; add a “fast dev” mode.

Verification (must run):
- `python3 -m compileall src`
- A short “fast dev” training run that completes and produces embeddings + metrics.

Output:
- Summarize files touched, commands run, and where embeddings/metrics are written.
- If training collapses, diagnose and propose the smallest stability fix.
```

---

## Prompt M3 — Perturbation Transition Predictor (Prototype → Set-Level)

Status label: ✅ DONE (accepted on Replogle S2 via ridge win with CIs; see `reports/parallel_m3_progress.md`)

```text
You are working in the `CellJEPA` repository.

Goal (Milestone M3):
Train perturbation prediction models in embedding space:
1) Prototype predictor (debug milestone)
2) Set-level predictor + set loss (core)
Then evaluate against baselines on the Stage A main-table splits.

Read and obey:
- `AGENTS.md`
- `docs/plan.md`
- `docs/metrics.md`

Deliverables (required):
1) Prototype predictor:
   - input: control prototype embedding + perturbation metadata encoding
   - output: predicted perturbed prototype embedding
2) Set-level predictor:
   - implement the set-level training recipe from `docs/plan.md`
   - implement one primary set distance metric (energy distance or MMD) consistently across training/eval
3) Evaluation + reporting:
   - compare: no-change, mean-shift, ridge, plus JEPA-based predictors
   - produce per-condition and aggregate metrics with confidence intervals
   - write `runs/<run_id>/report.md`
4) Ablations (minimum):
   - teacher EMA on/off
   - random vs module masks (if module masks exist)
   - anti-collapse reg on/off

Constraints:
- No OT pseudo-pairing in v1 unless explicitly gated in as stretch.
- Avoid strawman baselines; use the baseline tuning protocol.

Decision gate:
- Before implementing set-level objectives, confirm the primary set-distance metric with the user (E-distance vs MMD) and document it in `docs/metrics.md`.

Verification (must run):
- `python3 -m compileall src`
- One full run on S1 and S2 splits producing a report.

Output:
- Summarize results location and whether acceptance criteria are met.
- If results do not beat baselines, characterize failure modes and propose next ablations.
```

---

## Prompt M3A — Headroom Audit (Sci-Plex3 S2) + Pick M3 Acceptance Target (Fast Path to M4)

Status label: ✅ DONE (Sci-Plex3 S2 baseline-saturated; acceptance target: Replogle S2)

```text
You are working in the `CellJEPA` repository.

Goal:
Determine whether Sci-Plex3 S2 (unseen context) has real headroom beyond strong baselines under the primary set metric (E-distance), and if not, pick an alternative M3 acceptance dataset/split with headroom so we can defensibly proceed to M4.

Read and obey:
- `AGENTS.md`
- `docs/plan.md`
- `docs/metrics.md`
- `docs/m3_strategy.md`

Constraints:
- No new runtime dependencies without asking.
- No leakage for any “acceptance” results.
- It is OK to satisfy M3 on S2 only (user-confirmed).

Deliverables (required):
1) Implement a headroom audit script:
   - `scripts/headroom_audit.py`
   - Inputs: `--dataset`, `--checkpoint`, `--split`, `--out`
   - Outputs: `runs/<run_id>/report.md` and `runs/<run_id>/metrics.json`
   - Must compute:
     a) split-safe baselines (mean ± CI): no-change, mean-shift, ridge (embedding-space; train contexts only)
     b) ORACLE analyses (explicitly labeled as oracle/leaky; do not use for acceptance):
        - oracle per-perturbation shift computed from test contexts
        - oracle per-(context, perturbation) shift computed from test pairs
        - optional: oracle linear map fit on test prototypes
     c) diagnostics: pair counts and filtering statistics
2) Run the audit for Sci-Plex3 S2 and write:
   - `runs/m3_headroom_sciplex3_s2/report.md`
3) If headroom looks tiny (oracles barely better than baselines), run the audit for at least one candidate acceptance dataset/split (default candidate: Replogle S2) and write:
   - `runs/m3_headroom_<candidate>/report.md`
4) Update docs:
   - `docs/PROJECT_STATE.md` with the headroom decision and next actions
   - `docs/DECISIONS.md` with the chosen M3 acceptance dataset/split if Sci-Plex3 is baseline-saturated

HPC/Slurm (recommended):
- Use `scripts/slurm/submit_headroom_audit.sh` and `scripts/slurm/headroom_audit.sbatch` to run the audits on the cluster.

Verification (must run):
- `python3 -m compileall src scripts`

Output:
- Summarize files touched, commands run, and the final “acceptance target” recommendation.
```

---

## Prompt M3B — Make JEPA More “JEPA-like” (Module Masks + Split-Safe S2 Pretraining)

Status label: 🟡 IN PROGRESS (split-safe random-mask pretraining queued; module masks pending MSigDB GMT files)

```text
You are working in the `CellJEPA` repository.

Goal:
Improve representation learning so that transition prediction is not baseline-saturated: use module masking and ensure JEPA pretraining is split-safe for S2 (train contexts only).

Read and obey:
- `AGENTS.md`
- `docs/plan.md`
- `docs/metrics.md`
- `docs/m3_strategy.md`

Decision gate (ask user before proceeding):
- Which module mask source and version to use, and in what gene ID space (must be compatible with `adata.var_names`).

Deliverables (required):
1) Versioned module file (no new deps):
   - `configs/modules/<source>_<version>_<gene_id_space>.json`
   - Format: list of gene lists OR `{module_name: {genes: [...]}}`
2) Split-safe JEPA pretraining for S2:
   - Add a way to train JEPA on only train-split cells for an S2 split (context holdout).
   - Ensure the run artifacts record exactly which split was used for pretraining.
3) Rerun M3 transition evaluation on the chosen M3 acceptance dataset/split:
   - 3 seeds
   - baselines + CIs
   - summarize in `reports/m3_<dataset>_s2_report.md`

Constraints:
- No leakage: pretraining must not use test contexts for S2.

Verification:
- `python3 -m compileall src scripts`

Output:
- Summarize whether representation upgrades increase headroom and whether we now beat at least one strong baseline on S2 with CIs.
```

---

## Prompt M3C — Action-Conditioned Set-to-Set World Model (Core JEPA Pivot)

Status label: 🟡 IN PROGRESS (world model jobs scheduled for Replogle S2; awaiting split-safe checkpoints)

```text
You are working in the `CellJEPA` repository.

Goal:
Implement an action-conditioned latent world model that predicts post-perturbation embedding distributions given a baseline population (control set) and an action embedding.

Read and obey:
- `AGENTS.md`
- `docs/plan.md`
- `docs/metrics.md`
- `docs/m3_strategy.md`

Deliverables (required):
1) Implement a minimal set-to-set world model under `src/celljepa/models/`:
   - Set encoder for control population (DeepSets or lightweight SetTransformer-style)
   - Action encoder (tokens + optional numeric metadata)
   - Conditional predictor that outputs a predicted embedding set
2) Training script `scripts/train_world_model.py`:
   - split-safe data construction
   - set-level loss (E-distance)
   - produces `runs/<run_id>/metrics.json` and `runs/<run_id>/report.md`
3) Evaluation:
   - compare to no-change / mean-shift / ridge baselines in embedding space
   - 3 seeds on the M3 acceptance dataset S2
4) Ablations (minimum):
   - teacher EMA on/off (if applicable)
   - random vs module masks (if available)
   - reg on/off

Constraints:
- Keep the first implementation minimal; add one complexity knob at a time.

Verification:
- `python3 -m compileall src scripts`

Output:
- Summarize whether the world model beats at least one strong baseline on S2 with CIs.
```

---

## Prompt M3D (Optional) — Meaningful Action Encoding (Prepare for S1 + M5)

Status label: 🟡 IN PROGRESS (gene harmonization + cross-dataset plan starting)

```text
You are working in the `CellJEPA` repository.

Goal:
Make action embeddings meaningful so that S1 (unseen perturbation) is not structurally blocked by `<UNK>` actions, and prepare action representations for multi-modal M5.

Read and obey:
- `AGENTS.md`
- `docs/plan.md`
- `docs/m3_strategy.md`

Deliverables (required):
1) Define a versioned action feature artifact (no new deps by default):
   - `configs/actions/<dataset_id>_<version>.json`
2) Extend ingestion/metadata encoding to use richer action tokens (dataset-dependent):
   - normalized drug names, dose/time bins if present
   - mechanism-of-action categories if available from curated metadata
3) Update the transition/world model to consume these action features.
4) Evaluate on S1 only after action features exist.

Decision gate (ask user before proceeding):
- Whether to use external drug-target/pathway resources (will require downloads + careful versioning).

Verification:
- `python3 -m compileall src scripts`

Output:
- Summarize whether S1 becomes meaningfully learnable and how much action features help.
```

---

## Prompt M4 — Multi-Dataset + Cross-Dataset Holdout

Status label: 🟡 IN PROGRESS (v1 cross-dataset evals completed; requires M4-v2 to avoid conflating generalization axes)

```text
You are working in the `CellJEPA` repository.

Goal (Milestone M4):
Scale from one dataset to 2–4 datasets and run cross-dataset holdout evaluation in a way that does **not** conflate domain shift with unseen-action evaluation.

Read and obey:
- `AGENTS.md`
- `docs/plan.md`
- `docs/data_contract.md`
- `docs/splits.md`
- `docs/datasets.md`

Deliverables (required):
1) Extend ingestion to additional datasets listed in `docs/datasets.md`.
2) Implement gene-set harmonization strategy (documented; versioned).
3) Implement cross-dataset split generation and evaluation runner.
4) Add cross-dataset **overlap diagnostics** (mandatory):
   - perturbation/action overlap between train/test datasets: `|A_train ∩ A_test|` and overlap fraction(s)
   - `%<UNK>` in evaluation pairs when vocab is built from train only
   - define the main cross-dataset results table on **shared-action only** (train-vocab actions); log `<UNK>`/unseen-action only as diagnostics
5) Make cross-dataset evaluation **safe-by-construction**:
   - residual + shrinkage head (α-gating including α=0 fallback)
   - bounded deltas (clamp/normalization) to prevent runaway updates
   - control-based embedding calibration (z-score on controls) **enabled by default**, clearly logged; add an ablation flag to disable
6) Produce a consolidated report comparing:
   - per-dataset results
   - cross-dataset holdout results (shared-action only main table + unseen-action diagnostics)
   - where CellJEPA helps vs fails

Decision gate:
- ✅ confirmed (2025-12-31): **shared-action cross-dataset only** for the main table; unseen-action is deferred unless semantic action embeddings are introduced (separately gated).

Constraints:
- Keep preprocessing consistent across datasets, or document differences explicitly.
- Guard carefully against leakage when harmonizing gene sets.

Verification (must run):
- `python3 -m compileall src`
- At least one cross-dataset holdout run producing a report artifact.

Output:
- Summarize artifacts, overlap diagnostics, and any blockers (e.g., `<UNK>` dominates due to near-zero overlap).
```

---

## Prompt M5 (Stretch) — Multi-Modal RNA+Protein

Status label: ⬜ NOT STARTED

```text
You are working in the `CellJEPA` repository.

Goal (Milestone M5):
Add multi-modal support (RNA + protein) and evaluate whether multi-view JEPA improves robustness.

Gate:
Only proceed if M3 acceptance criteria are met and documented in run reports.

Deliverables (required):
1) Ingest the multimodal dataset and produce a processed artifact with a clear schema for RNA/protein.
2) Implement a multi-view JEPA variant (document the design).
3) Evaluate against unimodal baselines and report gains/limitations.

Constraints:
- Do not add major dependencies without asking.
- Keep the evaluation comparable to Stage A protocols.
```
