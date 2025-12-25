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
- [ ] **M3** Transition predictor (prototype + set-level) + ablations
- [ ] **M4** Multi-dataset + cross-dataset holdout
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

Status label: ⬜ NOT STARTED

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

Status label: ⬜ NOT STARTED

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

Status label: ⬜ NOT STARTED

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

## Prompt M4 — Multi-Dataset + Cross-Dataset Holdout

Status label: ⬜ NOT STARTED

```text
You are working in the `CellJEPA` repository.

Goal (Milestone M4):
Scale from one dataset to 2–4 datasets and run cross-dataset holdout evaluation.

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
4) Produce a consolidated report comparing:
   - per-dataset results
   - cross-dataset holdout results
   - where CellJEPA helps vs fails

Constraints:
- Keep preprocessing consistent across datasets, or document differences explicitly.
- Guard carefully against leakage when harmonizing gene sets.

Verification (must run):
- `python3 -m compileall src`
- At least one cross-dataset holdout run producing a report artifact.

Output:
- Summarize artifacts and any blockers (e.g., dataset access issues).
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
