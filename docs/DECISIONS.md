# CellJEPA - Decision Log

Purpose: capture user-confirmed decisions made at critical turning points, with brief rationale and impact.

Update rules:
- Add a new entry only after the user explicitly confirms a decision in response to a question.
- Use ISO dates (YYYY-MM-DD) and keep entries short and scannable.
- If a decision is reversed, mark the old entry as superseded and link to the new one.
- If a decision changes project direction, update `docs/plan.md` first (then update this log).

Template:

## YYYY-MM-DD - <decision title>
- Status: active | superseded
- Decision: <one sentence>
- Rationale: <one sentence>
- Alternatives: <short list or note>
- Impacts: <what changes downstream>
- References: `docs/...` or `reports/...`
- Owner: user

---

## Seeded entries (from existing repo state; confirm/supersede as needed)

## 2026-01-05 - Track docs/ in git (PDFs remain ignored)
- Status: active
- Decision: Un-ignore `docs/` and commit plan/reports to version control; keep large PDFs ignored by default.
- Rationale: Docs are part of the source of truth and must be versioned for reproducibility.
- Alternatives: keep docs local-only; selectively track only a subset of markdown files.
- Impacts: `.gitignore` updated; future plan/report changes are committed.
- References: `docs/plan.md`, `.gitignore`
- Owner: user

## 2026-01-05 - Allow standard single-cell tooling for baselines only
- Status: active
- Decision: Standard single-cell toolkits (e.g., scanpy/pertpy/scvi-tools) may be added **for baselines only** if they strengthen credibility.
- Rationale: External baselines improve interpretability and external validity; core model stays dependency-minimal.
- Alternatives: keep dependency-minimal baselines only.
- Impacts: Baseline evaluation may use additional runtime dependencies; core CellJEPA training remains lightweight.
- References: `docs/plan.md`
- Owner: user

## 2025-12-25 - M3 acceptance can be satisfied on S2 only (unseen context)
- Status: active
- Decision: For M3 acceptance gating, it is acceptable to target **S2_unseen_context** only (S1 is deferred until action embeddings/generalization are improved).
- Rationale: Sci-Plex3 S1 is structurally hard without meaningful action encodings; we prioritize a defensible win vs baselines on S2 to unblock M4.
- Alternatives: require wins on both S1 and S2 before M4.
- Impacts: M3 evaluation focus, reporting, and gating for proceeding to M4.
- References: `docs/m3_strategy.md`, `docs/plan.md`, `docs/prompts.md`
- Owner: user

## 2025-12-25 - M3 evaluation filtering: drop nan perturbations and enforce minimum cells per condition
- Status: active
- Decision: Exclude `perturbation_id == "nan"` (and empty variants) and require a minimum number of cells per (context_id, perturbation_id) condition (default: 30).
- Rationale: Avoid noisy/degenerate condition pairs that destabilize metrics and inflate variance.
- Alternatives: keep all pairs but downweight; lower threshold.
- Impacts: Pair construction, reported `n_eval`, and comparability across runs (must record threshold in metrics).
- References: `scripts/train_transition.py`, `docs/metrics.md`
- Owner: user

## 2025-12-25 - Fast path to a defensible M4: run headroom audit and pick an M3 acceptance dataset with headroom
- Status: active
- Decision: If Sci-Plex3 S2 is baseline-saturated under our primary metric, pivot M3 acceptance to an alternative dataset/split with real headroom (default candidate: Replogle S2), then proceed to M4 only after a baseline win with CIs.
- Rationale: Prevent spending weeks optimizing on a baseline-saturated benchmark; ensure M4 is gated by a defensible M3 win.
- Alternatives: revise the M3 acceptance gate; proceed to M4 without an M3 win.
- Impacts: Adds a headroom audit step and potentially changes the dataset used to satisfy M3 acceptance.
- References: `docs/m3_strategy.md`, `docs/plan.md`
- Owner: user

## 2025-12-25 - M3 acceptance target selected: Replogle S2 (context OOD)
- Status: active
- Decision: Use **Replogle S2_unseen_context** as the M3 acceptance target (Sci-Plex3 S2 is baseline-saturated under E-distance).
- Rationale: Headroom audit showed minimal improvement over baselines on Sci-Plex3 S2, while Replogle S2 shows large oracle headroom.
- Alternatives: keep Sci-Plex3 S2; choose another dataset/split.
- Impacts: M3B/M3C runs will focus on Replogle S2; M4 remains gated on a baseline win with CIs there.
- References: `runs/m3_headroom_sciplex3_s2/report.md`, `runs/m3_headroom_replogle_s2/report.md`, `docs/m3_strategy.md`
- Owner: user

## 2025-12-25 - M3 accepted based on ridge win (Replogle S2)
- Status: active
- Decision: Mark M3 as accepted based on **Replogle S2** set‑predictor wins vs **ridge** with CIs (config `mask0.25_var0.5_cov0.5`, seeds 1–2), and proceed to M4.
- Rationale: Acceptance criterion requires beating at least one strong baseline with CIs; ridge is the strongest parametric baseline and was beaten with non‑overlapping CIs.
- Alternatives: require beating no‑change; continue M3 until no‑change is beaten.
- Impacts: M4 can start immediately; M3 remains noted as a mixed result (no‑change still best).
- References: `runs/m3_replogle_s2_sweep_set_mask0.25_var0.5_cov0.5_s1/metrics.json`, `runs/m3_replogle_s2_sweep_set_mask0.25_var0.5_cov0.5_s2/metrics.json`, `reports/parallel_m3_progress.md`
- Owner: user

## 2025-12-25 - M4 gene harmonization strategy (v1)
- Status: active
- Decision: Use **intersection gene sets** for initial M4 cross‑dataset evaluation:
  - `intersection_sciplex_v1` for drug datasets (sciplex2/3/4)
  - `intersection_genetic_v1` for genetic datasets (replogle + norman)
  - `intersection_all_v1` reserved for cross‑modality experiments
- Rationale: Strict comparability and minimal leakage risk for the first M4 pass.
- Alternatives: foundation set (>=N datasets) or union + missing‑gene tokens.
- Impacts: M4 harmonization uses `configs/harmonization/*` gene lists; cross‑dataset splits created under `runs/m4_splits/`.
- References: `configs/harmonization/README.md`, `reports/m4_harmonization_plan.md`, `runs/m4_splits/cross_dataset_sciplex3_holdout.json`
- Owner: user

## 2025-12-31 - “Usefulness gate” requires beating no-change
- Status: active
- Decision: Treat “usefulness” as a real success requirement: CellJEPA must beat **no-change** somewhere (not only beat ridge).
- Rationale: Beating ridge while losing to no-change is not a defensible “world model helps” claim.
- Alternatives: keep “beat ridge” as sufficient for success.
- Impacts: Updates M3/M4 acceptance framing and reporting; avoid declaring victory while still worse than no-change.
- References: `docs/plan.md`, `docs/prompts.md`
- Owner: user

## 2025-12-31 - M4 cross-dataset is shared-action only + default control z-score calibration
- Status: active
- Decision: Define M4 cross-dataset evaluation as **shared-action only** (do not pool unseen-action / `<UNK>` cases into the main cross-dataset results). Enable **control-based embedding calibration (z-score on controls)** by default for M4-v2 evaluation, with an ablation to disable.
- Rationale: Initial M4 splits have near-zero perturbation overlap, making “cross-dataset” effectively S1+domain shift unless we restrict to shared actions; control z-scoring is a minimal, dependency-free calibration to reduce domain shift.
- Alternatives: evaluate unseen-action via semantic action embeddings; treat calibration as optional only.
- Impacts: Requires overlap diagnostics and filtering in the M4 runner; cross-dataset splits may be adjusted/redefined to ensure meaningful overlap.
- References: `docs/plan.md`, `docs/prompts.md`, `reports/m4_cross_dataset_results.md`, `reports/m4_gene_overlap.md`
- Owner: user

## 2025-12-25 - Module mask sources + gene ID space
- Status: active
- Decision: Use **MSigDB Hallmark**, **Reactome 2023**, and **GO Biological Process** gene sets for module masking, mapped to **gene symbols** (matching `replogle_k562_rpe1_v1.h5ad` var index).
- Rationale: Replogle var index uses gene symbols; these sources cover broad biological modules for JEPA masking.
- Alternatives: Ensembl IDs; single-source modules; data-driven modules.
- Impacts: Requires MSigDB GMT downloads; module mask build uses `scripts/build_module_masks.py`.
- References: `configs/modules/README.md`, `scripts/build_module_masks.py`
- Owner: user

## 2025-12-25 - S1 action embeddings: PCA gene vectors from training split
- Status: active
- Decision: Use split-safe PCA-derived gene embeddings (from training cells) as action vectors for Replogle S1 experiments.
- Rationale: Unseen perturbations are blocked by `<UNK>`; gene embeddings provide a minimal, dependency-free action representation.
- Alternatives: external pathway/target embeddings; one-hot action IDs.
- Impacts: Adds `scripts/build_gene_action_embeddings.py` and uses `--action-embeddings` in world-model runs.
- References: `configs/actions/replogle_gene_pca50_v1.json`, `scripts/build_gene_action_embeddings.py`
- Owner: user

## 2025-12-24 - Primary set-distance metric for M3 is E-distance
- Status: active
- Decision: Use E-distance as the primary set-distance metric for v1 in M3.
- Rationale: Documented as the chosen default in the M3 decision gate.
- Alternatives: not recorded here.
- Impacts: M3 metrics, reports, and comparisons use E-distance.
- References: `docs/prompts.md`
- Owner: user

## 2025-12-24 - M1 ingestion uses scPerturb + Sci-Plex, with additional genetic datasets
- Status: active
- Decision: Use scPerturb v1.4 harmonized h5ad files for Sci-Plex (Sci-Plex2/3/4), plus NormanWeissman2019 (filtered) and Replogle 2022 merged for context OOD; current focus is Sci-Plex3 only for drug perturbations.
- Rationale: Selected as the initial Stage A datasets per the dataset shortlist decision log.
- Alternatives: see dataset shortlist for other candidates.
- Impacts: Ingestion targets, splits, baselines, and reports for M1.
- References: `docs/datasets.md`
- Owner: user

## 2025-12-24 - Multi-modal target for M5 is Perturb-CITE-seq
- Status: active
- Decision: Use Perturb-CITE-seq (RNA + protein) as the M5 multi-modal target dataset.
- Rationale: Selected in the dataset decision log.
- Alternatives: not recorded here.
- Impacts: M5 planning, ingestion, and evaluation.
- References: `docs/datasets.md`
- Owner: user

## 2025-12-25 - Later validation corpus includes Tahoe-100M
- Status: active
- Decision: Use Tahoe-100M as a later validation corpus (not required for M1).
- Rationale: Selected in the dataset decision log for later-stage validation.
- Alternatives: not recorded here.
- Impacts: Future scaling/validation planning.
- References: `docs/datasets.md`
- Owner: user
