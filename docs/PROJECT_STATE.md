# CellJEPA - Project State

Purpose: maintain a compact, current snapshot of progress, reasoning, and key artifacts so a new session can resume quickly.

Last updated: 2026-01-13

Update rules:
- Update after each milestone completion, major run, or shift in reasoning.
- Keep entries brief and link to concrete artifacts.
- Keep this in sync with `docs/DECISIONS.md` and `docs/plan.md`.

## Current snapshot
- Active phase: **Phase 0–2 (repo hygiene + data correctness + safety unification)** per `docs/plan.md`.
- Completed milestones: M0, M1, M2.
- Accepted milestones: M0, M1, M2, **M3** (ridge win with CIs on Replogle S2; see `docs/DECISIONS.md`).
- Pending milestones: M5.
- Current focus: align execution with the Phase 0–5 roadmap (docs tracked, gene ID policy, safe heads everywhere).
- Key decisions: see `docs/DECISIONS.md`.
- Operational reference: `docs/runbook.md`.
- Next decision gates (confirm with user before acting): sparse JEPA training path and a cross-dataset pair with meaningful action overlap.

## Progress log
- 2025-12-24: M0 complete. Implemented data-contract checks, split generator CLI, and toy dataset generator. Artifacts include `scripts/make_toy_dataset.py`, `scripts/make_splits.py`, and `runs/m0_splits/`.
- 2025-12-24: M1 complete. Ingested chosen datasets, added baseline harness (no-change, mean-shift, ridge in PCA), and generated golden reports (dataset suite is summarized in `docs/runbook.md`).
- 2025-12-24: M2 complete. Implemented JEPA pretraining (online/teacher/predictor), masking, collapse diagnostics, fast-dev run path, and embeddings export with metrics.
- 2025-12-25: M3 implemented but acceptance not met. Added split-safe evaluation upgrades (CIs, baselines in embedding space, resampling, filtering) and ran sciplex3-only sweeps. Current summary: `reports/m3_sciplex3_report.md`.
- 2025-12-25: Headroom audits completed:
  - Sci-Plex3 S2: baseline-saturated (see `runs/m3_headroom_sciplex3_s2/report.md`).
  - Replogle S2: clear oracle headroom (see `runs/m3_headroom_replogle_s2/report.md`).
- 2025-12-25: M3 acceptance target selected: **Replogle S2** (see `docs/DECISIONS.md`).
- 2025-12-25: Module mask sources decided (MSigDB Hallmark + Reactome 2023 + GO BP; gene symbols). Awaiting MSigDB GMT files to build combined module mask (see `configs/modules/README.md`).
- 2025-12-25: Launched split-safe JEPA pretraining (random masks, Replogle S2 train contexts; seeds 0/1/2) and scheduled dependent M3B (prototype/set) + M3C (world model) runs on Replogle S2.
- 2025-12-25: Built module mask file from MSigDB v2025.1 and launched split-safe JEPA pretraining (module masks) + dependent M3B/M3C jobs on Replogle S2 (seeds 0/1/2).
- 2025-12-25: M3B/M3C module-mask runs completed on Replogle S2; set/world-model E-distance is far worse than baselines (no-change remains best). Acceptance not met; see run outputs under `runs/m3_replogle_s2_module_*` and `runs/m3_world_model_replogle_s2_module_*`.
- 2025-12-25: Launched v2 module-mask reruns for Replogle S2 with world-model baseline fixes + residual-shrinkage (seeds 0/1/2; run IDs under `runs/m2_replogle_s2_module_v2_*`, `runs/m3_replogle_s2_module_v2_*`, `runs/m3_world_model_replogle_s2_module_v2_*`).
- 2025-12-25: Re-ran world-model residual sweep with corrected alpha grid parsing (`v2b` runs under `runs/m3_world_model_replogle_s2_module_v2b_*`).
- 2025-12-25: Launched world-model residual sweep using **mean-shift** baseline (`v2c` runs under `runs/m3_world_model_replogle_s2_module_v2c_*`).
- 2025-12-25: Created parallel command and progress docs for multi-session execution: `reports/parallel_m3_commands.md`, `reports/parallel_m3_progress.md`.
- 2025-12-25: Launched parallel tracks A/B/D (pair-val ridge tuning on S2; S1 gene-embedding world model; sweep-derived set predictors). See `reports/parallel_m3_progress.md` for job IDs.
- 2025-12-25: Parallel tracks completed. Results summary: Track D found a JEPA sweep config (`mask0.25_var0.5_cov0.5`, seeds 1–2) that **beats ridge with CI** on Replogle S2, but **no-change baseline remains best**; Track B (S1 gene‑embeds) beats no-change/mean-shift but loses to ridge. See `reports/parallel_m3_progress.md` for details.
- 2025-12-25: M3 accepted based on ridge win (see `docs/DECISIONS.md`); proceeding to M4 despite no‑change remaining strongest baseline.
- 2025-12-25: M4 started with cross‑dataset gene‑overlap analysis (see `reports/m4_gene_overlap.md`).
- 2025-12-25: M4 harmonization strategy selected (intersection gene sets) and cross‑dataset splits created (see `reports/m4_harmonization_plan.md`, `runs/m4_splits/*`).
- 2025-12-25: Harmonized datasets written under `data/processed/harmonized/` and M4 cross‑dataset training jobs launched (Sci‑Plex3 holdout + Replogle holdout).
- 2025-12-25: Replaced M4 cross‑dataset runner with `scripts/m4_cross_dataset_eval.py` and submitted new eval jobs (IDs 12834796, 12834797).
- 2025-12-26: M4 cross‑dataset effect‑filtered (top 20%) evals completed for Sci‑Plex3 and Replogle holdouts; results in `reports/m4_cross_dataset_results.md` and run outputs under `runs/m4_cross_*_holdout_set_effect20/`.
- 2025-12-31: Post‑M4 review: cross‑dataset splits have near‑zero perturbation overlap (cross‑dataset ≈ S1+domain unless restricted). Defined M4‑v2 as **shared‑action cross‑dataset only** with **control z‑score embedding calibration enabled by default** and “usefulness gate” = must beat **no‑change** somewhere; updated `docs/plan.md` + `docs/runbook.md` and recorded decisions in `docs/DECISIONS.md`.
- 2026-01-05: Plan updated to Phase 0–5 roadmap; docs are now tracked in git; baseline-only dependencies approved (see `docs/DECISIONS.md`).

## Key artifacts
- Plans and guardrails: `docs/plan.md`, `AGENTS.md`.
- Decision log: `docs/DECISIONS.md`.
- Runbook (contracts + how-to): `docs/runbook.md`.
- Reports: `reports/m3_sciplex3_report.md`, `reports/m3_summary.md`.
- Split artifacts: `runs/m0_splits/`.

## Run inventory (concrete IDs)
- M0 splits: `runs/m0_splits/` (files: `s1.json`, `s2.json`).
- M1 splits: `runs/m1_splits/` (sciplex2/3/4 S1/S2, replogle S1/S2, norman2019 S1).
- M1 baseline runs (full): `runs/m1_*_baselines_full/` (see metrics table below).
- M1 baseline runs (non-full): `runs/m1_sciplex3_s1_baselines/`, `runs/m1_sciplex3_s2_baselines/`.
- M2 JEPA runs: `runs/m2_*_jepa_*/` (see metrics table below).
- M3 runs: `runs/m3_full/`, `runs/m3_full_v3/`, plus fast/debug runs under `runs/m3_*_fast/`.

## Metrics snapshot (from `runs/` + `reports/m3_summary.md`)

M1 baseline metrics (ridge in PCA; full runs):

| run_id | ridge_mse | ridge_cos | n_train | n_val | n_test | ridge_alpha |
|---|---|---|---|---|---|---|
| m1_norman2019_s1_baselines_full | 19.1787 | 0.0018 | 140 | 48 | 48 | 1.0000 |
| m1_replogle_k562_rpe1_s1_baselines_full | 34.3315 | 0.1117 | 2664 | 900 | 886 | 10.0000 |
| m1_replogle_k562_rpe1_s2_baselines_full | 1256.3314 | 0.4581 | 2393 | 0 | 2057 | 0.1000 |
| m1_sciplex2_s1_baselines_full | 0.0000 | 0.0001 | 2 | 1 | 1 | 1.0000 |
| m1_sciplex3_s1_baselines_full | 4.6041 | 0.5216 | 342 | 114 | 108 | 0.1000 |
| m1_sciplex3_s2_baselines_full | 3.7191 | 0.0845 | 188 | 188 | 188 | 10.0000 |
| m1_sciplex4_s1_baselines_full | 0.3097 | 0.0003 | 8 | 4 | 2 | 1.0000 |
| m1_sciplex4_s2_baselines_full | 1.2381 | 0.0000 | 7 | 0 | 7 | 0.1000 |

M1 baseline metrics (non-full Sci-Plex3 runs):

| run_id | ridge_mse | ridge_cos | n_train | n_val | n_test |
|---|---|---|---|---|---|
| m1_sciplex3_s1_baselines | 10.5004 | 0.8002 | 342 | 114 | 108 |
| m1_sciplex3_s2_baselines | 0.3581 | 0.3204 | 188 | 188 | 188 |

M2 JEPA metrics (last epoch summary from `metrics.json` history):

| run_id | epochs | last_epoch | loss | mse | var | cov |
|---|---|---|---|---|---|---|
| m2_replogle_jepa_full | 10 | 9 | 0.1766 | 0.1407 | 0.0043 | 0.0317 |
| m2_replogle_jepa_test | 1 | 0 | 3.1227 | 0.4244 | 0.4515 | 2.2467 |
| m2_sciplex2_jepa_full | 20 | 19 | 0.1601 | 0.0716 | 0.0157 | 0.0728 |
| m2_sciplex3_jepa_fastdev | 2 | 1 | 0.8508 | 0.0036 | 0.6648 | 0.1824 |
| m2_sciplex3_jepa_full | 10 | 9 | 0.4358 | 0.0970 | 0.0994 | 0.2394 |
| m2_sciplex3_jepa_test | 1 | 0 | 1.2008 | 0.0207 | 0.7225 | 0.4576 |
| m2_sciplex4_jepa_full | 20 | 19 | 0.4536 | 0.3942 | 0.0076 | 0.0518 |

M3 metrics snapshot (from `reports/m3_summary.md`; run_id `m3_full_v3`):

Important: these are **not** the acceptance comparison. The current sciplex3-only baseline-comparison report is:
- `reports/m3_sciplex3_report.md` (includes set-level baselines + CIs)

Set-level (E-distance):

| dataset | split | test_edist | n_train | n_test | skipped_pairs |
|---|---|---|---|---|---|
| replogle | S1 | 1.0607 | 2664 | 886 | 0 |
| replogle | S2 | 3.4897 | 2393 | 2057 | 0 |
| sciplex3 | S1 | 0.1907 | 342 | 109 | 1 |
| sciplex3 | S2 | 0.3432 | 189 | 188 | 0 |

Prototype-level:

| dataset | split | test_mse | test_cos | n_train | n_test |
|---|---|---|---|---|---|
| replogle | S1 | 0.1302 | 0.1373 | 2664 | 886 |
| replogle | S2 | 0.0070 | 0.0040 | 2393 | 2057 |
| sciplex2 | S1 | 0.0075 | 0.0066 | 2 | 1 |
| sciplex3 | S1 | 0.0020 | 0.0022 | 342 | 109 |
| sciplex3 | S2 | 0.0022 | 0.0024 | 189 | 188 |
| sciplex4 | S1 | 0.0016 | 0.0019 | 9 | 2 |
| sciplex4 | S2 | 0.0015 | 0.0018 | 7 | 7 |

Notes:
- Some set-level E-distance entries for sciplex2/sciplex4 are blank in the report; see `reports/m3_summary.md` for full table and NaN handling notes.

## Reasoning notes (concise)
- Goal is a cell-centric world model for state transitions; perturbations are v1 action type, but avoid assuming they are the only action type.
- Prefer config-driven runs and Slurm for experiments; snapshot configs and splits for reproducibility.

## Open questions / upcoming confirmations
- Cross-dataset gene harmonization strategy for M4 (intersection vs foundation set).
- Module mask sources and versions if enabling module masks in M2/M3 follow-ons.
- Any preprocessing beyond libnorm + log1p and its leakage implications.
- Confirm whether to expand beyond Sci-Plex3 for drug perturbations in M4.
