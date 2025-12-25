# CellJEPA — Dataset Shortlist (to finalize before modeling)

This file tracks dataset selection for Stage A (core) and later stages.

## 1) Selection Rubric (Stage A)

Score each candidate on:
- perturbation type diversity (genetic vs chemical),
- availability of controls matched by context,
- metadata completeness (perturbation ID, dose, time, donor/cell line),
- size (cells per condition; supports set metrics),
- licensing / access friction,
- cross-dataset compatibility (gene IDs, annotation quality).

## 2) Initial Target

Stage A should start with **one** dataset to validate the full pipeline (ingest → splits → baselines → report).  
Only after the harness is stable should we expand to 2–4 datasets.

## 3) Shortlist (fill in before M1)

| dataset_id | perturbation type | contexts (donor/cell line) | modalities | notes |
|---|---:|---:|---:|---|
| TBD | TBD | TBD | RNA | first ingestion target |
| TBD | TBD | TBD | RNA | second dataset (diverse regime) |
| TBD | TBD | TBD | RNA | optional |
| TBD | TBD | TBD | RNA | optional |

## 4) Multi-modal Target (Stage M5)

Candidate:
- Perturb-CITE-seq (RNA + protein)

## 5) Decision Log

- 2025-12-25: created rubric and shortlist table; dataset IDs not yet finalized.

