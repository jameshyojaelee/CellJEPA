# CellJEPA — Dataset Shortlist (to finalize before modeling)

This file tracks dataset selection for Stage A (core) and later stages.

Decision gate:
- Do not select the first “real dataset” autonomously; confirm with the user before starting Milestone M1.

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

Recommended sources (well-established, widely used):
- **scPerturb (Nature Methods 2023)** provides harmonized `.h5ad` files for many canonical perturbation datasets (CRISPR + drug; RNA + some protein) and is the recommended *first stop* for Stage A/B ingestion.

### Suggested Stage A candidates (real datasets)

| dataset_id (proposed) | perturbation type | contexts (donor/cell line) | modalities | why it’s a strong default |
|---|---|---|---|---|
| `srivatsan_2020_sciplex3` | small molecules (drug) | 3 cell lines (A549, K562, MCF7) | RNA | very widely used drug perturbation dataset; supports context OOD; used in GeneJEPA |
| `replogle_2022_k562_essential` (or `replogle_2022_*`) | genetic (CRISPRi) | K562 (+ RPE1 in some splits) | RNA | canonical Perturb-seq resource; strong gene-perturbation benchmark; widely reused |
| `norman_2019` (NormanWeissman2019) | genetic (CRISPRi; includes combos) | single cell line | RNA | classic perturbation dataset used across many benchmarks; good early gene-perturbation baseline |
| `openproblems_perturbation_prediction_pbmc_2023` | small molecules (drug) | 3 donors (PBMC) | RNA (benchmark targets) | standardized benchmark w/ official metrics + splits; strong “external” evaluation anchor |

Notes:
- Exact dataset IDs in our code will follow whatever naming the chosen downloader uses (e.g., scPerturb/pertpy filenames). The table above is for planning.
- For M1 we pick **one** dataset to validate ingest → splits → baselines → report. After the harness is stable, expand to 2–4 datasets.

## 3.1 Chosen for M1 (user-confirmed)

These are the initial datasets for M1 ingestion + baseline harness:

- **scPerturb v1.4 (Zenodo 13350497)** — harmonized `.h5ad` files (includes Sci-Plex2/3/4).
- **Sci-Plex (Srivatsan 2020)** — drug perturbation across 3 cell lines. We will use the scPerturb-curated Sci-Plex `.h5ad` files for M1; raw source downloads can follow if needed.
- **NormanWeissman2019 (filtered)** — genetic perturbations; used as an additional M1 dataset for baseline coverage.

## 4) Multi-modal Target (Stage M5)

Candidate:
- Perturb-CITE-seq (RNA + protein; Frangieh et al. 2021) — **selected for M5**

## 4.1 Pretraining corpora (optional; not required for M1)

These are relevant for “world model” style pretraining but are not required to get Stage A working:
- CELLxGENE Census (large general scRNA compendium)
- Tahoe-100M (very large perturbation atlas; heavy download/compute) — **selected for later validation**
- Parse “10M PBMC cytokines” (very large cytokine perturbation dataset; license constraints apply)

## 5) Decision Log

- 2025-12-25: created rubric and shortlist table; dataset IDs not yet finalized.
- 2025-12-25: added recommended candidate datasets and noted that M1 dataset selection is a user-confirmed decision gate.
- 2025-12-25: selected scPerturb + Sci-Plex for M1; Tahoe-100M for later validation; Perturb-CITE-seq for M5.
