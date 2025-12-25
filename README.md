# CellJEPA

CellJEPA investigates **Joint-Embedding Predictive Architectures (JEPAs)** for **single-cell omics**, with a v1 focus on **perturbation response prediction under out-of-distribution (OOD) shift**.

This repo is intentionally **benchmark-driven**: the goal is to make a credible case for when JEPA-style learning is (or is not) practical and useful in omics.

---

## Why this project matters

Predicting transcriptional responses to perturbations is a core problem in functional genomics and drug discovery.
However, recent benchmarking work has highlighted a hard reality: in **gene perturbation effect prediction**, deep learning methods have **not yet consistently beaten deliberately simple linear baselines** on standard settings (especially once evaluation details are carefully controlled). See: *“Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines”* (Nature Methods, 2025): https://www.nature.com/articles/s41592-025-02772-6

CellJEPA is motivated by that gap:
- not "bigger models",
- but a **different training objective**: predict *latent state* rather than reconstructing noisy observations.

---

## Core idea (JEPA in one paragraph)

A JEPA trains an encoder to produce representations and a predictor to **predict the representation of masked targets from visible context**.
In I-JEPA (vision), the model predicts representations of masked image blocks from a context block: https://arxiv.org/abs/2301.08243

In CellJEPA (omics), we adapt the same principle:
- **Context**: a subset of gene features (or gene modules) from a cell or cell population
- **Target**: a held-out subset/module whose embedding is predicted
- **Loss**: prediction occurs in representation space (embeddings), not in raw count space
- **Stability**: we use teacher-student targets (EMA) and anti-collapse regularization when needed

We treat perturbations as **actions** and learn how actions move cells through latent state space.

---

## Why JEPA (vs. GenAI/LLMs/diffusion/AlphaFold)

CellJEPA is explicitly **not** trying to be a “generative model of single-cell counts” first.
Instead, it treats the transcriptome as a noisy, high-dimensional measurement of an underlying cell state and focuses on learning a stable latent representation that supports downstream prediction under distribution shift.

High-level contrasts:
- **JEPA vs. generative models (LLMs/diffusion):** generative objectives optimize for modeling the observation distribution; JEPA optimizes for **predictive representations** (no requirement to decode/reconstruct every observed detail).
- **JEPA vs. AlphaFold-style prediction:** AlphaFold predicts protein structure with strong supervision and inductive biases for a largely “static” object; CellJEPA targets **cell-state transitions under interventions**, where the state is latent, context-dependent, and evaluated via OOD generalization.

---

## Examples of JEPA in biology (starting points)

- `docs/GeneJepa- A Predictive World Model of the Transcriptome.pdf` (local reference)
- `docs/sc-JEPA.pdf` (local reference)
- GeneJEPA preprint (external; may be access-restricted in some environments): https://www.biorxiv.org/content/10.1101/2025.10.14.682378v1
- sc-JEPA (OpenReview): https://openreview.net/forum?id=MZDkttBUEd

---

## What CellJEPA v1 aims to test

### Primary thesis (testable)
We test whether JEPA-trained embeddings are a better substrate for perturbation prediction in OOD regimes than:
- reconstruction-focused objectives (e.g., masked reconstruction),
- and standard perturbation predictors, when evaluated with strong baselines.

### Practical deliverables
- A JEPA training pipeline for scRNA (and later RNA+protein)
- A perturbation transition model that predicts post-perturbation latent state
- A reproducible benchmarking harness with:
  - strict OOD splits,
  - leakage guardrails,
  - and deliberately simple baselines included by default

---

## Evaluation philosophy

We do not claim to predict "exact gene counts" as the primary objective.
Instead, we evaluate:
- **embedding prediction quality** (e.g., cosine distance, retrieval)
- **set-to-set similarity** between predicted and observed perturbed populations (distributional comparison)
- optional: decoded outputs for DE signature agreement (secondary)

---

## Datasets

v1 focuses on harmonized public perturbation datasets and multi-modal perturbation screens:

- **scPerturb**: a harmonized collection of **44** single-cell perturbation-response datasets with molecular readouts (transcriptomics, proteomics, epigenomics): https://www.nature.com/articles/s41592-023-02144-y
- **Perturb-CITE-seq** (M5 stretch): pooled CRISPR perturbations with multi-modal RNA + protein readouts (Nature Genetics, 2021): https://www.nature.com/articles/s41588-021-00779-1
- See `docs/datasets.md` for the repo’s current chosen v1 datasets and any constraints.

---

## Roadmap

### v1 (core)
1. JEPA pretraining for transcriptomics with explicit masking policies
2. Perturbation prediction in latent space under OOD splits
3. Extensive ablations (masking strategy, teacher/EMA, anti-collapse regularization, backbone choice)
4. Benchmark report: where JEPA wins, where it fails, and why

### Optional branches (later)
- Branch 1: morphology integration (Cell Painting + perturbation signatures)
- Branch 2: spatial perturbation context (predict spatially structured response)

### Potential later validation layers (not in v1 core)
- GWAS: trait-enriched gene module masking and enrichment-based evaluation
- Survival: outcome association as external validation of learned state axes

---

## Why this is exploratory (known risks)

- Omics does not have a natural patch topology like images, making masking policy critical.
- JEPA-style objectives can collapse or learn shortcuts without strong stability constraints.
- Perturbation prediction often requires distribution-level evaluation because cells are not paired pre/post.
- Strong baselines are hard to beat; results must be OOD-first and compute-aware.

We treat negative results as informative, as long as they are well-controlled.

---

## Getting started

See:
- Project plan: `docs/plan.md`
- Plan critique / risks: `docs/plan_critique.md`
- HPC/Slurm notes: `docs/HPC.md`
- Download manifests: `docs/downloads.md`

(Quickstart commands will live here once the training/eval CLI stabilizes.)

---

## Repo layout

- `src/celljepa/`   library code (models, losses, data, eval)
- `scripts/`        runnable entry points (train/eval/report)
- `configs/`        experiment and dataset configs
- `docs/`           design docs and reports
- `data/`           local data cache (gitignored)
- `runs/`           outputs and checkpoints (gitignored)

---

## References (starting points)

- I-JEPA: https://arxiv.org/abs/2301.08243
- sc-JEPA (paper + local): https://openreview.net/forum?id=MZDkttBUEd , `docs/sc-JEPA.pdf`
- GeneJEPA (preprint + local): https://www.biorxiv.org/content/10.1101/2025.10.14.682378v1 , `docs/GeneJepa- A Predictive World Model of the Transcriptome.pdf`
- scPerturb: https://www.nature.com/articles/s41592-023-02144-y
- Perturb-CITE-seq: https://www.nature.com/articles/s41588-021-00779-1
- AlphaFold2 (contrast point): https://www.nature.com/articles/s41586-021-03819-2
- Benchmark caution: https://www.nature.com/articles/s41592-025-02772-6
