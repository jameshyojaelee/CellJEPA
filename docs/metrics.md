# CellJEPA — Metrics (v1)

This document defines the metrics used in CellJEPA so evaluation is consistent across datasets and models.

Terminology note: metrics are described for perturbation transitions (v1), but are intended to generalize to any state-transition regime in embedding space.

## 1) Unit of Evaluation

Primary unit: **(context_id, perturbation_id)** condition pairs.

We recommend aggregating metrics:
- compute per condition pair first,
- then average across condition pairs (optionally with confidence intervals via bootstrap over condition pairs).

## 2) Prototype Metrics (debug + baseline)

Given control prototype `μ_control` and observed perturbed prototype `μ_perturbed`:
- **Cosine distance** between predicted and observed prototypes
- **MSE/L2** between predicted and observed prototypes

Prototype definition must be fixed (mean or robust mean) and documented.

## 3) Retrieval Metrics (embedding-native)

Goal: predicted embeddings should retrieve the correct perturbation condition.

Examples:
- **kNN accuracy**: among nearest neighbors in embedding space, is the majority label the correct perturbation?
- **MRR (mean reciprocal rank)** for retrieving the correct condition.

## 4) Set Metrics (core)

We compare predicted embedding sets vs observed perturbed sets.

Primary v1 recommendation: **energy distance (E-distance)** computed from samples in embedding space.

Why: it’s a true metric on distributions under mild conditions and aligns naturally with set-valued single-cell readouts.

Implementation detail:
- equalize set sizes by sampling `n` cells from each set (cap `n` for compute),
- average over multiple resamples for stability.

Alternative (acceptable if justified): **MMD** with a fixed kernel.

## 5) Confidence Intervals

Default: bootstrap over condition pairs:
- sample condition pairs with replacement,
- recompute aggregate metric,
- report mean ± CI (e.g., 95%).

## 6) Reporting Requirements

Every report should include:
- dataset stats (cells, genes, number of conditions),
- split definition (what is held out),
- baseline table,
- main results table,
- ablations table,
- compute summary (time, GPU, params).
