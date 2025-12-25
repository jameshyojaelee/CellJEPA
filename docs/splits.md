# CellJEPA — Split Protocols (v1)

This document defines split generation as **executable rules**, not just intentions. Splits must be deterministic, versioned, and designed to prevent leakage.

Terminology note: these splits are defined for perturbation evaluation (v1), but the same group-splitting principles apply to other action/transition regimes.

## 1) Core Principles

- Splits operate on **groups** (perturbations or contexts), not random per-cell splits.
- Train/val/test must be derivable from a **seed** and dataset metadata only.
- Split files are artifacts and should be reused across experiments for comparability.

## 2) Stage A Main-Table Splits

### S1 — Unseen perturbation (condition OOD)

Goal: evaluate generalization to **held-out perturbations**.

- Group key: `perturbation_id`
- Rule: all cells with the same `perturbation_id` are assigned to the same fold.
- Contexts may overlap between train and test (this isolates perturbation OOD).

### S2 — Unseen context (context OOD)

Goal: evaluate generalization to **held-out contexts** (e.g., donors/cell lines).

- Group key: `context_id`
- Rule: all cells with the same `context_id` are assigned to the same fold.
- Perturbations may overlap between train and test (this isolates context OOD).

## 3) Fold Counts and Seeds

Defaults:
- `k_folds = 5` where feasible; if the dataset is small, reduce but never below 3 without marking results “pilot-only.”
- training seeds per fold: `0, 1, 2`

## 4) Split File Format (recommended)

Store split definitions as JSON to avoid ambiguity:

```json
{
  "dataset_id": "…",
  "split_name": "S1_unseen_perturbation",
  "seed": 0,
  "fold": 0,
  "train_groups": ["…"],
  "val_groups": ["…"],
  "test_groups": ["…"]
}
```

Additionally store a per-cell index list (or cell IDs) for each split if the dataset has stable cell IDs; otherwise derive cell indices from group membership deterministically.

## 5) Leakage Prevention Checklist

- HVG selection: compute on training fold only.
- Any normalization/scaling beyond fixed log1p: fit on training fold only.
- Any learned masking/probe model: fit on training fold only.
- Do not use test-fold cells to tune hyperparameters; use validation fold.
