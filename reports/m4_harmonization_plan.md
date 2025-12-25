# M4 Harmonization Strategy (v1)

**Chosen strategy:** intersection gene sets.

Rationale:
- Strict comparability across datasets.
- Avoids leakage and re‑training artifacts across differing gene universes.

Gene sets:
- `intersection_sciplex_v1` → `configs/harmonization/genes_intersection_sciplex_v1.txt` (58,347 genes)
- `intersection_genetic_v1` → `configs/harmonization/genes_intersection_genetic_v1.txt` (6,903 genes)
- `intersection_all_v1` → `configs/harmonization/genes_intersection_all_v1.txt` (6,901 genes)

Cross‑dataset splits created:
- Drug holdout: `runs/m4_splits/cross_dataset_sciplex3_holdout.json`
  - train: sciplex2 + sciplex4
  - test: sciplex3
  - gene set: intersection_sciplex_v1
- Genetic holdout: `runs/m4_splits/cross_dataset_replogle_holdout.json`
  - train: norman2019
  - test: replogle
  - gene set: intersection_genetic_v1

Next steps:
1) Harmonize datasets to the selected gene sets (write subset .h5ad to `data/processed/harmonized/...`).
2) Run cross‑dataset training + evaluation on each split.
