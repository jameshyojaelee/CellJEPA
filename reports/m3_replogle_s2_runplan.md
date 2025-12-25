# M3 Replogle S2 Run Plan (Split-safe + Module Masks)

Generated: 2025-12-25

This plan records the exact jobs launched for M3B/M3C on Replogle S2 using split-safe JEPA pretraining.

## Inputs
- Dataset: `data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad`
- Split: `runs/m1_splits/replogle_k562_rpe1_s2.json`
- Module masks (MSigDB v2025.1): `configs/modules/msigdb_hallmark_reactome_go_bp_v2025.1_symbols.json`

## Jobs
- M2 (split-safe JEPA, module masks) seeds 0/1/2
- M3B (prototype + set) seeds 0/1/2, dependent on respective M2 jobs
- M3C (world model) seeds 0/1/2, dependent on respective M2 jobs

## Notes
- All runs use `min_cells_per_condition=30` and baselines + CIs.
- World model uses residual prediction.
