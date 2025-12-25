# Parallel M3 Progress Tracker (Replogle)

Update this file as each session completes or advances.

| Track | Branch | Owner/session | Status | Jobs/Run IDs | Best result vs baseline | Notes |
|---|---|---|---|---|---|---|
| A (S2 val‑split α) | exp/s2-valsplit-alpha | local | done | 12834777–12834779 (`runs/m3_world_model_replogle_s2_pairval_ridge_s{0,1,2}`) | **worse than ridge + no‑change** | pair‑level val tuning used (`pair_val_frac=0.2`), alpha=0.3 |
| B (S1 gene‑aware) | exp/s1-gene-action | local | done | 12834780–12834782 (`runs/m3_world_model_replogle_s1_geneembed_v1b_s{0,1,2}`) | **beats no‑change / mean‑shift**, loses to ridge | action embeddings: `configs/actions/replogle_gene_pca50_v1.json` (action_dim=50) |
| C (S2 residual‑ridge) | exp/s2-residual-ridge | local | done | `runs/m3_world_model_replogle_s2_ridge_resid_s{0,1,2}` | **worse than ridge + no‑change** | alpha grid parsed; alpha=0.3; baselines OK |
| D (JEPA sweep) | exp/jepa-sweep | local | done | 12834783–12834791 (`runs/m3_replogle_s2_sweep_set_*`) | **beats ridge with CI** for `mask0.25_var0.5_cov0.5` (seeds 1–2), but worse than no‑change | baseline no‑change remains best by large margin |

## Shared context
- Acceptance target: **Replogle S2**
- Module masks: `configs/modules/msigdb_hallmark_reactome_go_bp_v2025.1_symbols.json`
- Recent runs:
  - v2b world-model residual (no_change baseline): `runs/m3_world_model_replogle_s2_module_v2b_*` (no win)
  - v2c world-model residual (mean_shift baseline): `runs/m3_world_model_replogle_s2_module_v2c_*` (no win)
  - ridge residual (Track C): `runs/m3_world_model_replogle_s2_ridge_resid_*` (no win)
  - pair-val ridge (Track A): `runs/m3_world_model_replogle_s2_pairval_ridge_*` (no win)
  - gene‑embed S1 (Track B): `runs/m3_world_model_replogle_s1_geneembed_v1b_*` (beats no-change, loses to ridge)
  - sweep set predictor (Track D): `runs/m3_replogle_s2_sweep_set_*`

