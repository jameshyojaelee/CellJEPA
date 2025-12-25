# Parallel M3 Progress Tracker (Replogle)

Update this file as each session completes or advances.

| Track | Branch | Owner/session | Status | Jobs/Run IDs | Best result vs baseline | Notes |
|---|---|---|---|---|---|---|
| A (S2 val‑split α) | exp/s2-valsplit-alpha | | partial | split file created: `runs/m1_splits/replogle_k562_rpe1_s2_valsplit20.json` | no run outputs found | no `runs/*valsplit*` metrics yet |
| B (S1 gene‑aware) | exp/s1-gene-action | | not started | | | no `configs/actions/*` or S1 world-model runs found |
| C (S2 residual‑ridge) | exp/s2-residual-ridge | | done | `runs/m3_world_model_replogle_s2_ridge_resid_s{0,1,2}` | **worse than ridge + no‑change** | alpha grid parsed; alpha=0.3; baselines OK |
| D (JEPA sweep) | exp/jepa-sweep | | done (M2 only) | `runs/m2_replogle_s2_sweep/*` | no downstream eval yet | 81 runs; top loss configs: `mask0.1_var0.5_cov0.5_s{1,2}` |

## Shared context
- Acceptance target: **Replogle S2**
- Module masks: `configs/modules/msigdb_hallmark_reactome_go_bp_v2025.1_symbols.json`
- Recent runs:
  - v2b world-model residual (no_change baseline): `runs/m3_world_model_replogle_s2_module_v2b_*` (no win)
  - v2c world-model residual (mean_shift baseline): `runs/m3_world_model_replogle_s2_module_v2c_*` (no win)
  - v2b/v2c baselines correct; alpha tuning confirmed

