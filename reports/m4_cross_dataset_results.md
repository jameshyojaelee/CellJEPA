# M4 cross-dataset effect-filtered results (top 20% effects)

Source runs:
- `runs/m4_cross_sciplex3_holdout_set_effect20/metrics.json`
- `runs/m4_cross_replogle_holdout_set_effect20/metrics.json`

## Sci-Plex3 holdout (train: sciplex2 → test: sciplex3)
- run_id: `m4_cross_sciplex3_holdout_set_effect20` (seed 0)
- effect_filter: top_frac 0.2; n_before 564; n_after 112; mean_effect 0.5275
- test E-distance: 8.3744 (95% CI 8.2300–8.5187); n_eval 112; skipped_pairs 0
- baselines (E-distance, 95% CI):
  - no_change: 0.3690 (0.3477–0.3917); n_eval 112
  - mean_shift: 0.3691 (0.3499–0.3876); n_eval 112
  - ridge (alpha 0.1): 89.7576 (88.7964–90.6559); n_eval 112

## Replogle holdout (train: norman2019 → test: replogle)
- run_id: `m4_cross_replogle_holdout_set_effect20` (seed 0)
- effect_filter: top_frac 0.2; n_before 4450; n_after 890; mean_effect 7.6575
- test E-distance: 79.4347 (95% CI 79.1484–79.7399); n_eval 890; skipped_pairs 0
- baselines (E-distance, 95% CI):
  - no_change: 7.5842 (7.4415–7.7401); n_eval 890
  - mean_shift: 7.5663 (7.4289–7.7201); n_eval 890
  - ridge (alpha 0.1): 198.3227 (198.1033–198.5358); n_eval 890
