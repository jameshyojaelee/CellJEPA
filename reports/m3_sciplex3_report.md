# M3 Sciplex3 Report (Auto-generated)

This report summarizes sciplex3-only M3 runs with set/prototype metrics and baselines.

Baselines missing in 5 set-mode runs (older metrics files).

## Set-level results (E-distance)

| run_id | split | seed | ablation | n_eval | skipped | model_edist | no_change | mean_shift | ridge | ridge_alpha | tag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m3_full | S1 |  | ema=None, mask=random:None, reg=(None,None) |  |  |  |  |  |  |  | sciplex3_s1_set |
| m3_full_v3 | S1 |  | ema=None, mask=random:None, reg=(None,None) | 108 | 1 | 0.1907 |  |  |  |  | sciplex3_s1_set |
| m3_sciplex3_set_s1_fast | S1 |  | ema=None, mask=random:None, reg=(None,None) |  |  |  |  |  |  |  | m3_sciplex3_set_s1_fast |
| m3_full | S2 |  | ema=None, mask=random:None, reg=(None,None) |  |  |  |  |  |  |  | sciplex3_s2_set |
| m3_full_v3 | S2 |  | ema=None, mask=random:None, reg=(None,None) | 188 | 0 | 0.3432 |  |  |  |  | sciplex3_s2_set |

## Prototype results

| run_id | split | seed | ablation | n_eval | skipped | model_mse | model_cosine | tag |
|---|---|---|---|---|---|---|---|---|
| m3_full | S1 |  | ema=None, mask=random:None, reg=(None,None) |  |  | nan | nan | sciplex3_s1_proto |
| m3_full_v3 | S1 |  | ema=None, mask=random:None, reg=(None,None) | 108 | 1 | 0.0020 | 0.0022 | sciplex3_s1_proto |
| m3_sciplex3_proto_s1_fast | S1 |  | ema=None, mask=random:None, reg=(None,None) |  |  | 0.0090 | 0.0109 | m3_sciplex3_proto_s1_fast |
| m3_full | S2 |  | ema=None, mask=random:None, reg=(None,None) |  |  | nan | nan | sciplex3_s2_proto |
| m3_full_v3 | S2 |  | ema=None, mask=random:None, reg=(None,None) | 376 | 1 | 0.0022 | 0.0024 | sciplex3_s2_proto |

Notes:
- CIs are bootstrap 95% over condition pairs if available.
- `ablation` reflects JEPA config fields saved alongside the checkpoint when available.