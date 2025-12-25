# M3 Sciplex3 Report (Auto-generated)

This report summarizes sciplex3-only M3 runs with set/prototype metrics and baselines.

Baselines missing in 5 set-mode runs (older metrics files).

## Set-level results (E-distance)

| run_id | split | seed | ablation | n_eval | skipped | model_edist | residual_baseline | residual_alpha | no_change | mean_shift | ridge | ridge_alpha | tag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m3_full | S1 |  | ema=None, mask=random:None, reg=(None,None) |  |  |  |  |  |  |  |  |  | sciplex3_s1_set |
| m3_full_v3 | S1 |  | ema=None, mask=random:None, reg=(None,None) | 108 | 1 | 0.1907 |  |  |  |  |  |  | sciplex3_s1_set |
| m3_sciplex3_noema_resid_sweep | S1 | 0 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.2834 [0.2736, 0.2929] | ridge | 0.0 | 0.2901 [0.2780, 0.3021] | 0.2791 [0.2672, 0.2906] | 0.2858 [0.2742, 0.2980] | 0.1 | sciplex3_s1_set_seed0 |
| m3_sciplex3_noema_resid_sweep | S1 | 1 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.2942 [0.2827, 0.3060] | ridge | 0.0 | 0.2801 [0.2694, 0.2915] | 0.2905 [0.2788, 0.3023] | 0.2955 [0.2843, 0.3073] | 0.1 | sciplex3_s1_set_seed1 |
| m3_sciplex3_noema_resid_sweep | S1 | 2 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.2854 [0.2738, 0.2968] | ridge | 0.0 | 0.2858 [0.2730, 0.3002] | 0.2833 [0.2725, 0.2939] | 0.2932 [0.2807, 0.3059] | 0.1 | sciplex3_s1_set_seed2 |
| m3_sciplex3_noema_sweep | S1 | 0 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.3122 [0.3006, 0.3245] |  |  | 0.2901 [0.2780, 0.3021] | 0.2791 [0.2672, 0.2906] | 0.2858 [0.2742, 0.2980] | 0.1 | sciplex3_s1_set_seed0 |
| m3_sciplex3_noema_sweep | S1 | 1 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.3286 [0.3147, 0.3427] |  |  | 0.2801 [0.2694, 0.2915] | 0.2905 [0.2788, 0.3023] | 0.2955 [0.2843, 0.3073] | 0.1 | sciplex3_s1_set_seed1 |
| m3_sciplex3_noema_sweep | S1 | 2 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.3413 [0.3299, 0.3531] |  |  | 0.2858 [0.2730, 0.3002] | 0.2833 [0.2725, 0.2939] | 0.2932 [0.2807, 0.3059] | 0.1 | sciplex3_s1_set_seed2 |
| m3_sciplex3_noreg_resid_sweep | S1 | 0 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.0278 [0.0264, 0.0290] | ridge | 0.0 | 0.0288 [0.0275, 0.0303] | 0.0265 [0.0252, 0.0277] | 0.0277 [0.0267, 0.0288] | 0.1 | sciplex3_s1_set_seed0 |
| m3_sciplex3_noreg_resid_sweep | S1 | 1 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.0284 [0.0273, 0.0296] | ridge | 0.0 | 0.0267 [0.0255, 0.0280] | 0.0284 [0.0273, 0.0296] | 0.0291 [0.0278, 0.0304] | 0.1 | sciplex3_s1_set_seed1 |
| m3_sciplex3_noreg_resid_sweep | S1 | 2 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.0273 [0.0264, 0.0282] | ridge | 0.0 | 0.0286 [0.0269, 0.0303] | 0.0278 [0.0263, 0.0293] | 0.0284 [0.0272, 0.0296] | 0.1 | sciplex3_s1_set_seed2 |
| m3_sciplex3_noreg_sweep | S1 | 0 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.2268 [0.2237, 0.2300] |  |  | 0.0288 [0.0275, 0.0303] | 0.0265 [0.0252, 0.0277] | 0.0277 [0.0267, 0.0288] | 0.1 | sciplex3_s1_set_seed0 |
| m3_sciplex3_noreg_sweep | S1 | 1 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.3338 [0.3273, 0.3404] |  |  | 0.0267 [0.0255, 0.0280] | 0.0284 [0.0273, 0.0296] | 0.0291 [0.0278, 0.0304] | 0.1 | sciplex3_s1_set_seed1 |
| m3_sciplex3_noreg_sweep | S1 | 2 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.1557 [0.1530, 0.1584] |  |  | 0.0286 [0.0269, 0.0303] | 0.0278 [0.0263, 0.0293] | 0.0284 [0.0272, 0.0296] | 0.1 | sciplex3_s1_set_seed2 |
| m3_sciplex3_resid_sweep | S1 | 0 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.3338 [0.3238, 0.3439] | ridge | 0.0 | 0.3467 [0.3341, 0.3590] | 0.3346 [0.3230, 0.3467] | 0.3374 [0.3256, 0.3496] | 0.1 | sciplex3_s1_set_seed0 |
| m3_sciplex3_resid_sweep | S1 | 1 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.3458 [0.3332, 0.3585] | ridge | 0.0 | 0.3336 [0.3226, 0.3447] | 0.3413 [0.3298, 0.3522] | 0.3510 [0.3388, 0.3640] | 0.1 | sciplex3_s1_set_seed1 |
| m3_sciplex3_resid_sweep | S1 | 2 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.3413 [0.3290, 0.3525] | ridge | 0.0 | 0.3405 [0.3275, 0.3551] | 0.3401 [0.3282, 0.3523] | 0.3448 [0.3334, 0.3564] | 0.1 | sciplex3_s1_set_seed2 |
| m3_sciplex3_set_s1_fast | S1 |  | ema=None, mask=random:None, reg=(None,None) |  |  |  |  |  |  |  |  |  | m3_sciplex3_set_s1_fast |
| m3_sciplex3_sweep | S1 | 0 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.3739 [0.3601, 0.3879] |  |  | 0.3467 [0.3341, 0.3590] | 0.3346 [0.3230, 0.3467] | 0.3374 [0.3256, 0.3496] | 0.1 | sciplex3_s1_set_seed0 |
| m3_sciplex3_sweep | S1 | 1 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.3738 [0.3606, 0.3868] |  |  | 0.3336 [0.3226, 0.3447] | 0.3413 [0.3298, 0.3522] | 0.3510 [0.3388, 0.3640] | 0.1 | sciplex3_s1_set_seed1 |
| m3_sciplex3_sweep | S1 | 2 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.3706 [0.3595, 0.3826] |  |  | 0.3405 [0.3275, 0.3551] | 0.3401 [0.3282, 0.3523] | 0.3448 [0.3334, 0.3564] | 0.1 | sciplex3_s1_set_seed2 |
| m3_full | S2 |  | ema=None, mask=random:None, reg=(None,None) |  |  |  |  |  |  |  |  |  | sciplex3_s2_set |
| m3_full_v3 | S2 |  | ema=None, mask=random:None, reg=(None,None) | 188 | 0 | 0.3432 |  |  |  |  |  |  | sciplex3_s2_set |
| m3_sciplex3_noema_resid_sweep | S2 | 0 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.3045 [0.2951, 0.3143] | ridge | 0.0 | 0.2791 [0.2709, 0.2879] | 0.3327 [0.3220, 0.3432] | 0.2938 [0.2854, 0.3015] | 0.1 | sciplex3_s2_set_seed0 |
| m3_sciplex3_noema_resid_sweep | S2 | 1 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.2992 [0.2909, 0.3082] | ridge | 0.0 | 0.2769 [0.2693, 0.2849] | 0.3325 [0.3208, 0.3452] | 0.2966 [0.2875, 0.3056] | 0.1 | sciplex3_s2_set_seed1 |
| m3_sciplex3_noema_resid_sweep | S2 | 2 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.3039 [0.2941, 0.3142] | ridge | 0.0 | 0.2843 [0.2766, 0.2921] | 0.3301 [0.3184, 0.3420] | 0.3005 [0.2917, 0.3092] | 0.1 | sciplex3_s2_set_seed2 |
| m3_sciplex3_noema_sweep | S2 | 0 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.4360 [0.4227, 0.4497] |  |  | 0.2791 [0.2709, 0.2879] | 0.3327 [0.3220, 0.3432] | 0.2938 [0.2854, 0.3015] | 0.1 | sciplex3_s2_set_seed0 |
| m3_sciplex3_noema_sweep | S2 | 1 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.3838 [0.3715, 0.3961] |  |  | 0.2769 [0.2693, 0.2849] | 0.3325 [0.3208, 0.3452] | 0.2966 [0.2875, 0.3056] | 0.1 | sciplex3_s2_set_seed1 |
| m3_sciplex3_noema_sweep | S2 | 2 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.4332 [0.4189, 0.4478] |  |  | 0.2843 [0.2766, 0.2921] | 0.3301 [0.3184, 0.3420] | 0.3005 [0.2917, 0.3092] | 0.1 | sciplex3_s2_set_seed2 |
| m3_sciplex3_noreg_resid_sweep | S2 | 0 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0279 [0.0269, 0.0291] | ridge | 0.0 | 0.0270 [0.0262, 0.0279] | 0.0330 [0.0318, 0.0343] | 0.0271 [0.0262, 0.0279] | 0.1 | sciplex3_s2_set_seed0 |
| m3_sciplex3_noreg_resid_sweep | S2 | 1 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0278 [0.0269, 0.0287] | ridge | 0.0 | 0.0270 [0.0262, 0.0280] | 0.0331 [0.0318, 0.0346] | 0.0276 [0.0267, 0.0286] | 0.1 | sciplex3_s2_set_seed1 |
| m3_sciplex3_noreg_resid_sweep | S2 | 2 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0276 [0.0267, 0.0285] | ridge | 0.0 | 0.0280 [0.0271, 0.0290] | 0.0330 [0.0317, 0.0343] | 0.0279 [0.0270, 0.0289] | 0.1 | sciplex3_s2_set_seed2 |
| m3_sciplex3_noreg_sweep | S2 | 0 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0868 [0.0833, 0.0904] |  |  | 0.0270 [0.0262, 0.0279] | 0.0330 [0.0318, 0.0343] | 0.0271 [0.0262, 0.0279] | 0.1 | sciplex3_s2_set_seed0 |
| m3_sciplex3_noreg_sweep | S2 | 1 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0939 [0.0904, 0.0974] |  |  | 0.0270 [0.0262, 0.0280] | 0.0331 [0.0318, 0.0346] | 0.0276 [0.0267, 0.0286] | 0.1 | sciplex3_s2_set_seed1 |
| m3_sciplex3_noreg_sweep | S2 | 2 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0936 [0.0902, 0.0975] |  |  | 0.0280 [0.0271, 0.0290] | 0.0330 [0.0317, 0.0343] | 0.0279 [0.0270, 0.0289] | 0.1 | sciplex3_s2_set_seed2 |
| m3_sciplex3_resid_sweep | S2 | 0 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.3546 [0.3449, 0.3645] | ridge | 0.0 | 0.3301 [0.3205, 0.3405] | 0.3834 [0.3728, 0.3944] | 0.3426 [0.3335, 0.3507] | 0.1 | sciplex3_s2_set_seed0 |
| m3_sciplex3_resid_sweep | S2 | 1 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.3427 [0.3343, 0.3514] | ridge | 0.0 | 0.3342 [0.3265, 0.3421] | 0.3794 [0.3675, 0.3916] | 0.3481 [0.3398, 0.3574] | 0.1 | sciplex3_s2_set_seed1 |
| m3_sciplex3_resid_sweep | S2 | 2 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.3554 [0.3456, 0.3659] | ridge | 0.0 | 0.3400 [0.3310, 0.3487] | 0.3751 [0.3644, 0.3855] | 0.3492 [0.3406, 0.3588] | 0.1 | sciplex3_s2_set_seed2 |
| m3_sciplex3_sweep | S2 | 0 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.4961 [0.4826, 0.5090] |  |  | 0.3301 [0.3205, 0.3405] | 0.3834 [0.3728, 0.3944] | 0.3426 [0.3335, 0.3507] | 0.1 | sciplex3_s2_set_seed0 |
| m3_sciplex3_sweep | S2 | 1 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.4198 [0.4077, 0.4319] |  |  | 0.3342 [0.3265, 0.3421] | 0.3794 [0.3675, 0.3916] | 0.3481 [0.3398, 0.3574] | 0.1 | sciplex3_s2_set_seed1 |
| m3_sciplex3_sweep | S2 | 2 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.4223 [0.4108, 0.4333] |  |  | 0.3400 [0.3310, 0.3487] | 0.3751 [0.3644, 0.3855] | 0.3492 [0.3406, 0.3588] | 0.1 | sciplex3_s2_set_seed2 |

## Prototype results

| run_id | split | seed | ablation | n_eval | skipped | model_mse | model_cosine | tag |
|---|---|---|---|---|---|---|---|---|
| m3_full | S1 |  | ema=None, mask=random:None, reg=(None,None) |  |  | nan | nan | sciplex3_s1_proto |
| m3_full_v3 | S1 |  | ema=None, mask=random:None, reg=(None,None) | 108 | 1 | 0.0020 | 0.0022 | sciplex3_s1_proto |
| m3_sciplex3_noema_resid_sweep | S1 | 0 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0152 [0.0148, 0.0156] | 0.0176 [0.0174, 0.0179] | sciplex3_s1_prototype_seed0 |
| m3_sciplex3_noema_resid_sweep | S1 | 1 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0158 [0.0155, 0.0160] | 0.0186 [0.0183, 0.0189] | sciplex3_s1_prototype_seed1 |
| m3_sciplex3_noema_resid_sweep | S1 | 2 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0135 [0.0134, 0.0137] | 0.0163 [0.0161, 0.0165] | sciplex3_s1_prototype_seed2 |
| m3_sciplex3_noema_sweep | S1 | 0 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0152 [0.0148, 0.0156] | 0.0176 [0.0174, 0.0179] | sciplex3_s1_prototype_seed0 |
| m3_sciplex3_noema_sweep | S1 | 1 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0158 [0.0155, 0.0160] | 0.0186 [0.0183, 0.0189] | sciplex3_s1_prototype_seed1 |
| m3_sciplex3_noema_sweep | S1 | 2 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0135 [0.0134, 0.0137] | 0.0163 [0.0161, 0.0165] | sciplex3_s1_prototype_seed2 |
| m3_sciplex3_noreg_resid_sweep | S1 | 0 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.0061 [0.0061, 0.0061] | 0.4447 [0.4440, 0.4453] | sciplex3_s1_prototype_seed0 |
| m3_sciplex3_noreg_resid_sweep | S1 | 1 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.0103 [0.0103, 0.0103] | 0.3795 [0.3792, 0.3798] | sciplex3_s1_prototype_seed1 |
| m3_sciplex3_noreg_resid_sweep | S1 | 2 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.0069 [0.0069, 0.0069] | 0.3741 [0.3736, 0.3747] | sciplex3_s1_prototype_seed2 |
| m3_sciplex3_noreg_sweep | S1 | 0 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.0061 [0.0061, 0.0061] | 0.4447 [0.4440, 0.4453] | sciplex3_s1_prototype_seed0 |
| m3_sciplex3_noreg_sweep | S1 | 1 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.0103 [0.0103, 0.0103] | 0.3795 [0.3792, 0.3798] | sciplex3_s1_prototype_seed1 |
| m3_sciplex3_noreg_sweep | S1 | 2 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 108 | 0 | 0.0069 [0.0069, 0.0069] | 0.3741 [0.3736, 0.3747] | sciplex3_s1_prototype_seed2 |
| m3_sciplex3_proto_s1_fast | S1 |  | ema=None, mask=random:None, reg=(None,None) |  |  | 0.0090 | 0.0109 | m3_sciplex3_proto_s1_fast |
| m3_sciplex3_resid_sweep | S1 | 0 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0129 [0.0126, 0.0132] | 0.0168 [0.0166, 0.0171] | sciplex3_s1_prototype_seed0 |
| m3_sciplex3_resid_sweep | S1 | 1 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0159 [0.0156, 0.0161] | 0.0215 [0.0212, 0.0218] | sciplex3_s1_prototype_seed1 |
| m3_sciplex3_resid_sweep | S1 | 2 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0125 [0.0123, 0.0128] | 0.0168 [0.0166, 0.0171] | sciplex3_s1_prototype_seed2 |
| m3_sciplex3_sweep | S1 | 0 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0129 [0.0126, 0.0132] | 0.0168 [0.0166, 0.0171] | sciplex3_s1_prototype_seed0 |
| m3_sciplex3_sweep | S1 | 1 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0159 [0.0156, 0.0161] | 0.0215 [0.0212, 0.0218] | sciplex3_s1_prototype_seed1 |
| m3_sciplex3_sweep | S1 | 2 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 108 | 0 | 0.0125 [0.0123, 0.0128] | 0.0168 [0.0166, 0.0171] | sciplex3_s1_prototype_seed2 |
| m3_full | S2 |  | ema=None, mask=random:None, reg=(None,None) |  |  | nan | nan | sciplex3_s2_proto |
| m3_full_v3 | S2 |  | ema=None, mask=random:None, reg=(None,None) | 376 | 1 | 0.0022 | 0.0024 | sciplex3_s2_proto |
| m3_sciplex3_noema_resid_sweep | S2 | 0 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0158 [0.0153, 0.0164] | 0.0135 [0.0131, 0.0138] | sciplex3_s2_prototype_seed0 |
| m3_sciplex3_noema_resid_sweep | S2 | 1 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0158 [0.0152, 0.0165] | 0.0127 [0.0125, 0.0131] | sciplex3_s2_prototype_seed1 |
| m3_sciplex3_noema_resid_sweep | S2 | 2 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0148 [0.0143, 0.0153] | 0.0132 [0.0129, 0.0135] | sciplex3_s2_prototype_seed2 |
| m3_sciplex3_noema_sweep | S2 | 0 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0158 [0.0153, 0.0164] | 0.0135 [0.0131, 0.0138] | sciplex3_s2_prototype_seed0 |
| m3_sciplex3_noema_sweep | S2 | 1 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0158 [0.0152, 0.0165] | 0.0127 [0.0125, 0.0131] | sciplex3_s2_prototype_seed1 |
| m3_sciplex3_noema_sweep | S2 | 2 | ema=0.0, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0148 [0.0143, 0.0153] | 0.0132 [0.0129, 0.0135] | sciplex3_s2_prototype_seed2 |
| m3_sciplex3_noreg_resid_sweep | S2 | 0 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0007 [0.0007, 0.0007] | 0.1359 [0.1324, 0.1397] | sciplex3_s2_prototype_seed0 |
| m3_sciplex3_noreg_resid_sweep | S2 | 1 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0007 [0.0007, 0.0007] | 0.1382 [0.1345, 0.1422] | sciplex3_s2_prototype_seed1 |
| m3_sciplex3_noreg_resid_sweep | S2 | 2 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0007 [0.0007, 0.0007] | 0.1351 [0.1316, 0.1385] | sciplex3_s2_prototype_seed2 |
| m3_sciplex3_noreg_sweep | S2 | 0 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0007 [0.0007, 0.0007] | 0.1359 [0.1324, 0.1397] | sciplex3_s2_prototype_seed0 |
| m3_sciplex3_noreg_sweep | S2 | 1 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0007 [0.0007, 0.0007] | 0.1382 [0.1345, 0.1422] | sciplex3_s2_prototype_seed1 |
| m3_sciplex3_noreg_sweep | S2 | 2 | ema=0.99, mask=random:0.25, reg=(0.0,0.0) | 188 | 0 | 0.0007 [0.0007, 0.0007] | 0.1351 [0.1316, 0.1385] | sciplex3_s2_prototype_seed2 |
| m3_sciplex3_resid_sweep | S2 | 0 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0152 [0.0147, 0.0158] | 0.0145 [0.0141, 0.0149] | sciplex3_s2_prototype_seed0 |
| m3_sciplex3_resid_sweep | S2 | 1 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0162 [0.0156, 0.0169] | 0.0138 [0.0135, 0.0142] | sciplex3_s2_prototype_seed1 |
| m3_sciplex3_resid_sweep | S2 | 2 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0154 [0.0149, 0.0159] | 0.0150 [0.0147, 0.0155] | sciplex3_s2_prototype_seed2 |
| m3_sciplex3_sweep | S2 | 0 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0152 [0.0147, 0.0158] | 0.0145 [0.0141, 0.0149] | sciplex3_s2_prototype_seed0 |
| m3_sciplex3_sweep | S2 | 1 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0162 [0.0156, 0.0169] | 0.0138 [0.0135, 0.0142] | sciplex3_s2_prototype_seed1 |
| m3_sciplex3_sweep | S2 | 2 | ema=0.99, mask=random:0.25, reg=(1.0,1.0) | 188 | 0 | 0.0154 [0.0149, 0.0159] | 0.0150 [0.0147, 0.0155] | sciplex3_s2_prototype_seed2 |

Notes:
- CIs are bootstrap 95% over condition pairs if available.
- `ablation` reflects JEPA config fields saved alongside the checkpoint when available.