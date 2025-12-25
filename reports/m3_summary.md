# M3 Summary (Auto-generated)

This table aggregates all `runs/m3_*/*/metrics.json` and `runs/m3_*/metrics.json` files.

| run_id | dataset | split | mode | n_train | n_test | loss | test_mse | test_cos | test_edist | skipped_pairs | n_eval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m3_full | replogle | S1 | prototype | 2664 | 886 | 0.0618 | 0.1284 | 0.1371 |  |  |  |
| m3_full | replogle | S1 | set | 2664 | 886 | 0.5749 |  |  |  |  |  |
| m3_full | replogle | S2 | prototype | 2393 | 2057 | 0.0471 | 0.0070 | 0.0040 |  |  |  |
| m3_full | replogle | S2 | set | 2393 | 2057 | 0.4249 |  |  |  |  |  |
| m3_full | sciplex2 | S1 | prototype | 2 | 1 | 0.0700 | 0.0075 | 0.0066 |  |  |  |
| m3_full | sciplex2 | S1 | set | 2 | 1 | 2.2060 |  |  |  |  |  |
| m3_full | sciplex3 | S1 | prototype | 342 | 109 | 0.0422 | nan | nan |  |  |  |
| m3_full | sciplex3 | S1 | set | 342 | 109 | 0.2924 |  |  |  |  |  |
| m3_full | sciplex3 | S2 | prototype | 189 | 188 | nan | nan | nan |  |  |  |
| m3_full | sciplex3 | S2 | set | 189 | 188 | 0.3103 |  |  |  |  |  |
| m3_full | sciplex4 | S1 | prototype | 9 | 2 | nan | 0.0016 | 0.0019 |  |  |  |
| m3_full | sciplex4 | S1 | set | 9 | 2 | 1.1979 |  |  |  |  |  |
| m3_full | sciplex4 | S2 | prototype | 7 | 7 | 0.0608 | 0.0015 | 0.0018 |  |  |  |
| m3_full | sciplex4 | S2 | set | 7 | 7 | 1.3370 |  |  |  |  |  |
| m3_full_v3 | replogle | S1 | prototype | 2664 | 886 | 0.0442 | 0.1302 | 0.1373 |  |  |  |
| m3_full_v3 | replogle | S1 | set | 2664 | 886 | 1.4061 |  |  | 1.0607 | 0 | 886 |
| m3_full_v3 | replogle | S2 | prototype | 2393 | 2057 | 0.0311 | 0.0070 | 0.0040 |  |  |  |
| m3_full_v3 | replogle | S2 | set | 2393 | 2057 | 0.6631 |  |  | 3.4897 | 0 | 2057 |
| m3_full_v3 | sciplex2 | S1 | prototype | 2 | 1 | 0.0273 | 0.0075 | 0.0066 |  |  |  |
| m3_full_v3 | sciplex2 | S1 | set | 2 | 1 | 1.1614 |  |  |  |  |  |
| m3_full_v3 | sciplex3 | S1 | prototype | 342 | 109 | 0.3566 | 0.0020 | 0.0022 |  | 1 | 108 |
| m3_full_v3 | sciplex3 | S1 | set | 342 | 109 | 0.7162 |  |  | 0.1907 | 1 | 108 |
| m3_full_v3 | sciplex3 | S2 | prototype | 189 | 188 | 0.3463 | 0.0022 | 0.0024 |  | 1 | 376 |
| m3_full_v3 | sciplex3 | S2 | set | 189 | 188 | 1.0555 |  |  | 0.3432 | 0 | 188 |
| m3_full_v3 | sciplex4 | S1 | prototype | 9 | 2 | 0.3737 | 0.0016 | 0.0019 |  | 0 | 2 |
| m3_full_v3 | sciplex4 | S1 | set | 9 | 2 | 0.6109 |  |  |  |  |  |
| m3_full_v3 | sciplex4 | S2 | prototype | 7 | 7 | 0.0237 | 0.0015 | 0.0018 |  |  |  |
| m3_full_v3 | sciplex4 | S2 | set | 7 | 7 | 0.6806 |  |  |  |  |  |
| m3_replogle_proto_s1_fast | replogle | S1 | prototype | 127 | 43 | 0.3212 | 0.1369 | 0.1506 |  |  |  |
| m3_replogle_proto_s2_fast | replogle | S2 | prototype | 118 | 82 | 0.2658 | 0.0130 | 0.0085 |  |  |  |
| m3_replogle_set_s1_fast | replogle | S1 | set | 65 | 17 | 2.0634 |  |  |  |  |  |
| m3_replogle_set_s2_fast | replogle | S2 | set | 58 | 42 | 1.8131 |  |  |  |  |  |
| m3_sciplex3_proto_s1_fast | sciplex3 | S1 | prototype | 130 | 32 | 0.1859 | 0.0090 | 0.0109 |  |  |  |
| m3_sciplex3_set_s1_fast | sciplex3 | S1 | set | 58 | 18 | 1.8332 |  |  |  |  |  |

Notes:
- `loss` is the training loss.
- `test_mse`/`test_cos` are prototype test metrics.
- `test_edist` is the set model test E-distance.
- `skipped_pairs`/`n_eval` indicate NaN filtering behavior.