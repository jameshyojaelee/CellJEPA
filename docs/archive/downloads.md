# CellJEPA — Download Status

This file records dataset download jobs submitted on HPC so new sessions can resume without duplication.

## Submitted Jobs (2025-12-25)

- **scPerturb v1.4** (`configs/download/scperturb_v1_4_urls.tsv`)
  - Slurm job: `12828115` (array 0–53%10; partition `io`; time 48:00:00)
- **Sci-Plex (subset from scPerturb)** (`configs/download/sciplex_only_urls.tsv`)
  - Slurm job: `12828126` (array 0–2%3; partition `io`; time 48:00:00)
- **Sci-Plex (resubmitted to accelerate Sci-Plex2/4)** (`configs/download/sciplex_only_urls.tsv`)
  - Slurm job: `12833160` (array 0–2%3; partition `io`; time 48:00:00)
- **Tahoe-100M** (`configs/download/tahoe100m_urls.tsv`)
  - Slurm job: `12828129` (array 0–4426%20; partition `io`; time 48:00:00)
- **Perturb-CITE-seq (SCP1064)** (`configs/download/perturb_cite_seq_urls_all.tsv`)
  - Slurm job: `12833215` (array 0–15%10; partition `io`; time 48:00:00; `CURL_INSECURE=1`)

## Notes

- SCP portal downloads require SSL workaround on this cluster (`CURL_INSECURE=1`). The job `12833136` failed because the manifest included a portal `manifest` URL; resubmitted as `12833174` after removing that entry. New auth_code issued; resubmitted as `12833199`, but resume failed due to partial files. Wiped `data/raw/perturb_cite_seq/SCP1064` and resubmitted **all files** as `12833215`.
- Sci-Plex4 corruption fixed by deleting the bad file and resubmitting a single-file download (job `12833198`, now verified by MD5).

## How to monitor

```bash
squeue -u $USER
```

Or view logs under `logs/`.

## JEPA Pretrain Jobs (M2)

- Test: Sci-Plex3 GPU (`jepa_test_sciplex3`) → job `12833266`
- Test: Replogle K562+RPE1 GPU (`jepa_test_replogle`) → job `12833267`
- Full: Sci-Plex3 GPU (`jepa_full_sciplex3`) → job `12833268`
- Full: Replogle K562+RPE1 GPU (`jepa_full_replogle`) → job `12833269`

## Transition Training Jobs (M3, full-scale, v1)

Sci-Plex3:
- S1 prototype: `12833270`
- S1 set: `12833271`
- S2 prototype: `12833272`
- S2 set: `12833273`

Sci-Plex4:
- S1 prototype: `12833274`
- S1 set: `12833275`
- S2 prototype: `12833276`
- S2 set: `12833277`

Sci-Plex2 (S1 only):
- S1 prototype: `12833278`
- S1 set: `12833279`

Replogle K562+RPE1:
- S1 prototype: `12833280`
- S1 set: `12833281`
- S2 prototype: `12833282`
- S2 set: `12833283`

## Transition Training Jobs (M3, full-scale, v2 — higher epochs + larger groups)

Sci-Plex3:
- S1 prototype: `12833284`
- S1 set: `12833285`
- S2 prototype: `12833286`
- S2 set: `12833287`

Sci-Plex4:
- S1 prototype: `12833288`
- S1 set: `12833289`
- S2 prototype: `12833290`
- S2 set: `12833291`

Sci-Plex2 (S1 only):
- S1 prototype: `12833292`
- S1 set: `12833293`

Replogle K562+RPE1:
- S1 prototype: `12833294`
- S1 set: `12833295`
- S2 prototype: `12833296`
- S2 set: `12833297`

## JEPA Pretrain Jobs (Sci-Plex2/4, separate encoders)

- Sci-Plex2 full JEPA: `12833298`
- Sci-Plex4 full JEPA: `12833299`

## Transition Training Jobs (Sci-Plex2/4, v3 with dependencies)

Sci-Plex2 (after `12833298`):
- S1 prototype: `12833300` (afterok:12833298)
- S1 set: `12833301` (afterok:12833298)

Sci-Plex4 (after `12833299`):
- S1 prototype: `12833302` (afterok:12833299)
- S1 set: `12833303` (afterok:12833299)
- S2 prototype: `12833304` (afterok:12833299)
- S2 set: `12833305` (afterok:12833299)

## Transition Training Jobs (M3, v3 — much larger epochs/groups)

Sci-Plex3:
- S1 prototype: `12833308`
- S1 set: `12833309`
- S2 prototype: `12833310`
- S2 set: `12833311`

Sci-Plex4:
- S1 prototype: `12833312`
- S1 set: `12833313`
- S2 prototype: `12833314`
- S2 set: `12833315`

Sci-Plex2 (S1 only):
- S1 prototype: `12833316`
- S1 set: `12833317`

Replogle K562+RPE1:
- S1 prototype: `12833318`
- S1 set: `12833319`
- S2 prototype: `12833320`
- S2 set: `12833321`
