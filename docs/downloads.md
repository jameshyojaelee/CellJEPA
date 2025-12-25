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
