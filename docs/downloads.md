# CellJEPA — Download Status

This file records dataset download jobs submitted on HPC so new sessions can resume without duplication.

## Submitted Jobs (2025-12-25)

- **scPerturb v1.4** (`configs/download/scperturb_v1_4_urls.tsv`)
  - Slurm job: `12828115` (array 0–53%10; partition `io`; time 48:00:00)
- **Sci-Plex (subset from scPerturb)** (`configs/download/sciplex_only_urls.tsv`)
  - Slurm job: `12828126` (array 0–2%3; partition `io`; time 48:00:00)
- **Tahoe-100M** (`configs/download/tahoe100m_urls.tsv`)
  - Slurm job: `12828129` (array 0–4426%20; partition `io`; time 48:00:00)
- **Perturb-CITE-seq (SCP1064)** (`configs/download/perturb_cite_seq_urls.tsv`)
  - Slurm job: `12833136` (array 0–15%10; partition `io`; time 48:00:00; `CURL_INSECURE=1`)

## Notes

- SCP portal downloads require SSL workaround on this cluster (`CURL_INSECURE=1`). The original job `12831654` failed and was cancelled; resubmitted as `12833136`.

## How to monitor

```bash
squeue -u $USER
```

Or view logs under `logs/`.
