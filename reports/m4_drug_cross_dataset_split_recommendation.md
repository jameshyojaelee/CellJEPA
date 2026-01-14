# M4-v2 drug cross-dataset split recommendation (shared-action only)

## Executive summary

Our current drug datasets (Sci‑Plex2/3/4) have **zero shared non-control perturbations** between train and test in the existing cross-dataset holdouts (e.g., Sci‑Plex2/4 → Sci‑Plex3). Under the M4‑v2 decision (“cross-dataset = shared-action only”), this means **there is no valid drug cross-dataset evaluation** with the currently ingested Sci‑Plex datasets.

Therefore, a “valid” M4 drug cross-dataset run requires **adding at least one additional drug dataset** with **meaningful perturbation overlap** with the chosen training dataset (or choosing a different drug dataset pair entirely).

This doc proposes an actionable path to do that without new dependencies:
1) pick candidate drug dataset(s) to ingest,
2) enforce overlap at split-generation time, and
3) run M4‑v2 cross-dataset evaluation (shared-action only, control z-score on by default, safe residual head).

## Current blocker (why Sci‑Plex cross-dataset doesn’t work)

For the ingested scPerturb Sci‑Plex datasets:
- Sci‑Plex2 has 4 non-control drugs: `BMS`, `Dex`, `Nutlin`, `SAHA`.
- Sci‑Plex4 has 7 non-control perturbations (metabolites/inhibitors).
- Sci‑Plex3 has 188 non-control drugs.
- **Overlap (Sci‑Plex2 ∪ Sci‑Plex4) ∩ Sci‑Plex3 = 0** under the current `perturbation_id` strings.

Under shared-action cross-dataset evaluation, these splits will yield `n_after = 0` evaluation pairs and should be treated as invalid.

## Recommendation: how to get a valid *drug* cross-dataset split

### Step 0 — Define what “valid” means (overlap thresholds)

Pick one of:
- `--require-min-action-overlap 50` (absolute threshold), or
- `--require-min-action-overlap-frac 0.2` (fraction of test actions present in train).

Use the stricter constraint you can satisfy.

### Step 1 — Choose candidate drug dataset pair(s) (to ingest)

We need drug datasets that plausibly share the same perturbation library across distinct experimental settings.
Two practical heuristics:
- multiple files from the *same study* but different cell lines (often shared library),
- datasets known to be small-molecule perturbation studies (not CRISPR).

Suggested candidates to try first (scPerturb v1.4, Zenodo 13350497):
- `NadigOConner2024_hepg2.h5ad`
- `NadigOConner2024_jurkat.h5ad`

These are a good first attempt because they are the same study/year with different cell lines and may share a perturbation library. If they turn out to be genetic perturbations (or have low overlap), discard and try a different drug study pair.

### Step 2 — Download candidate raw `.h5ad` files

Use the existing scPerturb manifest and the repo’s Slurm downloader.

Example (download only the Nadig/O’Conner pair):

```bash
rg 'NadigOConner2024_(hepg2|jurkat)\\.h5ad' configs/download/scperturb_v1_4_urls.tsv > /tmp/scperturb_nadig_urls.tsv
scripts/download/submit_downloads.sh /tmp/scperturb_nadig_urls.tsv scperturb_nadig 2
```

This downloads to `data/raw/scperturb/<file>.h5ad` by default.

### Step 3 — Ingest into our processed format

```bash
python3 scripts/ingest_scperturb.py --input data/raw/scperturb/NadigOConner2024_hepg2.h5ad --dataset-id nadig_hepg2
python3 scripts/ingest_scperturb.py --input data/raw/scperturb/NadigOConner2024_jurkat.h5ad --dataset-id nadig_jurkat
```

This writes:
- `data/processed/nadig_hepg2/nadig_hepg2_v1.h5ad`
- `data/processed/nadig_jurkat/nadig_jurkat_v1.h5ad`

### Step 4 — Harmonize to a shared gene set

Create an intersection gene list and harmonize both datasets to it (mirror the existing M4 workflow):
- compute overlap (informational): `scripts/m4_gene_overlap.py`
- write a gene list to `configs/harmonization/genes_intersection_<id>.txt`
- run `scripts/m4_harmonize_dataset.py` for each dataset with that gene list

### Step 5 — Create a cross-dataset split JSON (with enforced action overlap)

Use the updated split generator with overlap checks:

```bash
python3 scripts/m4_make_cross_dataset_splits.py \\
  --train nadig_hepg2 \\
  --test nadig_jurkat \\
  --gene-set-id intersection_nadig_v1 \\
  --gene-set-path configs/harmonization/genes_intersection_nadig_v1.txt \\
  --dataset-paths \\
    nadig_hepg2=data/processed/harmonized/nadig_hepg2_intersection_nadig_v1.h5ad \\
    nadig_jurkat=data/processed/harmonized/nadig_jurkat_intersection_nadig_v1.h5ad \\
  --split-name cross_dataset_nadig_jurkat_holdout_v1 \\
  --out runs/m4_splits/cross_dataset_nadig_jurkat_holdout_v1.json \\
  --require-min-action-overlap 50
```

If this fails, it means the pair does not meet the shared-action requirement and should not be used for M4 drug cross-dataset.

### Step 6 — Run M4-v2 cross-dataset eval

Use the updated `scripts/m4_cross_dataset_eval.py` (shared-action only, control z-score enabled by default, safe residual head) and the updated Slurm wrapper.

Notes:
- By design, `<UNK>`/unseen actions are excluded from the main evaluation unless you explicitly pass `--allow-unk-actions` (diagnostic only; not for main tables).
- The runner logs `action_overlap`, `action_filter`, and `embedding_calibration` into `metrics.json`.

## If no valid drug cross-dataset pair is found

If we cannot identify a drug dataset pair with meaningful shared-action overlap:
- Treat *drug cross-dataset* as **out of scope for M4-v2** (for now).
- Keep drug evaluation focused on **within-dataset** holdout splits (e.g., Sci‑Plex3 S2 unseen cell line).
- Use cross-dataset evaluation primarily for genetic datasets where shared-action overlap can be made non-trivial (or ingest additional genetic datasets with higher overlap).
