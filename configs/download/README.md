# Download Manifests

This folder holds TSV manifests for Slurm-based downloads.

Each line in a manifest is:

```
<url>\t<relative_path>\t<optional_checksum>
```

Relative paths are resolved under `data/raw/` by default (see `scripts/download/submit_downloads.sh`).

## Manifests

- `scperturb_v1_4_urls.tsv`: scPerturb Zenodo v1.4 `.h5ad` files (includes Sci-Plex2/3/4).
- `sciplex_only_urls.tsv`: filtered subset from scPerturb (Sci-Plex only).
- `tahoe100m_urls.tsv`: Tahoe-100M files from Hugging Face (large).
- `perturb_cite_seq_urls.tsv`: placeholder for Perturb-CITE-seq (requires Single Cell Portal bulk download links).

## Perturb-CITE-seq download note

The Broad Single Cell Portal requires a logged-in session to generate time-limited bulk download commands.
To populate `perturb_cite_seq_urls.tsv`:

1) Log in at the Single Cell Portal study page (SCP1064).
2) Use the “Bulk download” or “Download all” feature.
3) Extract the direct file URLs from the generated curl command.
4) Paste them into `perturb_cite_seq_urls.tsv` (one URL per line, or URL + relpath).

Once that file is populated, you can submit a Slurm array download job via:

```bash
scripts/download/submit_downloads.sh configs/download/perturb_cite_seq_urls.tsv pcite
```

If you have the `cfg.txt` file (curl config) from the portal, you can convert it to a TSV manifest:

```bash
python3 scripts/download/prepare_scportal_manifest.py --cfg path/to/cfg.txt
```
