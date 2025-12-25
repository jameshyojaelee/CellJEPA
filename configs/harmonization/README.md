# M4 Gene Harmonization

We use **intersection gene sets** for initial M4 cross‑dataset evaluation.

Defined sets:
- `genes_intersection_sciplex_v1.txt` — intersection of sciplex2/3/4 (drug datasets)
- `genes_intersection_genetic_v1.txt` — intersection of replogle + norman (genetic datasets)
- `genes_intersection_all_v1.txt` — intersection across all five datasets

Rationale:
- Intersection avoids leakage and ensures strict comparability.
- It is conservative; may be revised to a foundation set later if needed.

Build script:
```
python3 scripts/m4_build_gene_sets.py \
  --drug data/processed/sciplex2/sciplex2_v1.h5ad \
         data/processed/sciplex3/sciplex3_v1.h5ad \
         data/processed/sciplex4/sciplex4_v1.h5ad \
  --genetic data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad \
          data/processed/norman2019/norman2019_v1.h5ad \
  --out-dir configs/harmonization
```
