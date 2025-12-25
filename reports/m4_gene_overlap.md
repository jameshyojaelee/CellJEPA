# M4 Gene Overlap Summary

## Dataset gene counts

| dataset | n_genes | index_key |
|---|---|---|
| sciplex2_v1.h5ad | 58347 | gene_symbol |
| sciplex3_v1.h5ad | 110983 | gene_symbol |
| sciplex4_v1.h5ad | 58347 | gene_symbol |
| replogle_k562_rpe1_v1.h5ad | 7226 | gene_name |
| norman2019_v1.h5ad | 33694 | _index |

## Global overlap

- union genes: 122129
- intersection genes: 6901

## Pairwise Jaccard

| dataset | sciplex2_v1.h5ad | sciplex3_v1.h5ad | sciplex4_v1.h5ad | replogle_k562_rpe1_v1.h5ad | norman2019_v1.h5ad |
|---|---|---|---|---|---|
| sciplex2_v1.h5ad | 1.0000 | 0.5257 | 1.0000 | 0.1211 | 0.3272 |
| sciplex3_v1.h5ad | 0.5257 | 1.0000 | 0.5257 | 0.0637 | 0.1860 |
| sciplex4_v1.h5ad | 1.0000 | 0.5257 | 1.0000 | 0.1211 | 0.3272 |
| replogle_k562_rpe1_v1.h5ad | 0.1211 | 0.0637 | 0.1211 | 1.0000 | 0.2029 |
| norman2019_v1.h5ad | 0.3272 | 0.1860 | 0.3272 | 0.2029 | 1.0000 |