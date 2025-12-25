# Parallel M3 Command Plan (Replogle S2)

Purpose: central, shared reference for running multiple Codex sessions in parallel. Each track is designed to run in its own git branch and write outputs to **unique `runs/` prefixes**.

How to use:
1) Create a branch per track (A/B/C/D).
2) Run the prompt in a separate Codex session.
3) Update `reports/parallel_m3_progress.md` with job IDs + results.

## Track A — S2 Val‑Split + Residual α Tuning (ridge)
**Branch:** `exp/s2-valsplit-alpha`

Goal: create a real validation split for S2, tune residual alpha on val to avoid overfitting.

Prompt A:
```
You are working in the CellJEPA repo. Follow AGENTS.md. No new runtime deps.

Goal: Create a Replogle S2 val-split and tune residual alpha on val, then re-run world-model residual.

Tasks:
1) Add `scripts/make_split_valsplit.py`:
   - Inputs: --in, --out, --val-frac, --seed
   - Split train_groups into train/val (keep test_groups unchanged)
   - Update split_name to include "valsplit"
2) Create new split:
   - in: `runs/m1_splits/replogle_k562_rpe1_s2.json`
   - out: `runs/m1_splits/replogle_k562_rpe1_s2_valsplit20.json`
3) Re-run world-model residual only (no JEPA retrain):
   - checkpoints: `runs/m2_replogle_s2_module_v2_s{seed}/checkpoint.pt`
   - baseline: ridge
   - alpha grid: `0;0.05;0.1;0.2;0.3`
   - 3 seeds
   - output: `runs/m3_world_model_replogle_s2_valsplit_ridge_s{seed}`
4) Summarize results (E-dist + CIs vs baseline).

Verification:
- python3 -m compileall scripts
```

Example commands:
```
python3 scripts/make_split_valsplit.py \
  --in runs/m1_splits/replogle_k562_rpe1_s2.json \
  --out runs/m1_splits/replogle_k562_rpe1_s2_valsplit20.json \
  --val-frac 0.2 --seed 0

RESIDUAL_BASELINE=ridge RESIDUAL_ALPHA_GRID="0;0.05;0.1;0.2;0.3" EVAL_BASELINES=1 \
  bash scripts/slurm/submit_world_model_train.sh \
  data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad \
  runs/m2_replogle_s2_module_v2_s0/checkpoint.pt \
  runs/m1_splits/replogle_k562_rpe1_s2_valsplit20.json \
  runs/m3_world_model_replogle_s2_valsplit_ridge_s0 \
  m3_world_model_replogle_s2_valsplit_ridge_s0
```

---

## Track B — S1 Gene‑Aware Action Embeddings (unseen perturbations)
**Branch:** `exp/s1-gene-action`

Goal: make S1 learnable by giving action embeddings for genes instead of `<UNK>`.

Prompt B:
```
You are working in the CellJEPA repo. Follow AGENTS.md. No new runtime deps.

Goal: Build gene-aware action embeddings for Replogle S1 and test world-model with them.

Tasks:
1) Add `scripts/build_gene_action_embeddings.py`:
   - Use training split only (Replogle S1)
   - Compute gene embeddings from training cells (e.g., PCA of gene-by-cell matrix)
   - Save JSON: gene -> vector
   - Output: `configs/actions/replogle_gene_pca50_v1.json`
2) Update world-model:
   - Add `--action-embeddings <json>`
   - If provided, use pretrained embeddings for action vectors
3) Run world-model on Replogle S1:
   - split: `runs/m1_splits/replogle_k562_rpe1_s1.json`
   - baseline: no_change
   - alpha grid: `0;0.05;0.1;0.2;0.3`
   - 3 seeds
   - output: `runs/m3_world_model_replogle_s1_geneembed_s{seed}`
4) Summarize results vs baselines.

Verification:
- python3 -m compileall src scripts
```

Example commands:
```
python3 scripts/build_gene_action_embeddings.py \
  --dataset data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad \
  --split runs/m1_splits/replogle_k562_rpe1_s1.json \
  --out configs/actions/replogle_gene_pca50_v1.json \
  --n-components 50 \
  --seed 0

ACTION_EMBEDDINGS=configs/actions/replogle_gene_pca50_v1.json \
  RESIDUAL_BASELINE=no_change RESIDUAL_ALPHA_GRID="0;0.05;0.1;0.2;0.3" \
  EVAL_BASELINES=1 SEED=0 \
  bash scripts/slurm/submit_world_model_train.sh \
  data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad \
  runs/m2_replogle_s2_module_v2_s0/checkpoint.pt \
  runs/m1_splits/replogle_k562_rpe1_s1.json \
  runs/m3_world_model_replogle_s1_geneembed_s0 \
  m3_world_model_replogle_s1_geneembed_s0
```

---

## Track C — S2 Residual World‑Model with Ridge Baseline
**Branch:** `exp/s2-residual-ridge`

Goal: force the world-model to improve on ridge (strong baseline).

Prompt C:
```
You are working in the CellJEPA repo. Follow AGENTS.md. No new runtime deps.

Goal: Run world-model residual on Replogle S2 using ridge baseline.

Tasks:
1) Re-run world-model residual only:
   - baseline: ridge
   - alpha grid: `0;0.05;0.1;0.2;0.3`
   - 3 seeds
   - output: `runs/m3_world_model_replogle_s2_ridge_resid_s{seed}`
2) Summarize results vs ridge + CIs.
```

Example command:
```
RESIDUAL_BASELINE=ridge RESIDUAL_ALPHA_GRID="0;0.05;0.1;0.2;0.3" EVAL_BASELINES=1 \
  bash scripts/slurm/submit_world_model_train.sh \
  data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad \
  runs/m2_replogle_s2_module_v2_s0/checkpoint.pt \
  runs/m1_splits/replogle_k562_rpe1_s2.json \
  runs/m3_world_model_replogle_s2_ridge_resid_s0 \
  m3_world_model_replogle_s2_ridge_resid_s0
```

---

## Track D — JEPA Hyperparameter Sweep (fast + wide)
**Branch:** `exp/jepa-sweep`

Goal: try to improve representations quickly.

Prompt D:
```
You are working in the CellJEPA repo. Follow AGENTS.md. No new runtime deps.

Goal: Sweep JEPA hyperparameters and evaluate set predictor on Replogle S2.

Tasks:
1) Launch sweep over:
   - mask_ratio: [0.1, 0.25, 0.5]
   - variance_weight: [0.5, 1.0, 2.0]
   - covariance_weight: [0.5, 1.0, 2.0]
   - seeds: 0/1/2
   - split-safe (S2 train contexts only)
   - outputs under `runs/m2_replogle_s2_sweep/<config>_s{seed}`
2) For top 3 configs, run:
   - set predictor on Replogle S2 with baselines + CI
3) Summarize results.
```

---

Example commands:
```
MODULE_MASK_PATH=configs/modules/msigdb_hallmark_reactome_go_bp_v2025.1_symbols.json \
  SPLIT_SCOPE=train MASK_TYPE=module \
  scripts/slurm/submit_jepa_sweep.sh \
  data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad \
  runs/m1_splits/replogle_k562_rpe1_s2.json \
  runs/m2_replogle_s2_sweep \
  m2_replogle_s2_sweep
```

After the sweep finishes, run set predictor for the top configs (example for one config/seed):
```
MODE=set EVAL_BASELINES=1 \
  scripts/slurm/submit_transition_train.sh \
  data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad \
  runs/m2_replogle_s2_sweep/mask0.25_var1.0_cov1.0_s0/checkpoint.pt \
  runs/m1_splits/replogle_k562_rpe1_s2.json \
  runs/m3_transition_replogle_s2_sweep/mask0.25_var1.0_cov1.0_set_s0 \
  m3_replogle_s2_sweep_mask0p25_var1p0_cov1p0_set_s0
```

## Progress logging
Update `reports/parallel_m3_progress.md` with:
- branch name
- session owner
- jobs launched (IDs)
- results summary (best baseline vs model)
- next action
