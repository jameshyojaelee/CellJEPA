#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <dataset_path> <split_json> <out_root> [job_prefix]" >&2
  exit 1
fi

DATASET=$1
SPLIT=$2
OUT_ROOT=$3
JOB_PREFIX=${4:-m2_replogle_s2_sweep}

MASK_TYPE=${MASK_TYPE:-module}
MODULE_MASK_PATH=${MODULE_MASK_PATH:-configs/modules/msigdb_hallmark_reactome_go_bp_v2025.1_symbols.json}
SPLIT_SCOPE=${SPLIT_SCOPE:-train}

MASK_RATIOS=(0.1 0.25 0.5)
VARIANCE_WEIGHTS=(0.5 1.0 2.0)
COVARIANCE_WEIGHTS=(0.5 1.0 2.0)
SEEDS=(0 1 2)

for mask in "${MASK_RATIOS[@]}"; do
  for var_w in "${VARIANCE_WEIGHTS[@]}"; do
    for cov_w in "${COVARIANCE_WEIGHTS[@]}"; do
      config_tag="mask${mask}_var${var_w}_cov${cov_w}"
      job_tag=${config_tag//./p}
      for seed in "${SEEDS[@]}"; do
        out_dir="${OUT_ROOT}/${config_tag}_s${seed}"
        job_name="${JOB_PREFIX}_${job_tag}_s${seed}"
        job_id=$(MASK_TYPE=$MASK_TYPE MODULE_MASK_PATH=$MODULE_MASK_PATH \
          MASK_RATIO=$mask VARIANCE_WEIGHT=$var_w COVARIANCE_WEIGHT=$cov_w \
          SEED=$seed SPLIT=$SPLIT SPLIT_SCOPE=$SPLIT_SCOPE \
          scripts/slurm/submit_jepa_pretrain.sh "$DATASET" "$out_dir" "$job_name" | awk '{print $4}')
        echo "${config_tag} s${seed} ${job_id}"
      done
    done
  done
done
