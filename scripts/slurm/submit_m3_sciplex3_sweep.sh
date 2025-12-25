#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 5 ]; then
  echo "Usage: $0 <dataset_path> <checkpoint_path> <split_s1_json> <split_s2_json> <out_root>" >&2
  exit 1
fi

DATASET=$1
CHECKPOINT=$2
SPLIT_S1=$3
SPLIT_S2=$4
OUT_ROOT=$5

SEEDS=(0 1 2)
MODES=(prototype set)
SPLITS=($SPLIT_S1 $SPLIT_S2)
MIN_CELLS_PER_CONDITION=${MIN_CELLS_PER_CONDITION:-30}
RESIDUAL_BASELINE=${RESIDUAL_BASELINE:-ridge}
RESIDUAL_ALPHA_GRID=${RESIDUAL_ALPHA_GRID:-0,0.25,0.5,0.75,1.0}

for split in "${SPLITS[@]}"; do
  split_tag=$(basename "$split" .json)
  for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      out_dir="${OUT_ROOT}/${split_tag}_${mode}_seed${seed}"
      JOB_NAME="m3_${split_tag}_${mode}_s${seed}"
      MODE=$mode SEED=$seed EVAL_BASELINES=1 MIN_CELLS_PER_CONDITION=$MIN_CELLS_PER_CONDITION \
        RESIDUAL_BASELINE=$RESIDUAL_BASELINE RESIDUAL_ALPHA_GRID=$RESIDUAL_ALPHA_GRID \
        scripts/slurm/submit_transition_train.sh "$DATASET" "$CHECKPOINT" "$split" "$out_dir" "$JOB_NAME"
    done
  done
done
