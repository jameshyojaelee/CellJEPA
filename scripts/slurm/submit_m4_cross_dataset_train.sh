#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 5 ]; then
  echo "Usage: $0 <dataset_path> <checkpoint_path> <split_json> <out_dir> <job_name>" >&2
  exit 1
fi

DATASET=$1
CHECKPOINT=$2
SPLIT=$3
OUT=$4
JOB_NAME=$5

PARTITION=${PARTITION:-gpu}
TIME=${TIME:-60:00:00}
CPUS=${CPUS:-8}
MEM=${MEM:-64G}
GPUS=${GPUS:-1}

sbatch \
  --partition="$PARTITION" \
  --time="$TIME" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --gres="gpu:$GPUS" \
  --job-name="$JOB_NAME" \
  --export=DATASET="$DATASET",CHECKPOINT="$CHECKPOINT",SPLIT="$SPLIT",OUT="$OUT",MODE="${MODE:-set}",EPOCHS="${EPOCHS:-10}",BATCH_SIZE="${BATCH_SIZE:-128}",SAMPLE_SIZE="${SAMPLE_SIZE:-128}",EVAL_RESAMPLES="${EVAL_RESAMPLES:-5}",BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}",MIN_CELLS_PER_CONDITION="${MIN_CELLS_PER_CONDITION:-30}" \
  scripts/slurm/m4_cross_dataset_train.sh
