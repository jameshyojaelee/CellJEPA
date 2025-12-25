#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <dataset_path> <out_dir> [job_name]" >&2
  exit 1
fi

DATASET=$1
OUT=$2
JOB_NAME=${3:-jepa_pretrain}

PARTITION=${PARTITION:-gpu}
TIME=${TIME:-48:00:00}
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
  --export=DATASET="$DATASET",OUT="$OUT",EPOCHS="${EPOCHS:-10}",BATCH_SIZE="${BATCH_SIZE:-256}",MAX_CELLS="${MAX_CELLS:-10000}",STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}",EMBED_DIM="${EMBED_DIM:-256}",HIDDEN_DIM="${HIDDEN_DIM:-512}",MASK_RATIO="${MASK_RATIO:-0.25}",EMA_DECAY="${EMA_DECAY:-0.99}",SEED="${SEED:-0}" \
  scripts/slurm/jepa_pretrain.sbatch

