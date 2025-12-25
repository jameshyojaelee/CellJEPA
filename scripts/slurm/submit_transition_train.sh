#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: $0 <dataset_path> <checkpoint_path> <split_json> <out_dir> [job_name]" >&2
  exit 1
fi

DATASET=$1
CHECKPOINT=$2
SPLIT=$3
OUT=$4
JOB_NAME=${5:-transition_train}

PARTITION=${PARTITION:-gpu}
TIME=${TIME:-48:00:00}
CPUS=${CPUS:-8}
MEM=${MEM:-64G}
GPUS=${GPUS:-1}
DEPENDENCY=${DEPENDENCY:-}

SBATCH_ARGS=(
  --partition="$PARTITION" \
  --time="$TIME" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --gres="gpu:$GPUS" \
  --job-name="$JOB_NAME" \
  --export=DATASET="$DATASET",CHECKPOINT="$CHECKPOINT",SPLIT="$SPLIT",OUT="$OUT",MODE="${MODE:-prototype}",EPOCHS="${EPOCHS:-10}",BATCH_SIZE="${BATCH_SIZE:-128}",SAMPLE_SIZE="${SAMPLE_SIZE:-128}",MAX_CELLS_PER_GROUP="${MAX_CELLS_PER_GROUP:-5000}",MAX_PAIRS="${MAX_PAIRS:-}",EMBED_DIM="${EMBED_DIM:-256}",HIDDEN_DIM="${HIDDEN_DIM:-512}" \
)

if [ -n "$DEPENDENCY" ]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi

sbatch "${SBATCH_ARGS[@]}" scripts/slurm/transition_train.sbatch
