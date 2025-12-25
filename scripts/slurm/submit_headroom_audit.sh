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
JOB_NAME=${5:-headroom_audit}

PARTITION=${PARTITION:-gpu}
TIME=${TIME:-08:00:00}
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
  --export=DATASET="$DATASET",CHECKPOINT="$CHECKPOINT",SPLIT="$SPLIT",OUT="$OUT",SAMPLE_SIZE="${SAMPLE_SIZE:-128}",RESAMPLES="${RESAMPLES:-10}",BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}",BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-0}",MIN_CELLS_PER_CONDITION="${MIN_CELLS_PER_CONDITION:-30}",MAX_CELLS_PER_GROUP="${MAX_CELLS_PER_GROUP:-5000}",SEED="${SEED:-0}",RIDGE_ALPHAS="${RIDGE_ALPHAS:-0.1,1.0,10.0,100.0}",INCLUDE_ORACLE_RIDGE="${INCLUDE_ORACLE_RIDGE:-}" \
)

if [ -n "$DEPENDENCY" ]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi

sbatch "${SBATCH_ARGS[@]}" scripts/slurm/headroom_audit.sbatch
