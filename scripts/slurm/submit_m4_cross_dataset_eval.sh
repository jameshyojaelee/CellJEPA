#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <split_json> <checkpoint_path> <out_dir> [job_name]" >&2
  exit 1
fi

SPLIT=$1
CHECKPOINT=$2
OUT=$3
JOB_NAME=${4:-m4_cross_eval}

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
  --export=SPLIT="$SPLIT",CHECKPOINT="$CHECKPOINT",OUT="$OUT",EPOCHS="${EPOCHS:-10}",SAMPLE_SIZE="${SAMPLE_SIZE:-128}",EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-}",EVAL_RESAMPLES="${EVAL_RESAMPLES:-5}",BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}",BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-0}",MIN_CELLS_PER_CONDITION="${MIN_CELLS_PER_CONDITION:-30}",MAX_CELLS_PER_GROUP="${MAX_CELLS_PER_GROUP:-5000}",SEED="${SEED:-0}",EVAL_BASELINES="${EVAL_BASELINES:-}" \
)

if [ -n "$DEPENDENCY" ]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi

sbatch "${SBATCH_ARGS[@]}" scripts/slurm/m4_cross_dataset_eval.sbatch
