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
  --export=DATASET="$DATASET",CHECKPOINT="$CHECKPOINT",SPLIT="$SPLIT",OUT="$OUT",MODE="${MODE:-prototype}",EPOCHS="${EPOCHS:-10}",BATCH_SIZE="${BATCH_SIZE:-128}",SAMPLE_SIZE="${SAMPLE_SIZE:-128}",EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-}",EVAL_RESAMPLES="${EVAL_RESAMPLES:-5}",BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}",BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-0}",MAX_CELLS_PER_GROUP="${MAX_CELLS_PER_GROUP:-5000}",MIN_CELLS_PER_CONDITION="${MIN_CELLS_PER_CONDITION:-30}",MAX_PAIRS="${MAX_PAIRS:-}",EMBED_DIM="${EMBED_DIM:-256}",HIDDEN_DIM="${HIDDEN_DIM:-512}",SEED="${SEED:-0}",EVAL_BASELINES="${EVAL_BASELINES:-}",RIDGE_ALPHAS="${RIDGE_ALPHAS:-0.1,1.0,10.0,100.0}",RESIDUAL_BASELINE="${RESIDUAL_BASELINE:-none}",RESIDUAL_ALPHA_GRID="${RESIDUAL_ALPHA_GRID:-0,0.25,0.5,0.75,1.0}" \
)

if [ -n "$DEPENDENCY" ]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi

sbatch "${SBATCH_ARGS[@]}" scripts/slurm/transition_train.sbatch
