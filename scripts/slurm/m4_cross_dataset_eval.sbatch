#!/usr/bin/env bash
#SBATCH --job-name=m4_cross_eval
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SPLIT=${SPLIT:?Set SPLIT JSON path}
CHECKPOINT=${CHECKPOINT:?Set JEPA checkpoint}
OUT=${OUT:?Set OUT run dir}

EPOCHS=${EPOCHS:-10}
SAMPLE_SIZE=${SAMPLE_SIZE:-128}
EVAL_SAMPLE_SIZE=${EVAL_SAMPLE_SIZE:-}
EVAL_RESAMPLES=${EVAL_RESAMPLES:-5}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-0}
MIN_CELLS_PER_CONDITION=${MIN_CELLS_PER_CONDITION:-30}
MAX_CELLS_PER_GROUP=${MAX_CELLS_PER_GROUP:-5000}
SEED=${SEED:-0}
EVAL_BASELINES=${EVAL_BASELINES:-}

cd /gpfs/commons/home/jameslee/CellJEPA
export PYTHONUNBUFFERED=1

CMD=(python3 scripts/m4_cross_dataset_eval.py \
  --split "$SPLIT" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --sample-size "$SAMPLE_SIZE" \
  --eval-resamples "$EVAL_RESAMPLES" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --min-cells-per-condition "$MIN_CELLS_PER_CONDITION" \
  --max-cells-per-group "$MAX_CELLS_PER_GROUP" \
  --seed "$SEED")

if [ -n "$EVAL_SAMPLE_SIZE" ]; then
  CMD+=(--eval-sample-size "$EVAL_SAMPLE_SIZE")
fi

if [ -n "$EVAL_BASELINES" ]; then
  CMD+=(--eval-baselines)
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
