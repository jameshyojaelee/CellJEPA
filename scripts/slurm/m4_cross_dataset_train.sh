#!/usr/bin/env bash
#SBATCH --job-name=m4_cross_train
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

DATASET=${DATASET:?Set DATASET path}
SPLIT=${SPLIT:?Set SPLIT JSON path}
OUT=${OUT:?Set OUT run dir}
MODE=${MODE:-set}
CHECKPOINT=${CHECKPOINT:?Set JEPA checkpoint}

EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
SAMPLE_SIZE=${SAMPLE_SIZE:-128}
EVAL_RESAMPLES=${EVAL_RESAMPLES:-5}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
MIN_CELLS_PER_CONDITION=${MIN_CELLS_PER_CONDITION:-30}

cd /gpfs/commons/home/jameslee/CellJEPA
export PYTHONUNBUFFERED=1

CMD=(python3 scripts/train_transition.py \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --out "$OUT" \
  --mode "$MODE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --sample-size "$SAMPLE_SIZE" \
  --eval-resamples "$EVAL_RESAMPLES" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --min-cells-per-condition "$MIN_CELLS_PER_CONDITION" \
  --eval-baselines)

echo "Running: ${CMD[*]}"
${CMD[@]}
