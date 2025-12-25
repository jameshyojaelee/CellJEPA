#!/usr/bin/env bash
#SBATCH --job-name=transition_train
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

DATASET=${DATASET:?Set DATASET to processed .h5ad path}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to JEPA checkpoint path}
SPLIT=${SPLIT:?Set SPLIT to split JSON path}
OUT=${OUT:?Set OUT to run output directory}

MODE=${MODE:-prototype}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
SAMPLE_SIZE=${SAMPLE_SIZE:-128}
MAX_CELLS_PER_GROUP=${MAX_CELLS_PER_GROUP:-5000}
MAX_PAIRS=${MAX_PAIRS:-}
EMBED_DIM=${EMBED_DIM:-256}
HIDDEN_DIM=${HIDDEN_DIM:-512}

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
  --max-cells-per-group "$MAX_CELLS_PER_GROUP" \
  --embed-dim "$EMBED_DIM" \
  --hidden-dim "$HIDDEN_DIM")

if [ -n "$MAX_PAIRS" ]; then
  CMD+=(--max-pairs "$MAX_PAIRS")
fi

echo "Running: ${CMD[*]}"
${CMD[@]}

