#!/usr/bin/env bash
#SBATCH --job-name=jepa_pretrain
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

DATASET=${DATASET:?Set DATASET to processed .h5ad path}
OUT=${OUT:?Set OUT to run output directory}

EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
MAX_CELLS=${MAX_CELLS:-10000}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-}
EMBED_DIM=${EMBED_DIM:-256}
HIDDEN_DIM=${HIDDEN_DIM:-512}
MASK_RATIO=${MASK_RATIO:-0.25}
MASK_TYPE=${MASK_TYPE:-random}
MODULE_MASK_PATH=${MODULE_MASK_PATH:-}
EMA_DECAY=${EMA_DECAY:-0.99}
NO_EMA=${NO_EMA:-}
VARIANCE_TARGET=${VARIANCE_TARGET:-1.0}
VARIANCE_WEIGHT=${VARIANCE_WEIGHT:-1.0}
COVARIANCE_WEIGHT=${COVARIANCE_WEIGHT:-1.0}
NO_REG=${NO_REG:-}
SEED=${SEED:-0}

cd /gpfs/commons/home/jameslee/CellJEPA
export PYTHONUNBUFFERED=1

CMD=(python3 scripts/train_jepa.py \
  --dataset "$DATASET" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --embed-dim "$EMBED_DIM" \
  --hidden-dim "$HIDDEN_DIM" \
  --mask-ratio "$MASK_RATIO" \
  --mask-type "$MASK_TYPE" \
  --ema-decay "$EMA_DECAY" \
  --variance-target "$VARIANCE_TARGET" \
  --variance-weight "$VARIANCE_WEIGHT" \
  --covariance-weight "$COVARIANCE_WEIGHT" \
  --seed "$SEED" \
  --max-cells "$MAX_CELLS")

if [ -n "$MODULE_MASK_PATH" ]; then
  CMD+=(--module-mask-path "$MODULE_MASK_PATH")
fi

if [ -n "$STEPS_PER_EPOCH" ]; then
  CMD+=(--steps-per-epoch "$STEPS_PER_EPOCH")
fi

if [ -n "$NO_EMA" ]; then
  CMD+=(--no-ema)
fi

if [ -n "$NO_REG" ]; then
  CMD+=(--no-reg)
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
