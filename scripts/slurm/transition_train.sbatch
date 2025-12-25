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
EVAL_SAMPLE_SIZE=${EVAL_SAMPLE_SIZE:-}
EVAL_RESAMPLES=${EVAL_RESAMPLES:-5}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-0}
MAX_CELLS_PER_GROUP=${MAX_CELLS_PER_GROUP:-5000}
MAX_PAIRS=${MAX_PAIRS:-}
EMBED_DIM=${EMBED_DIM:-256}
HIDDEN_DIM=${HIDDEN_DIM:-512}
SEED=${SEED:-0}
EVAL_BASELINES=${EVAL_BASELINES:-}
RIDGE_ALPHAS=${RIDGE_ALPHAS:-0.1,1.0,10.0,100.0}

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
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --max-cells-per-group "$MAX_CELLS_PER_GROUP" \
  --embed-dim "$EMBED_DIM" \
  --hidden-dim "$HIDDEN_DIM" \
  --seed "$SEED" \
  --ridge-alphas "$RIDGE_ALPHAS")

if [ -n "$EVAL_SAMPLE_SIZE" ]; then
  CMD+=(--eval-sample-size "$EVAL_SAMPLE_SIZE")
fi

if [ -n "$MAX_PAIRS" ]; then
  CMD+=(--max-pairs "$MAX_PAIRS")
fi

if [ -n "$EVAL_BASELINES" ]; then
  CMD+=(--eval-baselines)
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
