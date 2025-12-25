#!/usr/bin/env bash
#SBATCH --job-name=world_model_train
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

EPOCHS=${EPOCHS:-10}
SAMPLE_SIZE=${SAMPLE_SIZE:-128}
EVAL_SAMPLE_SIZE=${EVAL_SAMPLE_SIZE:-}
EVAL_RESAMPLES=${EVAL_RESAMPLES:-5}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-0}
MIN_CELLS_PER_CONDITION=${MIN_CELLS_PER_CONDITION:-30}
MAX_CELLS_PER_GROUP=${MAX_CELLS_PER_GROUP:-5000}
HIDDEN_DIM=${HIDDEN_DIM:-512}
RESIDUAL=${RESIDUAL:-}
RESIDUAL_BASELINE=${RESIDUAL_BASELINE:-none}
RESIDUAL_ALPHA_GRID=${RESIDUAL_ALPHA_GRID:-0,0.25,0.5,0.75,1.0}
SEED=${SEED:-0}
EVAL_BASELINES=${EVAL_BASELINES:-}
RIDGE_ALPHAS=${RIDGE_ALPHAS:-0.1,1.0,10.0,100.0}

cd /gpfs/commons/home/jameslee/CellJEPA
export PYTHONUNBUFFERED=1

CMD=(python3 scripts/train_world_model.py \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --sample-size "$SAMPLE_SIZE" \
  --eval-resamples "$EVAL_RESAMPLES" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --min-cells-per-condition "$MIN_CELLS_PER_CONDITION" \
  --max-cells-per-group "$MAX_CELLS_PER_GROUP" \
  --hidden-dim "$HIDDEN_DIM" \
  --seed "$SEED" \
  --ridge-alphas "$RIDGE_ALPHAS" \
  --residual-baseline "$RESIDUAL_BASELINE" \
  --residual-alpha-grid "$RESIDUAL_ALPHA_GRID")

if [ -n "$EVAL_SAMPLE_SIZE" ]; then
  CMD+=(--eval-sample-size "$EVAL_SAMPLE_SIZE")
fi

if [ -n "$RESIDUAL" ]; then
  CMD+=(--residual)
fi

if [ -n "$EVAL_BASELINES" ]; then
  CMD+=(--eval-baselines)
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
