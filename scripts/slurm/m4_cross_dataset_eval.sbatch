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
EFFECT_TOP_FRAC=${EFFECT_TOP_FRAC:-0}
EFFECT_SEED=${EFFECT_SEED:-0}
ALLOW_UNK_ACTIONS=${ALLOW_UNK_ACTIONS:-}
NO_CONTROL_ZSCORE=${NO_CONTROL_ZSCORE:-}
ALPHA_GRID=${ALPHA_GRID:-0:0.25:0.5:0.75:1.0}
ALPHA_VAL_FRAC=${ALPHA_VAL_FRAC:-0.2}
MAX_DELTA_NORM=${MAX_DELTA_NORM:-}

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
  --seed "$SEED" \
  --alpha-grid "$ALPHA_GRID" \
  --alpha-val-frac "$ALPHA_VAL_FRAC" \
  --effect-top-frac "$EFFECT_TOP_FRAC" \
  --effect-seed "$EFFECT_SEED")

if [ -n "$EVAL_SAMPLE_SIZE" ]; then
  CMD+=(--eval-sample-size "$EVAL_SAMPLE_SIZE")
fi

if [ -n "$EVAL_BASELINES" ]; then
  CMD+=(--eval-baselines)
fi

if [ -n "$ALLOW_UNK_ACTIONS" ]; then
  CMD+=(--allow-unk-actions)
fi

if [ -n "$NO_CONTROL_ZSCORE" ]; then
  CMD+=(--no-control-zscore)
fi

if [ -n "$MAX_DELTA_NORM" ]; then
  CMD+=(--max-delta-norm "$MAX_DELTA_NORM")
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
