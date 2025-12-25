#!/usr/bin/env bash
#SBATCH --job-name=headroom_audit
#SBATCH --time=08:00:00
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

SAMPLE_SIZE=${SAMPLE_SIZE:-128}
RESAMPLES=${RESAMPLES:-10}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-2000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-0}
MIN_CELLS_PER_CONDITION=${MIN_CELLS_PER_CONDITION:-30}
MAX_CELLS_PER_GROUP=${MAX_CELLS_PER_GROUP:-5000}
SEED=${SEED:-0}
RIDGE_ALPHAS=${RIDGE_ALPHAS:-0.1,1.0,10.0,100.0}
INCLUDE_ORACLE_RIDGE=${INCLUDE_ORACLE_RIDGE:-}

cd /gpfs/commons/home/jameslee/CellJEPA
export PYTHONUNBUFFERED=1

CMD=(python3 scripts/headroom_audit.py \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --out "$OUT" \
  --sample-size "$SAMPLE_SIZE" \
  --resamples "$RESAMPLES" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --min-cells-per-condition "$MIN_CELLS_PER_CONDITION" \
  --max-cells-per-group "$MAX_CELLS_PER_GROUP" \
  --seed "$SEED" \
  --ridge-alphas "$RIDGE_ALPHAS")

if [ -n "$INCLUDE_ORACLE_RIDGE" ]; then
  CMD+=(--include-oracle-ridge)
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
