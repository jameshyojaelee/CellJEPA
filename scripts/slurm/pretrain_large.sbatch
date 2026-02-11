#!/usr/bin/env bash
#SBATCH --job-name=pretrain
#SBATCH --time=88:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#
# Multi-GPU: uncomment and adjust for DDP
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=4
# #SBATCH --gres=gpu:4

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /gpfs/commons/home/jameslee/CellJEPA
mkdir -p logs

# --- Configuration (override via env or submit script) ---
ENCODER=${ENCODER:-transformer}
DATASET=${DATASET:-data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad}
GENE_VOCAB=${GENE_VOCAB:-configs/gene_vocab_human_v1.txt}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-3}
SEED=${SEED:-42}
RUN_ID=${RUN_ID:-p4_pretrain_${ENCODER}_$(date +%Y%m%d_%H%M%S)}

echo "=== P4 Pretraining ==="
echo "  Encoder:   ${ENCODER}"
echo "  Dataset:   ${DATASET}"
echo "  Gene vocab: ${GENE_VOCAB}"
echo "  Epochs:    ${EPOCHS}"
echo "  Batch:     ${BATCH_SIZE}"
echo "  LR:        ${LR}"
echo "  Seed:      ${SEED}"
echo "  Run ID:    ${RUN_ID}"
echo ""

python3 -m compileall src scripts 2>/dev/null

# Single GPU
python3 scripts/pretrain_jepa_v2.py \
    --datasets "${DATASET}" \
    --gene-vocab "${GENE_VOCAB}" \
    --encoder "${ENCODER}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --out "runs/${RUN_ID}/"

# For multi-GPU (uncomment single-GPU above, uncomment below):
# torchrun --nproc_per_node=4 scripts/pretrain_jepa_v2.py \
#     --datasets "${DATASET}" \
#     --gene-vocab "${GENE_VOCAB}" \
#     --encoder "${ENCODER}" \
#     --epochs "${EPOCHS}" \
#     --batch-size "${BATCH_SIZE}" \
#     --lr "${LR}" \
#     --seed "${SEED}" \
#     --out "runs/${RUN_ID}/"

echo "=== Done ==="
