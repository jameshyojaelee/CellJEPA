#!/usr/bin/env bash
#SBATCH --job-name=pretrain
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=210G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /gpfs/commons/home/jameslee/CellJEPA
mkdir -p logs

# --- Configuration (override via env or submit script) ---
ENCODER=${ENCODER:-transformer}
DATASET=${DATASET:-data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad}
GENE_VOCAB=${GENE_VOCAB:-configs/gene_vocab_human_v1.txt}
GENE_GRAPH=${GENE_GRAPH:-configs/graphs/ppi_go_graph_v1.pt}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-3}
SEED=${SEED:-42}
RESUME=${RESUME:-}

# Per-encoder batch sizes per GPU (fit within 44GB GPU VRAM each)
# With 4 GPUs, effective throughput is 4× per step
# grad_accum adjusts so effective batch ≈ 256
if [ "${ENCODER}" = "transformer" ]; then
    BATCH_SIZE=${BATCH_SIZE:-4}
    GRAD_ACCUM=${GRAD_ACCUM:-21}  # 4 * 21 * 3gpus = 252
elif [ "${ENCODER}" = "perceiver" ]; then
    BATCH_SIZE=${BATCH_SIZE:-32}
    GRAD_ACCUM=${GRAD_ACCUM:-3}   # 32 * 3 * 3gpus = 288
elif [ "${ENCODER}" = "gnn" ]; then
    BATCH_SIZE=${BATCH_SIZE:-16}
    GRAD_ACCUM=${GRAD_ACCUM:-5}   # 16 * 5 * 3gpus = 240
else
    BATCH_SIZE=${BATCH_SIZE:-16}
    GRAD_ACCUM=${GRAD_ACCUM:-5}
fi

N_GPUS=3
RUN_ID=${RUN_ID:-p4_pretrain_${ENCODER}_$(date +%Y%m%d_%H%M%S)}

echo "=== P4 Pretraining (Multi-GPU DDP) ==="
echo "  Encoder:    ${ENCODER}"
echo "  Dataset:    ${DATASET}"
echo "  Gene vocab: ${GENE_VOCAB}"
echo "  Epochs:     ${EPOCHS}"
echo "  GPUs:       ${N_GPUS}"
echo "  Batch/GPU:  ${BATCH_SIZE} (grad_accum=${GRAD_ACCUM}, eff=$(( BATCH_SIZE * GRAD_ACCUM * N_GPUS )))"
echo "  LR:         ${LR}"
echo "  Seed:       ${SEED}"
echo "  Run ID:     ${RUN_ID}"
echo ""

# Build optional args
EXTRA_ARGS=""
if [ "${ENCODER}" = "gnn" ]; then
    echo "  Gene graph: ${GENE_GRAPH}"
    EXTRA_ARGS="--gene-graph ${GENE_GRAPH}"
fi
if [ -n "${RESUME}" ]; then
    echo "  Resuming from: ${RESUME}"
    EXTRA_ARGS="${EXTRA_ARGS} --resume ${RESUME}"
fi

torchrun --standalone --nproc_per_node=${N_GPUS} --master_port=0 \
    scripts/pretrain_jepa_v2.py \
    --datasets "${DATASET}" \
    --gene-vocab "${GENE_VOCAB}" \
    --encoder "${ENCODER}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --out "runs/${RUN_ID}/" \
    ${EXTRA_ARGS}

echo "=== Done ==="
