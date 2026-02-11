#!/usr/bin/env bash
# Submit P4 pretraining jobs for each encoder backend.
#
# Usage:
#   bash scripts/slurm/submit_pretrain_large.sh [encoder] [dataset]
#
# Examples:
#   bash scripts/slurm/submit_pretrain_large.sh transformer
#   bash scripts/slurm/submit_pretrain_large.sh gnn data/processed/norman2019/norman2019_v1.h5ad
#   bash scripts/slurm/submit_pretrain_large.sh all  # submit all 3 backends

set -euo pipefail
cd "$(dirname "$0")/../.."

ENCODER=${1:-transformer}
DATASET=${2:-data/processed/replogle_k562_rpe1/replogle_k562_rpe1_v1.h5ad}
EPOCHS=${3:-50}

submit_one() {
    local enc=$1
    echo "Submitting ${enc} pretraining..."
    sbatch \
        --export=ALL,ENCODER="${enc}",DATASET="${DATASET}",EPOCHS="${EPOCHS}" \
        --job-name="celljepa-p4-${enc}" \
        scripts/slurm/pretrain_large.sbatch
}

if [ "${ENCODER}" = "all" ]; then
    for enc in transformer gnn perceiver; do
        submit_one "${enc}"
    done
else
    submit_one "${ENCODER}"
fi
