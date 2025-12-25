# CellJEPA — HPC / Slurm Execution Notes

CellJEPA is intended to be developed and run primarily on an HPC cluster using **Slurm**. This file documents the local execution environment assumptions and recommended job practices so long-running experiments are robust and reproducible.

## 1) Current Environment (observed on 2025-12-25)

- OS/kernel: Ubuntu Linux (kernel `5.15.x`)
- Filesystem: project lives on **GPFS** (`/gpfs/...`)
- Python: `python3` is available (Python 3.10); `python` may not exist
- Slurm: available (Slurm `24.05.2`; commands like `sbatch`, `srun`, `sinfo`, `squeue`)

Note: GPUs may not be visible on the login node (e.g., `nvidia-smi` can be empty). Request GPUs via Slurm when needed.

## 2) Default Policy (avoid job timeouts)

We assume generous access to CPU/memory/GPU partitions. To avoid lost work:

- Prefer **batch jobs** for anything beyond quick checks.
- Default to a **walltime > 24 hours** for training/evaluation jobs.
  - Recommended default: `--time=48:00:00` (adjust upward if you routinely run longer).
- Always write **checkpoints** and **run artifacts** under `runs/<run_id>/…`.
- Write logs under `logs/` (gitignored).

## 3) Slurm Job Script Template (copy/paste)

Create `logs/` first:

```bash
mkdir -p logs
```

Minimal template (edit `--partition`, resources as needed):

```bash
#!/usr/bin/env bash
#SBATCH --job-name=celljepa
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Optional GPU:
#SBATCH --gres=gpu:1
#SBATCH --partition=<YOUR_GPU_PARTITION>

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /gpfs/commons/home/jameslee/CellJEPA

# Example:
python3 -m compileall src
# python3 scripts/train_jepa.py --dataset ... --split ... --out runs/<run_id>/
```

Submit:

```bash
sbatch path/to/job.sbatch
```

## 4) Interactive Sessions (when debugging on compute)

Use interactive jobs for debugging (still set long walltime if you don’t want interruptions):

```bash
srun --pty --time=48:00:00 --cpus-per-task=8 --mem=32G bash
```

For GPU debugging:

```bash
srun --pty --time=48:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G bash
```

## 5) Sweeps (hyperparams / ablations)

Prefer Slurm job arrays for sweeps:

```bash
#SBATCH --array=0-19
```

Ensure each array task writes to a unique `runs/<run_id>/` directory and snapshots its config.
