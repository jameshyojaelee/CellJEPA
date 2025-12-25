# CellJEPA

CellJEPA is a **cell-centric world-modeling** project for single-cell omics.

What “CellJEPA” implies:
- a cell-centric “world model” where a cell’s latent state is the core object
- JEPA as a general state representation learner (not limited to a single task)
- perturbations as one instantiation of state transitions (action → next state)
- perturbation prediction as the primary evaluation regime in v1 (not the only conceivable one)

In v1, we focus on perturbation datasets because they provide clean action labels and rigorous OOD evaluation protocols.

Start here:
- Project plan: `docs/plan.md`
- Critique of the original draft plan: `docs/plan_critique.md`
- HPC/Slurm notes (how we run long jobs): `docs/HPC.md`

Repo layout (high-level):
- `src/celljepa/`: library code
- `scripts/`: runnable entry points (train/eval/report)
- `configs/`: experiment + dataset configs
- `docs/`: design docs, prompts, reports
- `data/`: local data cache (gitignored)
- `runs/`: experiment outputs (gitignored)
