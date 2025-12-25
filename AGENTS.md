# CellJEPA — AGENTS.md

This file provides repo-specific working agreements for Codex (and other agentic tooling).

## Project Intent

CellJEPA builds a **cell-centric world model** for single-cell omics:
- JEPA is used as a general **state representation learner** (latent cell state embeddings, not reconstructed counts).
- “Actions” induce state transitions; **perturbations** are the v1 action type and the primary evaluation regime.
- The codebase should avoid assumptions that perturbations are the only transition type (future extensions may include time, differentiation, disease, treatment, etc.).

The canonical execution roadmap lives in `docs/plan.md`.

## Sources of Truth

- **Plan:** `docs/plan.md`
- **Original draft (do not edit for “truth”):** `docs/initial_plan.md`
- **Critique/history:** `docs/plan_critique.md`, `docs/plan_self_critique.md`

If you change the project direction, update `docs/plan.md` first (then implement).

## Repo Structure (where things go)

- `src/celljepa/`: library code (models, data, eval, utilities)
- `scripts/`: runnable entry points (train/eval/report). Keep these thin; logic belongs in `src/`.
- `configs/`: versioned configs for datasets, splits, training, evaluation
- `docs/`: documentation, prompts, design notes
- `data/`: local dataset cache (gitignored; never commit data)
- `runs/`: experiment outputs (gitignored; each run writes config + metrics + report)

## Workflow Rules (keep execution unblocked)

- **Order of operations:** splits + baselines + harness → JEPA encoder → transition predictor (perturbations v1) → multi-dataset → multimodal.
- **Reproducibility:** prefer config-driven runs; save split files; snapshot configs in run outputs.
- **No leakage:** any distribution- or label-dependent preprocessing must be computed on training folds only.
- **Small coherent changes:** avoid drive-by refactors; keep patches scoped to the task.

## Dependencies

- Ask before adding new **runtime** dependencies or new services.
- Prefer standard libraries or existing repo patterns; keep first implementations minimal.

## Verification (when applicable)

- Prefer the closest check to the change:
  - syntax/type sanity: `python3 -m compileall src`
  - if tests exist: `python3 -m pytest`

If verification isn’t possible, state what you would run and why.

## Prompting Assets

- System prompts/templates: `docs/SYSTEM_PROMPTS.md`
- Mega-prompts for executing the plan: `docs/prompts.md`
- HPC/Slurm execution context: `docs/HPC.md`

## HPC / Slurm Defaults

We primarily run experiments via **Slurm** on HPC.

- Prefer `sbatch` for long training/eval jobs.
- Default to **walltime > 24 hours** (recommended: `--time=60:00:00`) to avoid timeouts.
- Ensure every job writes checkpoints + artifacts under `runs/<run_id>/…` and logs under `logs/`.

## References (what AGENTS.md is / how it’s used)

- OpenAI Codex guide for `AGENTS.md`: https://developers.openai.com/codex/guides/agents-md
- Community spec/reference for `AGENTS.md`: https://agents.md/
