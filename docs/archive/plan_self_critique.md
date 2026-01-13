# CellJEPA — Self-Critique of `docs/plan.md` (and Refinements to Make)

Date: 2025-12-25  
Purpose: Independently critique the revised plan (`docs/plan.md`) as if reviewing it for execution risk, then list concrete improvements that should be folded back into the plan.

## What the revised plan gets right

- **Milestones with gates**: M0→M5 provides an execution spine and prevents scope creep.
- **Reproducibility-first ordering**: baselines + harness before JEPA is the right risk posture.
- **Explicit non-goals**: helps avoid turning CellJEPA into “everything everywhere all at once.”
- **Embedding-first evaluation**: consistent with the JEPA thesis and avoids pretending we can predict noisy counts perfectly.

## Where the revised plan is still weak (and why it matters)

### 1) Key early decisions are not pinned down
The plan lists choices (split types, set-distance metric, backbone, preprocessing), but doesn’t select defaults.

Why this matters: without defaults, implementation stalls and experiments become incomparable.

### 2) The split protocols are not yet “code-level” executable
We say “unseen perturbation” / “unseen context,” but do not define the exact grouping rules, fold counts, and what happens when some perturbations are absent in held-out contexts.

Why this matters: leakage and silent evaluation bugs are common in perturbation benchmarks.

### 3) Baseline fairness is still too vague
“Fixed tuning budget” is stated, but not operationalized (number of trials, which knobs, early stopping rules).

Why this matters: the whole project’s credibility depends on avoiding strawman baselines.

### 4) Perturbation metadata encoding is unspecified
We mention `a` (perturbation metadata) but not how we encode:
- single gene vs drug vs combos,
- dose/time,
- unknown perturbations at test time.

Why this matters: this becomes the interface between datasets and models; ambiguity here breaks portability.

### 5) Set-prediction training loop details are missing
We don’t define how predicted and observed sets are matched for set losses:
- equalizing set sizes,
- sampling strategy,
- preventing “mean embedding” collapse.

Why this matters: set objectives are easy to implement incorrectly; without a clear recipe results will be unstable.

### 6) Deliverables could be more explicit (artifacts and file outputs)
We reference reports and metrics, but not the exact artifact schema nor the expected folder structure under `runs/`.

Why this matters: reproducibility is an outcome of disciplined artifact design, not a slogan.

## Concrete refinements to fold back into `docs/plan.md`

1) **Choose defaults now** (even if later ablated):
   - Main-table split(s) for Stage A.
   - Primary set-distance metric.
   - Initial preprocessing transform.
   - Initial backbone (and why).

2) **Write a split-spec as executable rules**:
   - specify group IDs, fold counts, seed handling, and exclusion rules.

3) **Define a baseline tuning protocol**:
   - per-baseline hyperparameters, trial counts, and early stopping policy.

4) **Define perturbation metadata schema + encoding**:
   - a canonical `perturbation_token` (string) and structured fields for dose/time/combos.

5) **Specify the set-loss training recipe**:
   - how to sample cells, how to equalize sizes, how to aggregate.

6) **Specify run artifact layout**:
   - what files exist after every run and what they contain.

Next action: incorporate these refinements into `docs/plan.md` to produce a v2 plan that is closer to “implementable without further clarification.”

