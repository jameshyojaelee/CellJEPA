# CellJEPA — Systematic Critique of `docs/initial_plan.md`

Date: 2025-12-25  
Scope: This critique evaluates `docs/initial_plan.md` as a *project execution artifact* (something that should drive implementation, experiments, and reporting), not only as an idea sketch.

## Executive Summary

`docs/initial_plan.md` shows strong research instincts:
- JEPA-first framing focused on predicting *representations* rather than reconstructing noisy counts.
- Early emphasis on **OOD splits** and **deliberately simple baselines** (critical in perturbation prediction).
- Helpful uncertainty tagging (“Verified / Inference / Speculation”) and early leakage awareness.

Main blockers to execution-readiness:
- **No single source of truth:** the doc mixes a plan, a critique, and a “finalized plan,” which makes it unclear what to implement first.
- **Insufficient operational detail:** dataset list, preprocessing schema, split protocol, training configs, and evaluation harness are not concretely specified.
- **Citations are placeholders** (“Nature”, “PMC”, “arXiv”, etc.) rather than traceable references.
- **Scope creep appears early** (diffusion / LLM-JEPA / morphology / spatial), without strict go/no-go gates tied to Stage A success.

The revised plan should become an engineering-ready roadmap with measurable milestones, acceptance criteria, and reproducibility-first repo conventions.

## 1) Document-Level Critique (Structure, Traceability, and “What Do We Do Next?”)

### 1.1 Mixed document intent
The document contains:
1) an “extensive project plan,”  
2) a “systematic critique,”  
3) a “finalized bulletproof project report plan,” and  
4) a “claim check / gaps & next steps.”

This is useful thinking, but it creates ambiguity: *which section is binding?* A plan should present one clear path with dependencies and minimal branching.

**Recommendation:** split into:
- `docs/plan.md`: the canonical execution plan (milestones + deliverables).
- `docs/plan_critique.md`: keep critique separate (this file).
- optional: `docs/decisions.md`: track major decisions and rationale over time.

### 1.2 Placeholder citations are not usable
Tokens like “Nature +1” or “PMC” are not actionable for writing or verification.

**Recommendation:** replace with stable references: title + year + link (or DOI). If a claim is important enough to shape the plan, it’s important enough to cite precisely.

### 1.3 Missing “definition of done”
Stages A/B/C/D exist, but each lacks “done means X.” Without this, it’s hard to prevent endless iteration.

**Recommendation:** per stage define:
- deliverables (code + artifacts),
- primary metric(s) + target outcomes (or at least “must not regress vs baseline”),
- required ablations,
- reproducibility checklist (seeded runs, config snapshot, fixed splits),
- stop/go criteria for the next stage.

## 2) Scientific Framing Critique (Make the Thesis Testable)

### 2.1 Thesis is plausible but not operational
Thesis: “A JEPA-trained embedding space makes perturbation prediction more practical, robust, and benchmarkable in omics than objectives focused on reconstructing noisy measurements.”

This needs translation into measurable claims:
- “practical” → faster training, fewer failure cases, simpler pipeline, better compute/accuracy tradeoff?
- “robust” → which OOD axes (perturbation, donor, cell line, dataset)? how many datasets?
- “benchmarkable” → what standardized harness outputs? how is it packaged?

**Recommendation:** define 2–3 primary claims and bind each to:
- one experiment,
- one primary metric,
- one null baseline,
- one acceptance threshold (even if modest for v1).

### 2.2 Risk: “embedding metrics are moving the goalposts”
The plan already notes this, and it is real: cosine distance improvements can feel unconvincing if they don’t connect to decisions.

**Recommendation:** anchor at least one downstream task:
- perturbation *retrieval* (“does predicted state retrieve the correct condition?”),
- perturbation *ranking* (“does predicted state rank candidates correctly?”),
- signature matching (nearest-neighbor evaluation),
- multi-modal agreement (RNA ↔ protein consistency) for Stage B.

## 3) Data Plan Critique (The Biggest Practical Risk)

### 3.1 Dataset selection is necessary but underspecified
“Start with 2–4 scPerturb datasets spanning regimes” is directionally correct, but the exact choice determines feasibility and credibility:
- gene set alignment needs,
- perturbation label quality,
- covariate confounding (batch/donor/library),
- and which OOD splits are possible.

**Recommendation:** add an explicit rubric + commit to a concrete initial dataset list (IDs/names), plus a fallback list.

### 3.2 Preprocessing is described conceptually, not concretely
Proposed ideas (log1p, residual-like, foundation gene set, missing tokens) need a concrete spec:
- gene identifier standard (Ensembl? HGNC?),
- normalization choice(s) and why,
- HVG selection timing (per dataset vs global),
- how processed data is stored and versioned (e.g., `h5ad` with a stable schema).

**Recommendation:** write a minimal preprocessing contract:
- inputs/outputs schema,
- deterministic steps,
- caching strategy,
- split-safe computation rules (training-fold only).

### 3.3 Leakage risks are broader than OT
OT leakage is correctly flagged, but leakage can happen earlier:
- computing global gene statistics before splitting,
- selecting HVGs on the full dataset,
- batch correction across train+test,
- defining “adversarial masks” using labels across folds.

**Recommendation:** formalize a “no-peeking” protocol: anything potentially label- or distribution-dependent must be computed on training folds only.

## 4) JEPA Model Design Critique (Needs Concretization and Debug Strategy)

### 4.1 Gene-block masking is not a trivial analog of vision masking
Transcriptomes lack spatial locality; expression has heavy tails; dropout/library effects dominate.

**Recommendation:** make masking strategy a first-class design axis with an explicit implementation plan:
- module sources (Hallmark/Reactome/GO/regulons) and versions,
- gene-ID mapping and overlap handling,
- module sampling distribution (size balancing, frequency),
- ablation matrix (random vs module masks as a minimum).

### 4.2 Collapse prevention is mentioned but not operationalized
Variance/cov regularization + EMA teacher are good ideas, but you also need:
- explicit collapse metrics logged each run,
- a short “stability playbook” (what knobs to change first),
- a minimal baseline architecture that is known to train reliably.

**Recommendation:** define:
- collapse indicators (embedding variance; singular values; predictor output variance),
- early stopping rules,
- stabilization levers (LR, EMA decay, reg weights, predictor depth, weight decay).

### 4.3 Backbone choice needs constraints and a decision
The plan lists Perceiver-style vs Transformer, but does not tie this to:
- parameter count targets,
- expected batch sizes,
- training time constraints,
- ease of debugging.

**Recommendation:** pick one backbone for Stage A (lowest engineering risk), and treat the other as a later ablation once the harness is stable.

## 5) Perturbation Prediction Head Critique (Core Value, Must Be Sharply Defined)

### 5.1 Prototype transition is an excellent debugging milestone
But prototypes need a standardized recipe:
- per cell type? per batch? per donor?
- robust mean vs mean?
- how to handle composition changes induced by perturbation?

**Recommendation:** define prototypes and explicitly test sensitivity to:
- cell count,
- cell-type mixing,
- batch/donor confounding.

### 5.2 Set-to-set matching requires an explicit training loop spec
Key missing details:
- how to batch conditions and sample cells,
- whether the predictor outputs a set directly or outputs per-cell predictions with noise,
- how to avoid collapsing to a mean embedding that “wins” on some metrics.

**Recommendation:** define a deterministic baseline set predictor first, then add stochasticity later if needed.

### 5.3 OT pseudo-pairing is high risk for the core path
Even if leakage is avoided, OT adds complexity and interpretability challenges.

**Recommendation:** gate OT to a later milestone and only attempt if set-loss plateaus.

## 6) Benchmarking and Evaluation Critique (Must Be Specified as Code-Level Rules)

### 6.1 Splits must be explicit and reproducible
The plan lists desirable split types (hold-out perturbations, donor/cell line holdout, cross-dataset), but it does not specify:
- which splits define the “main table,”
- how many folds / seeds,
- exact grouping rules (“unseen perturbation” means what, concretely?),
- and how split logic will be enforced to prevent leakage.

**Recommendation:** define:
- a single primary split protocol for Stage A (e.g., *hold-out perturbations* + *hold-out context*),
- a secondary split (e.g., cross-dataset holdout) if feasible,
- and a deterministic split generator committed to version control.

### 6.2 Metrics must map to claims, and aggregation must be defined
The metrics list is directionally good (cosine distance, retrieval, set distances), but it needs:
- a primary metric per split,
- a clear aggregation scheme (per condition then average? weighted by cell count?),
- uncertainty reporting (bootstrap across conditions; confidence intervals).

**Recommendation:** predefine:
- which metric drives model selection,
- which metrics are reported but not optimized,
- and how confidence intervals are computed and logged.

### 6.3 Baseline fairness and tuning budget are missing
“Simple baselines win” is recognized, but the plan doesn’t define how to ensure baselines are competitive:
- comparable tuning effort,
- comparable inputs and preprocessing,
- compute-normalized comparisons where possible.

**Recommendation:** define a baseline protocol:
- fixed number of trials per model,
- standardized early stopping,
- published configs (so results are reproducible and comparable).

### 6.4 Reporting as a deliverable is under-specified
The plan mentions a “fixed report,” but not what it contains.

**Recommendation:** define the report output early:
- a single command produces `report.md` (or `report.html`) and machine-readable `results.json`,
- includes: split definitions, dataset stats, baseline table, main results table, ablations, compute summary.

## 7) Engineering & Reproducibility Critique (Underdeveloped)

### 7.1 “Single command” goal needs decomposition
The “one command to download/train/eval/report” bar is excellent, but the plan doesn’t decompose the components:
- dataset registry,
- preprocessing cache,
- training pipeline,
- evaluation harness,
- report generator.

**Recommendation:** commit to a minimal CLI surface (even internal) and a “golden run” that reproduces a small benchmark end-to-end.

### 7.2 No explicit experiment artifact plan
Without a standard for:
- experiment naming,
- config snapshots,
- checkpoint retention,
- metric logging,
you will lose time and credibility.

**Recommendation:** define:
- a `runs/` layout that stores `config.yaml`, `metrics.json`, and checkpoints,
- deterministic seeding rules,
- a small “run index” file for quick comparison across experiments.

### 7.3 Compute and runtime expectations are not stated
The plan discusses scaling curves conceptually but does not set expectations:
- target GPU-hours for Stage A,
- what constitutes “small/medium/large,”
- and what is feasible in this environment.

**Recommendation:** include a rough compute budget per milestone and a “fast dev mode” (tiny dataset subset + short run).

## 8) Scope & Sequencing Critique (Prevent Research Sprawl)

### 8.1 Optional hybrids/branches appear too early
Diffusion/LLM-JEPA/morphology/spatial are exciting, but they represent distinct research programs.

**Recommendation:** move them into an explicit “Not in v1” / “Stretch” section with strict gating, e.g.:
- only attempt after the Stage A main table is complete **and** the JEPA baseline shows a clear advantage on at least one OOD split.

### 8.2 Stage ordering should be: splits → baselines → JEPA
The current narrative is model-forward. In this problem setting, the safer order is:
1) lock splits and leakage rules,
2) build evaluation harness + linear baselines,
3) implement the simplest JEPA variant,
4) iterate on masking and stability,
5) then add stronger heads (set losses), and only later consider hybrids.

## 9) Risk Register (What Could Kill the Project?)

Top risks not fully handled in the current plan:
- **R1: JEPA collapse or trivial heuristics** (library size/dropout shortcuts).
- **R2: No consistent improvement over linear controls** across OOD splits.
- **R3: Dataset heterogeneity overwhelms the model** (label noise, confounding).
- **R4: Evaluation ambiguity** (embedding metrics not persuasive).
- **R5: Engineering sprawl** (too many datasets, too many models, too many optional branches).

**Recommendation:** map each risk to:
- a diagnostic,
- a mitigation,
- and a fallback (“if R2 occurs, we pivot to X deliverable”).

## 10) Recommended Outline for a Revised Plan

1. **One-page summary** (goal, thesis, deliverables, success criteria).
2. **Definitions & scope** (what is “state,” what is predicted, what is *not* targeted).
3. **Milestones** with acceptance criteria and gates.
4. **Data** (dataset list, schema, preprocessing, split protocol, leakage prevention).
5. **Models** (JEPA encoder spec, masking spec, predictors, baselines).
6. **Evaluation harness** (metrics, reporting, reproducibility).
7. **Engineering** (repo layout, configs, logging, runtime expectations).
8. **Risks & mitigations** (plus fallback pathways).
9. **Stretch goals** (explicitly gated).
