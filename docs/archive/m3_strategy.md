# CellJEPA — M3 Recovery Strategy (Fast Path to a Defensible M4)

Date: 2025-12-25  
Status: draft → critiqued → final (see below)

This doc exists because our initial M3 implementation (SciPlex3) did **not** meet the M3 acceptance gate (“beat ≥1 strong baseline on the primary metric with CIs”).

We are continuing to treat negative results as informative, but we need a **reproducible, defensible** path to:
1) determine whether there is *any* headroom to win on SciPlex3 S2, and
2) if not, satisfy M3 on an alternative dataset/split with real headroom, then proceed to M4.

Important constraints:
- **No leakage:** any model selection must be split-safe (S2 = context holdout).
- **No strawman baselines:** comparisons must include competitive baselines and CIs.
- **S2-only is acceptable for M3 acceptance** (user-confirmed; recorded in `docs/DECISIONS.md`).

Current decision:
- **Sci-Plex3 S2 is baseline-saturated** under E-distance (headroom audit).
- **Replogle S2** is the M3 acceptance target (see `runs/m3_headroom_replogle_s2/report.md`).
- **M3 accepted** based on ridge win with CIs (config `mask0.25_var0.5_cov0.5`, seeds 1–2); no-change remains best baseline (documented in `reports/parallel_m3_progress.md`).

---

## Definitions

- **Headroom (operational):** the gap between best split-safe baselines and a reasonable upper bound (oracle-style analysis). If oracles barely improve, the task/metric may be baseline-saturated.
- **Primary metric:** set-level **energy distance / E-distance** in embedding space (v1).
- **Unit of evaluation:** (context_id, perturbation_id) condition pairs; bootstrap CIs over condition pairs (see `docs/metrics.md`).

---

## Draft Plan (extensive)

### Step 0 (≤1 day): SciPlex3 S2 headroom audit (go/no-go)

Goal: decide if SciPlex3 S2 has enough headroom to be a realistic M3 acceptance target **given our current evaluation metric**.

Deliverables:
1) A headroom report `reports/m3_headroom_sciplex3.md` that contains:
   - Split-safe baselines on SciPlex3 S2 (mean ± CI):
     - no-change
     - mean-shift (trained on train contexts)
     - ridge (trained on train contexts)
   - Oracle analyses (explicitly labeled “oracle / leaky”, **not** used for acceptance):
     - oracle per-perturbation shift computed using *test* contexts
     - oracle per-(context, perturbation) shift using *test* pairs (upper bound for “shift-only” models)
     - (optional) oracle linear map fit on test prototypes (upper bound for linear predictors)
   - Pair-count diagnostics (after filtering):
     - number of condition pairs in train/val/test
     - distribution of cells per pair
     - fraction of dropped pairs (e.g., `perturbation_id == nan`, too-few-cells, non-finite embeddings)
2) A decision note in `docs/PROJECT_STATE.md`:
   - “SciPlex3 S2 is baseline-saturated” **or** “SciPlex3 S2 has headroom; proceed with Step 1/2 on SciPlex3”.

Decision rule (suggested):
- If the best oracle only improves baseline E-distance by **< ~5–10%** (and CIs overlap heavily), treat SciPlex3 S2 as baseline-saturated for our current representation/metric.

Branch:
- If **baseline-saturated**, do **not** spend a week on complex modeling here; pivot acceptance to a dataset/split with headroom (Step 0b).
- If **headroom exists**, proceed to Step 1/2 on SciPlex3.

### Step 0b (≤1 day): Identify an “M3 acceptance dataset” with headroom

Goal: find a dataset/split where a JEPA-style model could plausibly beat ridge under strict OOD evaluation.

Principles:
- Do not overfit to a single dataset’s quirks.
- Prefer datasets where perturbation response is known to be context-dependent and nonlinear.

Candidate targets (already ingested / available in this repo state):
- **Replogle (context OOD / S2)**: larger condition pairs; more structure; likely more headroom than SciPlex3.
- **Norman2019**: smaller contexts but may have signal; depends on split definition.

Deliverables:
- A short report `reports/m3_headroom_candidates.md` with baseline performance + pair counts for each candidate and a recommendation of the acceptance target.

### Step 1 (2–4 days): Make representation learning “JEPA-like” enough to matter

Goal: improve representation quality so transitions become more predictable than in a baseline embedding space.

Key change:
- Use **module/pathway masks** as the default masking strategy (random masking remains as ablation).

Deliverables:
1) A versioned module mask file in-repo (no new deps), e.g.:
   - `configs/modules/<source>_<version>_<gene_id_space>.json`
   - Format: list of gene lists or `{module_name: {genes: [...]}}`.
2) JEPA training improvements:
   - split-safe JEPA pretraining for S2 (train contexts only),
   - ablation knobs: teacher EMA on/off, reg on/off, random vs module masking.
3) Updated M2 report(s) for the acceptance dataset/split:
   - include collapse metrics and quick sanity checks.

Risks:
- Module sources/versions are a decision gate; must be user-confirmed.
- SciPlex drug names are not “semantic modules”; the masking benefit may be limited if module mapping is poor.

### Step 2 (3–7 days): Action-conditioned JEPA “world model” (set-to-set)

Goal: implement the *core JEPA strength* for this project: **an action-conditioned latent world model** that predicts the *distribution* of post-perturbation states given a baseline population.

Key shift:
- Replace the per-cell MLP set predictor with a **set-to-set** model that conditions on:
  - the control set (not only the mean),
  - perturbation/action embedding.

Deliverables:
1) `src/celljepa/models/world_model.py` (or equivalent) with:
   - a set encoder (DeepSets or lightweight SetTransformer),
   - an action encoder (tokens + numeric metadata),
   - a conditional predictor that outputs a predicted set of embeddings.
2) Training script `scripts/train_world_model.py`:
   - uses set-level loss (E-distance),
   - logs pair counts, compute, and metrics with CIs.
3) A report `runs/<run_id>/report.md` with:
   - baseline table,
   - main results,
   - ablations (teacher/reg/mask).

### Step 3 (optional / high-leverage): Make action encoding meaningful (S1-ready)

Goal: enable generalization beyond `<UNK>` actions and support unseen-perturbation (S1) later.

Approach (dependency-minimal first):
- Start with richer **perturbation tokenization** from existing metadata:
  - drug name normalization,
  - dose/time bins if present,
  - mechanism-of-action categories if available in the curated dataset.
- Only then consider external mappings (requires downloads and careful versioning):
  - drug targets/pathways,
  - gene interaction graphs for gene perturbations.

Deliverables:
- A versioned action-feature artifact:
  - `configs/actions/<dataset_id>_<version>.json`
- Updated evaluation for S1 once action embeddings exist.

---

## Critique of the Draft Plan (what could go wrong)

1) **Headroom may be genuinely tiny** on multiple datasets under E-distance in embedding space.
   - Mitigation: explicitly treat “no win vs baselines” as a valid outcome and pivot to characterizing *where* JEPA helps (or doesn’t), but then update acceptance gates before M4.

2) **Module masking can become a tarpit** (choosing sources/versions, mapping gene IDs, uneven module sizes).
   - Mitigation: pick a small, frozen module set first (e.g., 50–200 modules), version it, and keep a strict fallback to random masks.

3) **Set-to-set world model engineering could be large** and still fail to beat ridge if the task is mostly linear.
   - Mitigation: implement the smallest set model first (DeepSets), add only one complexity knob at a time.

4) **Risk of leakage** is high when reusing embeddings trained on all contexts.
   - Mitigation: enforce split-safe JEPA pretraining for S2 and ensure reports call out which cells were used in each stage.

5) **“Fast path to M4” could become “skip M3 in disguise.”**
   - Mitigation: define an explicit acceptance dataset for M3 (even if not SciPlex3) and *actually* beat baselines with CIs before starting M4.

---

## Final Plan (refined)

### Phase A — Decide headroom (SciPlex3 S2) and pick the acceptance target
1) Implement and run the SciPlex3 S2 headroom audit.
2) If baseline-saturated:
   - select a larger/headroom dataset for M3 acceptance (default candidate: Replogle S2),
   - document SciPlex3 S2 as baseline-saturated under current metric/embedding (not a failure, a finding).

### Phase B — Improve JEPA representations (module masks + split-safe)
3) Choose and version a module-mask source (decision gate).
4) Train split-safe JEPA (S2 train contexts only) on the acceptance dataset.
5) Re-run transition predictors and compare to baselines with CIs.

### Phase C — If still no win, pivot to world-modeling (set-to-set)
6) Implement the smallest action-conditioned set-to-set world model.
7) Evaluate and ablate; stop when we either:
   - beat at least one strong baseline with CIs on S2, or
   - convincingly show no headroom (documented) and decide whether to revise the M3 gate.

### Phase D — Prepare for M4 and later S1
8) Only after M3 acceptance: proceed to M4 multi-dataset + cross-dataset holdout.
9) Add richer action embeddings to pursue S1 and multi-dataset robustness.

---

## References (high-level pointers)

These are conceptual anchors; we still must validate empirically on our splits.

- I-JEPA (masking + representation prediction): https://arxiv.org/abs/2301.08243
- V-JEPA 2 (action-conditioned world model framing): https://arxiv.org/abs/2506.09985
- Perturbation modeling baselines and inductive biases:
  - CPA: https://link.springer.com/article/10.15252/msb.202211517
  - scGen: https://pubmed.ncbi.nlm.nih.gov/31363220/
  - CellOT: https://www.nature.com/articles/s41592-023-01969-x
  - CINEMA-OT: https://www.nature.com/articles/s41592-023-02040-5
