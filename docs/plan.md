# CellJEPA — Project Plan (v2: SOTA-Competitive Overhaul)

Date: 2026-02-11  
Status: v2.0 (complete architectural overhaul; P1–P7 milestones replace M0–M5)  
Canonical goal: Build a **SOTA-competitive perturbation prediction model** using JEPA-style representation learning with biologically-grounded gene-token architectures, evaluated head-to-head against GEARS, scGPT, CPA, and GeneFormer on standard benchmarks.

This file is the **canonical roadmap**: what we are building and the **gates** that determine whether we proceed.
Operational contracts (schemas, CLI expectations, metrics, HPC defaults) live in `docs/runbook.md`.

Doc precedence (if anything conflicts):
1) `docs/DECISIONS.md`
2) `docs/plan.md`
3) `docs/runbook.md`
4) `docs/archive/`

Current progress and run inventory: `docs/PROJECT_STATE.md`.

## 0) One-Page Summary

### Core thesis (testable)
A JEPA-style representation of cell state, built on **gene-token architectures** (Transformer / GNN / Perceiver) with **biologically-grounded perturbation encodings** and **gene-level decoded predictions**, will produce perturbation response predictions that **beat current SOTA deep learning methods** on standard benchmarks under strict evaluation.

### What changed from v1 (and why)
The v1 plan used a 3-layer MLP encoder, embedding-lookup perturbation encoding, and embedding-distance–only evaluation. A comprehensive critique (2026-02-11) identified these as structurally insufficient to compete with GEARS (GNN + Gene Ontology), scGPT (Transformer + 33M cells), CPA (disentangled latent factors), and GeneJEPA (Perceiver + 100M cells). The v2 plan:
- Replaces the MLP encoder with three gene-token encoder backends (Transformer, GNN, Perceiver)
- Replaces embedding-lookup perturbations with graph/fingerprint/identity encodings
- Adds gene-level decoded prediction metrics as primary evaluation
- Adds large-scale pretraining and head-to-head SOTA benchmarking
- Retains all evaluation rigor (mandatory baselines, CIs, split safety, leakage prevention)

### Product deliverable (what exists at the end)
1. A reproducible **benchmark harness** that:
   - downloads/prepares a curated suite of perturbation datasets (10+),
   - generates split files (holdout protocols),
   - trains simple + strong baselines,
   - trains CellJEPA models (all encoder backends),
   - runs SOTA methods (GEARS, scGPT, CPA) on identical splits,
   - produces head-to-head comparison tables with CIs.
2. A **CellJEPA model** package with:
   - Gene-token JEPA encoder pretraining (Transformer / GNN / Perceiver),
   - biologically-grounded perturbation encoders,
   - advanced transition/world-model predictors,
   - gene-level prediction decoders,
   - ablations isolating what matters.

### Non-goals (v2)
- Not trying to be the largest foundation model (we leverage existing pretraining corpora).
- Not claiming JEPA is universally better; we characterize *where* it wins and *where* it doesn't.

## 0.1 Default Decisions (to unblock implementation)

These are *defaults*, not permanent commitments. If a default changes, record it in `docs/DECISIONS.md`.

- **Stage A main-table splits:** `S1_unseen_perturbation` and `S2_unseen_context` (defined in §4).
- **v2 preprocessing:** library-size normalize → log1p; no batch correction.
- **v2 gene IDs:** gene symbols in `var.index` (store `var["ensembl_id"]` when available).
- **v2 gene tokenization:** Fourier expression features + learned gene identity embeddings.
- **v2 perturbation encoding:** gene identity embeddings for genetic perturbations; Morgan fingerprints for drugs; dose/time as continuous covariates.
- **v2 action tokens:** `obs["perturbation_tokens"]` is a deterministic `|`-separated string; control token is `control:CTRL`.
- **v2 evaluation filters:** drop `perturbation_id` in `{nan, none, ""}` ; default `min_cells_per_condition=30`.
- **v2 primary metrics:** gene-level LFC Pearson correlation + Top-20 DEG recall (embedding E-distance as secondary).
- **Safety-by-construction:** residual/shrinkage heads with `α ∈ [0,1]`, always include `α=0` (no-change), and clamp updates.
- **Cross-dataset:** shared-action only; control z-score calibration enabled by default; gene harmonization uses intersection sets.
- **Baselines are mandatory:** always report `no_change`, `mean_shift`, and `ridge` (plus shrinkage variants when relevant).
- **Encoder backends:** implement Transformer, GNN, and Perceiver; compare systematically.

## 0.2 Stop-the-Line Invariants (do not proceed if violated)

- **Data contract passes:** every `.h5ad` used in any run passes `celljepa.data.validation.validate_or_raise`.
- **Split artifacts are real artifacts:** train/val/test are group-based, deterministic, saved as JSON, and reused across methods.
- **No leakage:** any distribution- or label-dependent preprocessing is fit on **train only** and saved per split.
- **Evaluation unit is condition pairs:** aggregate metrics over `(context_id, perturbation_id)` pairs (not cell-weighted averages).
- **Baselines cannot be optional:** `no_change` must always be reported.
- **Cross-dataset must report overlap:** always report action overlap and restrict main tables to **shared-action** evaluation.
- **No non-finite values:** embeddings/predictions/metrics must be finite; any NaNs/Infs are a "stop and fix" event.
- **Gene-level metrics required:** every P3+ report must include LFC correlation and DEG recall, not just embedding distances.

## 1) Definitions (so experiments are unambiguous)

- **Gene token**: `(gene_id_embedding, expression_fourier_features)` — the atomic input unit.
- **Cell state embedding**: output of the encoder aggregated over gene tokens — `z = Encoder(gene_tokens)`.
- **Action / perturbation condition**: metadata `a` describing an intervention (gene KO, drug, dose, time).
- **State transition**: mapping from baseline/control state distribution to post-action state distribution within a context.
- **Baseline context**: the *control* distribution for a context (donor/cell line/cell type/batch).
- **Prediction target**: the **post-perturbation gene expression** for condition `a` within a context.

We support two prediction granularities:
- **Embedding-level** (latent space): predict condition embedding and evaluate via E-distance/cosine.
- **Gene-level** (decoded): predict per-gene log-fold changes and evaluate via LFC correlation/DEG recall.

## 2) Milestone Roadmap (P1–P7)

Previous milestones M0–M2 are complete and retained. M3 is superseded by P1–P3. M4–M5 are superseded by P4–P7.

### Completed (from v1)
- **M0** — Repo + reproducibility baseline ✅
- **M1** — Data correctness + baseline harness ✅
- **M2** — Basic JEPA pretraining (MLP, proof-of-concept) ✅

### P1 — Gene-Token Encoder + JEPA Rewrite
Deliverables:
- Shared gene tokenization layer (Fourier expression + gene identity embeddings).
- Three encoder backends: Transformer, GNN (with PPI/GO gene graph), Perceiver.
- Refactored JEPA class that works with any encoder backend.
- Gene-token–aware masking strategies (random gene mask, regulon-aware, pathway-block).
- Anti-collapse diagnostics logged for all backends.

Gate:
- All three backends train stably across ≥3 seeds; non-degenerate embeddings; collapse diagnostics logged; at least one backend produces clearly better downstream embeddings than the old MLP on Replogle.

### P2 — Perturbation Encoding Overhaul
Deliverables:
- GNN-based gene perturbation encoder using PPI/GO graph node embeddings.
- Chemical fingerprint encoder for drug perturbations (SMILES → Morgan fingerprints → MLP).
- Shared gene identity embeddings as perturbation vectors for single-gene perturbations.
- Combinatorial perturbation encoder (attentive pooling over individual embeddings).
- Continuous dose/time conditioning MLP.

Gate:
- S1 (unseen perturbation) with biologically-grounded encodings significantly outperforms the old embedding-lookup S1 on at least one dataset.

### P3 — World Model Overhaul + Gene-Level Decoder
Deliverables:
- Three world model architectures: attention-based, graph-conditioned, disentangled.
- Gene-level prediction decoder (latent → per-gene LFC predictions).
- Evaluation with gene-level metrics (LFC Pearson, DEG recall) as primary.
- Safety-by-construction retained (residual/shrinkage/α).

Gate (usefulness):
- **Must beat `no_change`** on at least one dataset/split with gene-level decoded metrics (LFC correlation or DEG recall). This is the fundamental usefulness gate.

### P4 — Large-Scale Pretraining + Dataset Expansion
Deliverables:
- Pretrain on CellxGene Census (~50M cells) or Tahoe-100M using the best encoder backend(s).
- Expand perturbation dataset suite to 10+ datasets covering diverse cell types and perturbation modalities.
- Efficient data loading infrastructure (streaming, multi-GPU).
- Gene interaction graph artifacts (STRING-db PPI + Gene Ontology).
- Regulon database artifacts (DoRothEA/CollecTRI).

Gate:
- Pretrained encoder outperforms from-scratch encoder on perturbation downstream tasks (S1 and S2) with CIs.

### P5 — Evaluation Framework + Head-to-Head Benchmarking
Deliverables:
- Complete metric suite: LFC Pearson, DEG recall, PerturBench rank metric, calibrated E-distance, cosine, MSE, kNN retrieval.
- Benchmark adapter scripts for GEARS, scGPT, CPA.
- Unified benchmark runner producing standardized comparison tables with CIs.
- All methods run on identical splits.

Gate (SOTA):
- CellJEPA beats at least one of {GEARS, scGPT, CPA} on at least one standard benchmark split with CIs under standard evaluation protocol.

### P6 — Multi-Modal Extension
Deliverables:
- scATAC-seq integration as regulatory prior (peak tokens, cross-modal attention).
- Perturb-CITE-seq (RNA + protein) joint encoder.
- Multi-modal JEPA objective (predict masked tokens in one modality from context in another).

Gate:
- Multi-modal model outperforms single-modal on at least one matched dataset.

### P7 — Publication Artifact + Final Benchmarking
Deliverables:
- Full ablation study: encoder backbone, perturbation encoding, world model, masking strategy, pretraining scale.
- Manuscript-quality benchmark report with CIs.
- Interpretability analysis (attention maps, GNN node importances).
- Reproducible artifact package (configs + splits + metrics + code).

Gate:
- Reproducible report artifact; beat SOTA with CIs on the primary benchmark; ablation study complete.

## 3) Data Plan

### 3.1 Dataset selection rubric
Score each candidate on:
- perturbation type diversity (genetic vs chemical),
- availability of controls matched by context,
- metadata completeness (perturbation ID, dose, time, donor/cell line),
- size (enough cells per condition for set metrics),
- cross-dataset compatibility (gene IDs, annotation quality),
- cell type diversity (prioritize datasets with cell types not already covered),
- minimal licensing / access friction.

### 3.2 Data contract (v2)
Hard requirement: every processed dataset artifact must pass `celljepa.data.validation.validate_or_raise`.

Required fields (v2):
- `X`: numeric expression matrix (fixed preprocessing).
- `var.index`: gene identifiers (must be unique).
- `obs` columns:
  - `perturbation_id` (string)
  - `is_control` (bool)
  - `context_id` (string)
  - `perturbation_tokens` (string; deterministic serialization)
- `uns` keys (provenance):
  - `dataset_id`, `preprocess_name`, `preprocess_version`, `created_at`

Recommended `obs` columns: `cell_type`, `batch`, `dose`, `time_hours`.

### 3.3 Gene tokenization contract (v2, new)
All encoder backends consume tokenized gene inputs:
- Each expressed gene → `(gene_id_embedding, expression_fourier_features)` token
- Gene identity embedding layer covers ~20,000 human genes
- Expression values encoded as Fourier features (continuous, no binning)
- Variable numbers of expressed genes per cell supported (sparse input)

### 3.4 Gene interaction graph artifacts
- `configs/graphs/ppi_go_graph_v1.pt`: STRING-db PPI + Gene Ontology co-annotation graph
- Used by: GNN encoder, gene perturbation encoder
- Gene nodes identified by gene symbol (matching `var.index`)

### 3.5 Regulon database artifacts
- `configs/regulons/dorothea_v1.json`: DoRothEA TF → target gene mappings
- Used by: regulon-aware masking strategy

### 3.6 Preprocessing
- library-size normalize → log1p (same as v1)
- fixed gene identifier standardization
- no batch correction unless strictly split-safe and justified
- any distribution-dependent statistics computed on training fold only

### 3.7 Action/perturbation metadata schema
Same as v1: `obs["perturbation_tokens"]` is a `|`-separated deterministic string.

## 4) Split Protocols

Identical to v1. See `docs/runbook.md` for operational details.

### 4.1 Stage A main-table splits
- **S1 — Unseen perturbation**: generalize to perturbations not seen during training.
- **S2 — Unseen context**: generalize to new donors/cell lines.

### 4.2 Cross-dataset holdout
- Shared-action only for main tables.
- Always report action overlap.
- Control z-score calibration enabled by default.

## 5) Modeling Plan

### 5.1 Gene Tokenization (shared across all encoders)
- `GeneIdentityEmbedding`: learned d-dim embedding per gene symbol (~20K genes)
- `FourierExpressionEncoder`: continuous expression → Fourier feature vector
- `GeneTokenizer`: combines identity + expression into per-gene tokens

### 5.2 Encoder Backends (P1)

**Transformer Encoder** (scGPT-inspired):
- Standard Transformer encoder on gene tokens
- Self-attention across gene dimension
- No positional encoding (gene identity provides identity)
- Cell-level readout via `[CLS]` token or mean-pooling
- Configurable: 4–12 layers, 4–8 heads, 256–512 embed_dim

**GNN Encoder** (GEARS-inspired):
- Input: gene tokens as node features on the PPI/GO gene interaction graph
- Message passing: GAT or GIN layers
- Cell-level readout: attention-weighted global pooling
- The graph provides inductive bias for gene-gene interaction modeling

**Perceiver Encoder** (GeneJEPA-inspired):
- Cross-attention from fixed latent tokens to variable-length gene tokens
- Fixed computational cost regardless of gene count
- Efficient for full-transcriptome input
- Configurable: 64–256 latent tokens

### 5.3 JEPA Training (P1)
- Student encoder `fθ` and teacher encoder `fθ̄` (EMA) — both use the same backend
- Gene-token–aware masking: mask specific gene tokens
- Predictor: `gφ(visible_token_repr, mask_info) → target_token_repr`
- Loss: representation regression (cosine/L2) between predicted and stop-grad teacher targets
- Anti-collapse: VICReg-style variance/covariance regularization

### 5.4 Masking Strategies (P1, first-class ablation)
- **RandomGeneMask**: randomly mask k% of gene tokens
- **RegulonMask**: mask a TF + its target genes (forces learning of regulatory logic)
- **PathwayBlockMask**: mask entire MSigDB/Reactome pathways
- **GORoleMask**: mask by GO functional category
- Report: mask type, fraction, module sizes, overlap handling

### 5.5 Perturbation Encoders (P2)
- **GeneGraphEmbedding**: encode perturbed gene via GNN node embedding in PPI/GO graph
- **ChemicalFingerprint**: SMILES → Morgan fingerprints → MLP → embedding
- **GeneIdentityEmbedding**: reuse gene identity embeddings as perturbation vectors
- **CombinatorialEncoder**: attentive pooling over individual perturbation embeddings
- **DoseTimeEncoder**: continuous (dose, time) → MLP → embedding

### 5.6 World Model / Transition Predictors (P3)
Three architectures, all retaining residual/shrinkage safety:

**Attention-Based:**
- Cross-attention from perturbation embedding to cell gene-token embeddings
- Perturbation "attends to" genes it affects → interpretable

**Graph-Conditioned:**
- Perturbation modeled as node intervention on gene interaction graph
- Message passing propagates perturbation signal through regulatory network
- Directly models the biological causal mechanism

**Disentangled:**
- Factorized latent space: base cell state × perturbation effect × covariates (dose, time, cell type)
- Enables counterfactual reasoning

### 5.7 Gene-Level Decoder (P3)
- Maps predicted post-perturbation latent embeddings → per-gene log-fold changes
- Light-weight (1–2 layer MLP), trained after JEPA pretraining (frozen encoder + trained decoder)
- Enables gene-level evaluation metrics

## 6) Baselines (mandatory and non-negotiable)

Same as v1, plus SOTA methods:

Simple baselines (always reported):
- no-change
- mean-shift per perturbation
- ridge regression in PCA / embedding space

SOTA baselines (P5, head-to-head):
- GEARS (GNN + Gene Ontology)
- scGPT (Transformer + pretraining)
- CPA (disentangled VAE)

Baseline fairness protocol:
- all methods run on identical splits
- fixed tuning budget and standardized early stopping
- report compute and hyperparameters
- publish search space and best config

## 7) Metrics and Reporting

### 7.1 Primary metrics (gene-level, new in v2)
- **LFC Pearson correlation**: correlation between predicted and observed log-fold changes across genes
- **Top-20 DEG recall**: of the true top-20 DEGs, how many are in the predicted top-20?
- **Direction accuracy**: binary up/down classification per gene
- **PerturBench rank metric**: novel ranking metric (0 = perfect ordering)

### 7.2 Secondary metrics (embedding-level, retained from v1)
- Energy distance (E-distance) over embedding sets
- Prototype cosine distance / MSE
- kNN retrieval accuracy / MRR
- Calibrated E-distance (Oct 2025 counter-benchmark variant)

### 7.3 Reporting + artifact contract
Per run (non-negotiable):
- `runs/<run_id>/metrics.json` (machine-readable; includes CIs)
- `runs/<run_id>/config.json` (CLI args / config snapshot)
- `runs/<run_id>/report.md` (human-readable summary)

Cross-dataset runs must additionally log:
- action overlap stats
- embedding scale diagnostics
- metrics stratified by shared-action vs unseen-action

Head-to-head benchmark reports (P5+):
- Comparison tables with CIs across all methods
- Ablation tables isolating component contributions

## 8) Engineering Plan (Reproducibility-First)

### Repo conventions
- config-snapshotted runs, deterministic split files, seeded training
- "Golden run" that completes on a small subset quickly
- no ad-hoc notebooks as execution path
- track `docs/` in git
- minimal environment spec committed

### New source code layout (P1+)
```
src/celljepa/
  models/
    gene_tokenizer.py         # Shared gene tokenization
    encoder_transformer.py    # Transformer backend
    encoder_gnn.py            # GNN backend
    encoder_perceiver.py      # Perceiver backend
    encoder_multimodal.py     # Multi-modal fusion (P6)
    masking.py                # Gene-token–aware masking
    perturbation_encoders.py  # Biologically-grounded perturbation encoding
    world_model.py            # Advanced world models (attention/graph/disentangled)
    decoder.py                # Gene-level prediction decoder
    jepa.py                   # Encoder-agnostic JEPA framework
    transition.py             # Legacy (retained for backward compat)
```

### HPC / Slurm execution context
- prefer `sbatch` for training/evaluation jobs
- default walltime > 24 hours (recommended: `--time=48:00:00`)
- multi-GPU support for large-scale pretraining (DDP/FSDP)
- run artifacts to `runs/<run_id>/…` and logs to `logs/`

## 9) Risk Register + Mitigations

- **Collapse / shortcut learning:** log collapse metrics for all encoder backends; enforce anti-collapse; regulon masks force non-trivial prediction.
- **Transformer overfitting on small datasets:** use pretraining on large corpus; then fine-tune on perturbation data.
- **GNN over-smoothing:** limit message-passing depth; use residual connections; monitor embedding distinguishability.
- **Gene-level decoding is too easy/hard:** calibrate decoder capacity; compare against linear probes.
- **SOTA comparison unfairness:** use identical splits, tuning budgets, and early stopping across all methods.
- **Pretraining compute:** start with CellxGene subset if full corpus is infeasible; scale up as validated.
- **External codebase drift:** pin GEARS/scGPT/CPA versions; use adapter pattern to insulate.

## 10) Stretch Goals (Explicitly Gated)

Only consider after P5 acceptance:
- Condition-level JEPA over sets of cells
- OT-based pseudo-pairing for cell-level matching
- Diffusion/LLM hybrids in latent space
- Cell Painting / morphology integration
- GWAS enrichment evaluation
- Survival outcome association validation
