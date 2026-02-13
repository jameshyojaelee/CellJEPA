# CellJEPA

CellJEPA investigates **Joint-Embedding Predictive Architectures (JEPAs)** for **single-cell omics**, with the goal of building a competitive perturbation prediction model that outperforms current deep learning methods (GEARS, scGPT, CPA, GeneFormer) on standard benchmarks.

This repo is **benchmark-driven**: the goal is to make a credible, head-to-head case for when JEPA-style learning beats (or loses to) existing approaches.

---

## Why I started this project

Predicting transcriptional responses to perturbations is a core problem in functional genomics and drug discovery. Recent benchmarking work has highlighted a hard reality: deep learning methods have **not yet consistently beaten simple linear baselines** on standard settings. See: *"Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines"* (Nature Methods, 2025): https://www.nature.com/articles/s41592-025-02772-6

CellJEPA is motivated by two hypotheses:
1. **A different training objective** — predicting *latent state* rather than reconstructing noisy observations — may produce better representations for downstream perturbation prediction.
2. **Gene-token architectures** (Transformer / GNN / Perceiver) with **biological priors** (gene interaction graphs, regulon masks) provide the right inductive biases for modeling how perturbations propagate through regulatory networks.

---

## Architecture

### Pipeline overview

```mermaid
flowchart TD
    %% ── Data Layer ──────────────────────────────────────────────
    subgraph DATA ["📦 Data Ingestion & Preprocessing"]
        RAW["Raw scRNA-seq\n(.h5ad)"]
        NORM["Library-size normalize\n→ log1p"]
        CONTRACT["Data contract validation\n(validate_or_raise)"]
        RAW --> NORM --> CONTRACT
    end

    subgraph ARTIFACTS ["🧬 Biological Artifacts"]
        PPI["PPI / GO Gene Graph\n(STRING-db)"]
        REG["Regulon Database\n(DoRothEA / CollecTRI)"]
        MSIG["Module Gene Sets\n(MSigDB / Reactome)"]
    end

    %% ── Gene Tokenization (P1) ─────────────────────────────────
    subgraph TOKENIZE ["🔤 Gene Tokenization (P1)"]
        GENEID["Gene Identity\nEmbedding (~20K genes)"]
        FOURIER["Fourier Expression\nEncoder (continuous)"]
        TOKEN["Gene Token\n= (id_emb, expr_features)"]
        GENEID --> TOKEN
        FOURIER --> TOKEN
    end

    CONTRACT --> TOKENIZE

    %% ── Encoder Backends (P1) ──────────────────────────────────
    subgraph ENCODERS ["🧠 Encoder Backends (P1)"]
        direction LR
        TF["Transformer\n(scGPT-inspired)\nSelf-attention on\ngene tokens"]
        GNN["GNN\n(GEARS-inspired)\nMessage passing on\nPPI/GO graph"]
        PERC["Perceiver\n(GeneJEPA-inspired)\nCross-attention from\nlatent tokens"]
    end

    TOKEN --> TF
    TOKEN --> GNN
    TOKEN --> PERC
    PPI -.-> GNN

    TF --> Z["Cell State Embedding z"]
    GNN --> Z
    PERC --> Z

    %% ── JEPA Pretraining (P1) ──────────────────────────────────
    subgraph JEPA ["🔁 JEPA Pretraining Loop (P1)"]
        direction TB
        STUDENT["Student Encoder fθ"]
        TEACHER["Teacher Encoder fθ̄\n(EMA)"]
        MASK["Gene-Token Masking"]
        PRED["Predictor gφ"]
        LOSS["Representation Loss\n(cosine / L2)"]
        VICREG["VICReg Anti-Collapse\n(variance + covariance)"]
        STUDENT --> PRED --> LOSS
        TEACHER --> LOSS
        MASK --> STUDENT
        MASK --> TEACHER
        VICREG --> LOSS
    end

    TOKEN --> JEPA
    REG -.-> MASK
    MSIG -.-> MASK

    subgraph MASKS ["🎭 Masking Strategies"]
        direction LR
        M1["Random\nGene Mask"]
        M2["Regulon Mask\n(TF + targets)"]
        M3["Pathway Block\nMask"]
    end

    MASKS --> MASK

    %% ── Perturbation Encoding (P2) ─────────────────────────────
    subgraph PERTURB ["💊 Perturbation Encoding (P2)"]
        direction LR
        PE1["Gene Identity\nEmbedding (KO/KD)"]
        PE2["Gene Graph\nEmbedding (GAT\non PPI/GO)"]
        PE3["Chemical Fingerprint\n(SMILES → Morgan → MLP)"]
        PE4["Combinatorial\n(Attentive Pooling)"]
        PE5["Dose / Time\n(Continuous → MLP)"]
    end

    PPI -.-> PE2

    PE1 --> A["Perturbation\nEmbedding a"]
    PE2 --> A
    PE3 --> A
    PE4 --> A
    PE5 --> A

    %% ── World Models (P3) ──────────────────────────────────────
    subgraph WORLD ["🌍 World Model / Transition Predictor (P3)"]
        direction LR
        W1["Attention-Based\nPerturbation cross-attends\nto gene tokens"]
        W2["Graph-Conditioned\nMessage passing on\nregulatory network"]
        W3["Disentangled\nFactorized: cell state\n× perturbation × covariates"]
    end

    Z --> W1
    Z --> W2
    Z --> W3
    A --> W1
    A --> W2
    A --> W3
    PPI -.-> W2

    W1 --> ZPRIME["Predicted Post-Perturbation\nEmbedding z'"]
    W2 --> ZPRIME
    W3 --> ZPRIME

    %% ── Gene-Level Decoder (P3) ────────────────────────────────
    subgraph DECODER ["🔬 Gene-Level Decoder (P3)"]
        DEC["MLP Decoder\n(1–2 layers, frozen encoder)"]
    end

    ZPRIME --> DEC
    DEC --> LFC["Per-Gene Log-Fold\nChange Predictions"]

    %% ── Evaluation (P5) ────────────────────────────────────────
    subgraph EVAL ["📊 Evaluation & Benchmarking (P5)"]
        direction LR
        subgraph PRIMARY ["Primary Metrics\n(Gene-Level)"]
            E1["LFC Pearson\nCorrelation"]
            E2["Top-20\nDEG Recall"]
            E3["Direction\nAccuracy"]
        end
        subgraph SECONDARY ["Secondary Metrics\n(Embedding-Level)"]
            E4["E-distance"]
            E5["Cosine / MSE"]
            E6["kNN Retrieval"]
        end
        subgraph BASELINES ["Baselines"]
            B1["no-change\nmean-shift\nridge"]
            B2["GEARS\nscGPT\nCPA"]
        end
    end

    LFC --> PRIMARY
    ZPRIME --> SECONDARY
    BASELINES -.->|"same splits"| PRIMARY

    %% ── Future Phases ──────────────────────────────────────────
    subgraph FUTURE ["🚀 Future Phases"]
        P4F["P4: Large-Scale Pretraining\n(CellxGene 50M+ / Tahoe 100M+)"]
        P6F["P6: Multi-Modal\n(scATAC + Perturb-CITE-seq)"]
        P7F["P7: Publication\n(Ablation + Manuscript)"]
    end

    JEPA -.->|"scale up"| P4F
    EVAL -.->|"extend"| P6F
    P6F -.-> P7F

    %% ── Styles ─────────────────────────────────────────────────
    classDef completed fill:#d4edda,stroke:#28a745,color:#000
    classDef active fill:#fff3cd,stroke:#ffc107,color:#000
    classDef future fill:#e2e3e5,stroke:#6c757d,color:#000

    class TOKENIZE,ENCODERS,JEPA,MASKS completed
    class PERTURB completed
    class WORLD,DECODER completed
    class EVAL active
    class FUTURE future
```

### Gene-token encoding
Every cell is tokenized at the gene level:
- **Gene identity embedding:** Learned embedding per gene (~20K human genes)
- **Fourier expression encoding:** Continuous expression → Fourier features (no binning)
- Each expressed gene becomes a token: `(gene_id_embedding, expression_features)`

### Three encoder backends (compared systematically)

| Backend | Inspiration | Key Property |
|---------|-------------|-------------|
| **Transformer** | scGPT | Self-attention captures gene-gene regulatory relationships |
| **GNN** | GEARS | Gene interaction graph provides compositional inductive bias |
| **Perceiver** | GeneJEPA | Fixed compute cost for variable gene counts |

### JEPA training
- Student/teacher (EMA) architecture applied to gene tokens
- Masking at the gene-token level: predict teacher representations of masked genes from visible context
- Masking strategies: random, regulon-aware (TF + targets), pathway-block
- VICReg-style anti-collapse regularization

### Perturbation encoding (biologically grounded)
- **Genetic perturbations:** Gene interaction graph node embeddings (PPI/GO)
- **Drug perturbations:** SMILES → Morgan fingerprints → MLP
- **Dose/time:** Continuous covariates via MLP
- **Combinations:** Attentive pooling over individual perturbation embeddings

### World model / transition prediction
- **Attention-based:** Perturbation cross-attends to gene tokens (interpretable)
- **Graph-conditioned:** Message passing propagates perturbation through regulatory network
- **Disentangled:** Factorized cell state × perturbation × covariates (CPA-inspired)

### Gene-level decoder
- Predicts per-gene log-fold changes from latent post-perturbation embeddings
- Enables gene-level evaluation metrics (not just embedding distances)

---

## Evaluation philosophy

### Primary metrics (gene-level)
- **LFC Pearson correlation**: predicted vs observed log-fold changes
- **Top-20 DEG recall**: correct identification of top differentially expressed genes
- **PerturBench rank metric**: perturbation ranking quality

### Secondary metrics (embedding-level)
- Energy distance, cosine distance, kNN retrieval

### Baselines (mandatory)
- Simple: no-change, mean-shift, ridge regression
- SOTA head-to-head: GEARS, scGPT, CPA (run on identical splits)

### Cross-dataset
- Shared-action only; control z-score calibration by default

---

## Datasets

### Pretraining (no perturbation labels needed)
- CellxGene Census (~50M human cells)
- Tahoe-100M (100M+ cells)

### Perturbation evaluation
- **scPerturb** (harmonized): Sci-Plex 2/3/4 (drug), Norman 2019 (CRISPRa), Replogle 2022 (CRISPRi)
- **Additional:** Dixit 2016, CROP-seq, McFarland 2020, NadigOConner 2024
- **Multi-modal (P6):** Perturb-CITE-seq (RNA + protein), 10x Multiome (RNA + ATAC)

---

## Known risks

- Omics lacks spatial topology → masking policy design is critical (we use regulon-aware masking)
- JEPA can collapse without stability constraints (we use VICReg + collapse diagnostics)
- Perturbation prediction requires distribution-level evaluation (we use per-condition-pair aggregation)
- Strong baselines are hard to beat (we include them mandatory and report honest results)
- DL benchmarking is metric-sensitive (we use both classical and calibrated metrics)

Negative results are treated as informative, as long as they are well-controlled.

---

## Getting started

- Project plan: `docs/plan.md`
- Runbook (contracts + splits + metrics + HPC): `docs/runbook.md`
- Decision log: `docs/DECISIONS.md`
- Project state: `docs/PROJECT_STATE.md`

---

## Repo layout

- `src/celljepa/`   library code (models, losses, data, eval)
- `scripts/`        runnable entry points (train/eval/report)
- `configs/`        experiment, dataset, graph, and regulon configs
- `docs/`           design docs and reports
- `data/`           local data cache (gitignored)
- `runs/`           outputs and checkpoints (gitignored)

---

## References

- I-JEPA: https://arxiv.org/abs/2301.08243
- GeneJEPA: https://www.biorxiv.org/content/10.1101/2025.10.14.682378v1
- sc-JEPA: https://openreview.net/forum?id=MZDkttBUEd
- GEARS: https://www.nature.com/articles/s41587-023-01905-6
- scGPT: https://www.nature.com/articles/s41592-024-02201-0
- GeneFormer: https://www.nature.com/articles/s41586-023-06139-9
- CPA: https://www.embopress.org/doi/full/10.15252/msb.202211517
- scPerturb: https://www.nature.com/articles/s41592-023-02144-y
- Perturb-CITE-seq: https://www.nature.com/articles/s41588-021-00779-1
- Benchmark caution: https://www.nature.com/articles/s41592-025-02772-6
