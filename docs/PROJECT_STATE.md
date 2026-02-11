# CellJEPA - Project State

Purpose: maintain a compact, current snapshot of progress, reasoning, and key artifacts so a new session can resume quickly.

Last updated: 2026-02-11 (P5 benchmark suite complete)

Update rules:
- Update after each milestone completion, major run, or shift in reasoning.
- Keep entries brief and link to concrete artifacts.
- Keep this in sync with `docs/DECISIONS.md` and `docs/plan.md`.

## Current snapshot
- Active milestone: **P5 (evaluation framework + benchmarking)** per `docs/plan.md` v2. Code infrastructure complete; actual GEARS/scGPT/CPA runs pending.
- Completed milestones: M0, M1, M2 (from v1 roadmap), P1 (gene-token encoder + JEPA rewrite), P2 (perturbation encoding overhaul), P3 (world model overhaul + gene-level decoder), P4 (code infrastructure), P5 (evaluation framework code).
- Superseded milestones: M3 (ridge win insufficient; replaced by P3 usefulness gate), M4, M5 (replaced by P4–P7).
- Current focus: Train CellJEPA models, run head-to-head evaluations against GEARS/scGPT/CPA.
- Key decisions: see `docs/DECISIONS.md` (2026-02-11 v2 overhaul entries).
- Operational reference: `docs/runbook.md`.

## v2 Milestone Status (P1–P7)

| Milestone | Status | Notes |
|-----------|--------|-------|
| M0 Repo + reproducibility | ✅ Complete | Toy quickstart works |
| M1 Data + baselines | ✅ Complete | All datasets ingested; baselines running |
| M2 Basic JEPA (MLP) | ✅ Complete | Proof of concept; superseded by P1 |
| **P1** Gene-token encoder | ✅ Complete | Transformer, GNN, Perceiver + JEPAv2 + masking + artifacts |
| **P2** Perturbation encoding | ✅ Complete | 5 encoder classes: GeneIdentity, GeneGraph, ChemicalFingerprint, Combinatorial, DoseTime |
| **P3** World model + decoder | ✅ Complete | 3 architectures: Attention, Graph, Disentangled + GeneLevelDecoder + 3 gene-level metrics |
| **P4** Large-scale pretraining | ⏳ Code complete | Streaming data, DDP training, 12-dataset catalog, v2 pretrain script. Operational runs pending. |
| **P5** Benchmark suite | ⏳ Code complete | 10-metric suite, 6 SOTA adapters, unified runner, report generator. Head-to-head runs pending. |
| **P6** Multi-modal | 🔲 Not started | scATAC + Perturb-CITE-seq |
| **P7** Publication | 🔲 Not started | Ablation + manuscript |

## Key decisions from v2 overhaul (2026-02-11)
- MLP encoder replaced by gene-token Transformer/GNN/Perceiver
- Embedding-lookup perturbation encoding replaced by PPI/GO graph embeddings and chemical fingerprints
- Primary metrics changed from embedding E-distance to gene-level LFC Pearson + DEG recall
- M3 acceptance (ridge win) superseded → P3 requires beating no-change with gene-level metrics
- Head-to-head SOTA benchmarking added (GEARS, scGPT, CPA)
- Three encoder backends explored in parallel (no pre-commitment)
- Multi-modal extension formalized (scATAC, Perturb-CITE-seq)

## Progress log

### v2 (2026-02-11+)
- 2026-02-11: v2 complete architectural overhaul approved. Rewrote `docs/plan.md` with P1–P7 milestones. Updated `docs/DECISIONS.md` with 4 new decision entries.
- 2026-02-11: P1 complete. Implemented gene tokenizer (Fourier + identity), 3 encoder backends (Transformer, GNN, Perceiver), JEPAv2 framework, regulon-aware masking, graph/regulon artifact builders. Anti-collapse validated: 9/9 runs healthy across all backends × 3 seeds.
- 2026-02-11: P2 complete. Implemented 5 biologically-grounded perturbation encoders in `src/celljepa/models/perturbation_encoders.py`: GeneIdentityPerturbationEncoder (with optional weight sharing), GeneGraphPerturbationEncoder (GAT on PPI/GO graph), ChemicalFingerprintEncoder (Morgan FP → MLP), CombinatorialPerturbationEncoder (attentive pooling), DoseTimeEncoder. All 7 smoke tests passed.
- 2026-02-11: P3 complete. Rewrote `src/celljepa/models/world_model.py` with 3 world model architectures (AttentionWorldModel, GraphConditionedWorldModel, DisentangledWorldModel) + safety-by-construction (residual + α + delta clamping). Created `src/celljepa/models/decoder.py` with GeneLevelDecoder (cell-level + gene-level input modes). Added gene-level metrics (LFC Pearson, top-k DEG recall, direction accuracy) to `src/celljepa/eval/metrics.py`. All 9 smoke tests passed.
- 2026-02-11: P4 code infrastructure complete. Created streaming data loading (`streaming_dataset.py`: CellDataset + MultiDatasetMixer), dataset catalog (`dataset_catalog.py`: 12 datasets), DDP utilities (`distributed.py`), and v2 pretraining script (`pretrain_jepa_v2.py`). All 7 smoke tests passed. Operational runs (data downloads, HPC pretraining) pending.
- 2026-02-11: P5 evaluation framework complete. Extended `metrics.py` with 4 new metrics (PerturBench rank, kNN retrieval, MRR, calibrated E-distance). Created `benchmark_adapters.py` (6 adapters: NoChange, MeanShift, CellJEPA, GEARS, scGPT, CPA), `benchmark_runner.py` (unified runner with CIs), and rewritten `report.py` (comparison tables). All 7 smoke tests passed.

### v1 (2025-12-24 – 2026-01-14)
- 2025-12-24: M0 complete. Implemented data-contract checks, split generator CLI, and toy dataset generator.
- 2025-12-24: M1 complete. Ingested chosen datasets, added baseline harness.
- 2025-12-24: M2 complete. Implemented JEPA pretraining (online/teacher/predictor), masking, collapse diagnostics.
- 2025-12-25: M3 implemented but acceptance marginal. Beat ridge on Replogle S2 but not no-change.
- 2025-12-25: M4 started with cross-dataset gene overlap analysis.
- 2026-01-14: M4-v2 Nadig downloads and ingestion submitted.
- See earlier entries in git history for full v1 progress log.

## Key artifacts
- Plans and guardrails: `docs/plan.md` (v2), `AGENTS.md`.
- Decision log: `docs/DECISIONS.md`.
- Runbook: `docs/runbook.md`.
- v2 encoder modules: `src/celljepa/models/gene_tokenizer.py`, `encoder_transformer.py`, `encoder_gnn.py`, `encoder_perceiver.py`.
- v2 JEPA framework: `src/celljepa/models/jepa.py` (JEPAv2 + v1 preserved).
- v2 perturbation encoders: `src/celljepa/models/perturbation_encoders.py` (5 encoder classes + factory).
- v2 world models: `src/celljepa/models/world_model.py` (3 architectures + safety + factory + v1 preserved).
- v2 gene-level decoder: `src/celljepa/models/decoder.py` (cell-level + gene-level modes).
- v2 gene-level metrics: `src/celljepa/eval/metrics.py` (LFC Pearson, DEG recall, direction accuracy).
- v2 data infrastructure: `src/celljepa/data/streaming_dataset.py` (CellDataset + MultiDatasetMixer), `dataset_catalog.py` (12-dataset registry).
- v2 distributed training: `src/celljepa/train/distributed.py` (DDP + scheduler).
- v2 pretraining script: `scripts/pretrain_jepa_v2.py` (JEPAv2 + streaming + DDP).
- v2 benchmark suite: `src/celljepa/eval/benchmark_adapters.py` (6 adapters), `benchmark_runner.py` (unified runner), `report.py` (comparison tables).
- v2 masking: `src/celljepa/models/masking.py`.
- Gene graph artifact: `configs/graphs/synthetic_test_graph.pt`.
- Regulon artifact: `configs/regulons/synthetic_test.json`.
- v1 reports (reference): `reports/m3_sciplex3_report.md`, `reports/m3_summary.md`.
- Split artifacts: `runs/m0_splits/`, `runs/m1_splits/`.

## Run inventory (v1 runs — retained as reference)
- M0 splits: `runs/m0_splits/`
- M1 splits: `runs/m1_splits/`
- M1 baselines: `runs/m1_*_baselines_full/`
- M2 JEPA: `runs/m2_*_jepa_*/`
- M3 runs: `runs/m3_full/`, `runs/m3_full_v3/`

## Open questions / upcoming work
- P4 operational: Download perturbation datasets from scPerturb, ingest with data contract, run pretraining on HPC.
- P4 operational: Generate real STRING-db PPI + GO graph artifacts and DoRothEA regulon artifacts.
- P5: Decide on GEARS/scGPT/CPA baseline integration strategy.
