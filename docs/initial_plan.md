1) Extensive project plan with JEPA as the core
Core objective

Build and benchmark a JEPA-first perturbation prediction system for omics, where the primary predicted object is a latent cell state embedding, not reconstructed counts.

Why this is aligned with JEPA:

I-JEPA’s defining move is to predict representations of withheld parts from a context, with masking design as a core lever. 
arXiv

SC-JEPA explicitly adapts JEPA to single-cell transcriptomics, using a context subset and target subset, with a predictor that uses pointer tokens to specify which subset to predict. 
OpenReview
+1

scPerturb motivates benchmarking across heterogeneous perturbation datasets and provides an evaluation framing via energy statistics and E-distance between sets of expression profiles. 
PMC

Primary claim to demonstrate (project thesis): A JEPA-trained embedding space makes perturbation prediction more practical, robust, and benchmarkable in omics than objectives focused on reconstructing noisy measurements.
This is not guaranteed; you will need to show it empirically against strong baselines including linear controls. 
Nature
+1

1.1 Scope and staged structure
Stage A: Core (must work without extra modalities)

Goal: A JEPA encoder + JEPA-style conditional latent predictor that can predict perturbation response embeddings and optionally decode into gene expression deltas.

Data: single-cell perturbation datasets from scPerturb (harmonized, multi-study, multi-tech). 
PMC

Task: perturbation prediction under rigorous OOD splits (details below).

Deliverable: open evaluation harness that includes deliberately simple baselines (mandatory because deep models often do not beat them). 
Nature
+1

Stage B: Core+Multi-modal (RNA+protein)

Goal: Extend JEPA to multi-modal perturbation readouts (RNA+protein) to test whether JEPA gains robustness when the “state” is multi-view.

Data: Perturb-CITE-seq (pooled CRISPR screen with multi-modal RNA+protein single-cell readout). 
PMC

Stage C: Optional hybrids (if Stage A/B are solid)

LLM-JEPA-style token models for discrete representations or perturbation text metadata. 
arXiv

Diffusion-JEPA hybrids in latent space or via noise schedules (see N-JEPA). 
arXiv

Stage D: Later additions (explicitly not core)

Branch 1: Morphology integration (JUMP Cell Painting + L1000). 
Virtual Cell Models
+2
Lincs Portal
+2

Branch 2: Spatial perturbation prediction context. 
OUP Academic
+1

GWAS + survival as external validation layers (not core per your request).

1.2 Data plan
Datasets for the core
(A) scPerturb-based benchmark suite (Core)

scPerturb includes 44 perturbation-response datasets with molecular readouts, and provides standardized QC and harmonized feature annotations. 
PMC

Key value: cross-study heterogeneity is a primary threat model for “practical” omics ML.

Selection strategy (practical):

Start with 2–4 scPerturb datasets that cover distinct regimes:

CRISPR Perturb-seq-like genetic perturbations

chemical perturbations

at least one dataset with multiple cell lines/donors to enable “hold-out context” evaluation
This mirrors SC-JEPA’s focus on generalizing perturbation response across held-out contexts. 
OpenReview

(B) Perturb-CITE-seq (Core+Multi-modal)

Perturb-CITE-seq is explicitly multi-modal, enabling a much stronger test of “latent state” learning than RNA alone. 
PMC

Preprocessing and representation choices (omics-specific risks)

Omics-specific issues that matter for JEPA:

scRNA is sparse and noisy; masking objectives can degenerate into learning “library size / dropout heuristics” if masking is naive. [Inference]

Across datasets, gene sets differ; JEPA with tokenization can help but introduces vocabulary alignment issues. [Inference]

Conservative preprocessing:

Work in log1p-normalized expression or standardized residual-like space (exact method depends on dataset; keep it consistent).

Define a “foundation gene set”:

intersection across datasets for cross-dataset training, plus

dataset-specific tails handled via missing tokens / masked tokens.

Preserve raw counts only if you later add a generative decoder.

1.3 JEPA model design (centerpiece)
A. Adapt JEPA’s “context predicts target embedding” to transcriptomes
What JEPA is operationally (for this project)

Use the SC-JEPA formulation:

partition input into context subset and target subset

encoder produces summaries for subsets

predictor takes context summary and a pointer describing the target subset, and predicts target summary 
OpenReview
+1

This maps cleanly to transcriptomes:

“object” = a cell’s gene-expression features (or a set of cells if you do population-level)

“subset” = a subset of genes (or gene modules)

Architecture options (prioritize stability and speed)

You want something that is:

stable against collapse

handles high-dimensional sparse-ish input

scales to multiple datasets

Option 1 (recommended): Perceiver-style gene-set encoder
GeneJEPA’s public description uses a Perceiver-style encoder over unordered gene sets, a continuous-value tokenizer (gene identity + Fourier features for expression), and EMA teacher targets with variance–covariance regularization. 
GitHub

This is aligned with JEPA’s need for a stable teacher and anti-collapse regularization.

Option 2: Transformer over gene tokens
More standard but heavier; also closer to scGPT-style pipelines (useful for comparisons). 
Nature

Minimum viable JEPA recipe (implementation-critical):

Online encoder: 
𝑓
𝜃
f
θ
	​


Target encoder (teacher): 
𝑓
𝜃
ˉ
f
θ
ˉ
	​

 updated via EMA

Predictor head: 
𝑔
𝜙
g
ϕ
	​


Loss: cosine or L2 between 
𝑔
𝜙
(
𝑓
𝜃
(
context
)
,
𝜋
𝑇
)
g
ϕ
	​

(f
θ
	​

(context),π
T
	​

) and stopgrad
(
𝑓
𝜃
ˉ
(
target
)
)
(f
θ
ˉ
	​

(target))

Anti-collapse: variance/covariance regularization (GeneJEPA-style) 
GitHub

B. Masking strategies (this is where JEPA likely wins or fails)

I-JEPA highlights masking design as central, especially that target blocks should be large-scale and context informative. 
arXiv

For transcriptomes, “large-scale” translates to “biologically coherent blocks” more than contiguous indices.

Masking strategies to implement and ablate:

Random gene block masking

easy baseline; likely to work but may overfit technical artifacts [Inference]

Pathway/module masking (Hallmark, Reactome, GO-based modules)

more semantically aligned; higher chance JEPA learns “programs” [Inference]

Transcription factor regulon masking

stresses causal structure; may improve perturbation transfer [Speculation]

Adversarial masking

mask genes most predictive of perturbation label (estimated via a probe) to force use of broader context [Speculation]

Key ablation knobs (must report):

fraction masked

number of target blocks per cell

whether targets overlap

module size distributions

C. Train JEPA as both cell-level and (optionally) population-level

SC-JEPA is fundamentally about population-level representation (sets of samples) and argues JEPA is attractive for set-structured data without needing handcrafted augmentations. 
OpenReview

You can use this in two complementary ways:

Cell-level JEPA (core)
Each cell is an object; subsets are gene blocks.

Condition-level JEPA (strong for perturbation prediction)
Each perturbation condition is an object defined by a set of cells; context is a subset of cells and target is another subset.
This directly targets distribution shift and heterogeneity. [Inference, high value but more engineering]

1.4 Perturbation prediction head in latent space (JEPA-centric)
What you predict

Predict post-perturbation embedding, not raw counts, as the primary target.

This mirrors SC-JEPA’s evaluation: they train a lightweight MLP to predict the embedding of a perturbed state from the embedding of its unperturbed baseline in a held-out-cell-line experiment. 
OpenReview

The pairing problem (no paired pre/post cells)

Most perturbation datasets are not paired at single-cell level. You need a strategy that is honest about this.

Three progressively stronger training objectives:

Level 0: Prototype transition (fast, low-risk)

Compute baseline prototype embedding 
𝜇
0
,
𝑡
μ
0,t
	​

 for cell type 
𝑡
t

Compute perturbed prototype 
𝜇
𝑝
,
𝑡
μ
p,t
	​


Train 
ℎ
(
𝜇
0
,
𝑡
,
𝑎
𝑝
)
→
𝜇
𝑝
,
𝑡
h(μ
0,t
	​

,a
p
	​

)→μ
p,t
	​


Pros: quick; stable; good for early signal.
Cons: collapses heterogeneity; may hide failure modes. [Inference]

Level 1: Set-to-set distribution matching (recommended for core paper-quality results)

Train a conditional model that maps a batch of baseline embeddings into a batch of predicted perturbed embeddings, and minimize a set distance to the empirical perturbed set.

You can use:

Energy distance / E-distance framing as highlighted by scPerturb for comparing sets of profiles. 
PMC

MMD / sliced Wasserstein (standard, but you should justify choice)

Pros: respects heterogeneity; fits JEPA’s philosophy of predicting state rather than exact counts.
Cons: heavier training loop; careful batching needed. [Inference]

Level 2: OT-based pseudo-pairing (optional, but strong if done carefully)

Use optimal transport to map baseline distribution to perturbed distribution and create pseudo-pairs for training. CINEMA-OT is an example of OT being used in perturbation analysis. 
Nature

Pros: can yield sharper conditional mapping.
Cons: easy to leak information / overfit if OT computed on full data; must do OT only on training folds. [Inference]

Output options

Deterministic output: single predicted embedding

Stochastic output: distribution over embeddings (see diffusion hybrid)

1.5 Hybrid models (optional, but pre-plan them so results are interpretable)
A. JEPA + generative decoder (practical compromise)

If reviewers demand transcriptome outputs:

Keep JEPA as the state predictor

Add a decoder to map predicted embedding to:

differential expression vector, or

reconstructed expression distribution

This makes the “product” optional and keeps the main thesis as JEPA. [Inference]

B. Diffusion-JEPA hybrid (two defensible interpretations)

You asked “if that’s a thing.” There is at least vision-side work connecting diffusion noise and JEPA-like masked modeling (N-JEPA). 
arXiv

In omics, a diffusion-JEPA hybrid is not something I can verify as standard. [Unverified]

Two versions you can justify:

Noise-augmented JEPA training (robustness)

Add a noise schedule to expression inputs or token embeddings (analogous to diffusion noise-as-mask viewpoint). 
arXiv

Hypothesis: improves robustness across dataset shifts. [Speculation]

Latent diffusion for stochastic perturbation outcomes

Train diffusion in latent space to model 
𝑝
(
𝑧
𝑝
𝑜
𝑠
𝑡
∣
𝑧
𝑝
𝑟
𝑒
,
𝑎
)
p(z
post
	​

∣z
pre
	​

,a)

JEPA supplies the latent space and teacher targets; diffusion supplies calibrated stochasticity. [Speculation]

C. LLM-JEPA-style approach (discrete barrier workaround)

LLM-JEPA proposes JEPA-based objectives for language models, motivated by embedding-space objectives outperforming input-space reconstruction in vision. 
arXiv

Omics adaptation ideas (optional):

Tokenized transcriptomes

Represent a cell as tokens (gene id + binned expression or rank)

Train with JEPA objective over token subsets

Perturbation metadata encoder

If you have text descriptions (drug MOA, gene annotations), encode with an LLM-JEPA-style encoder and fuse with cell embedding

These are plausible but higher risk and not necessary for the core JEPA claim. [Speculation]

1.6 Benchmarking and evaluation (make it defensible against the “linear baselines win” critique)

This must be treated as a first-class deliverable because there is direct evidence that deep models often fail to beat simple baselines in perturbation prediction. 
Nature
+1

A. Baselines (mandatory)
Deliberately simple controls

No-change baseline

Mean-shift baseline per perturbation

Linear regression in PCA space

Additive baseline for combos (if combo perturbations exist)

Justification: multiple benchmarks emphasize the importance of these controls. 
Nature
+1

Established perturbation predictors

scGen 
PubMed

CPA 
BioRxiv
+1

GEARS 
Nature

Foundation/generative comparators (for objective comparison)

scGPT (generative transformer; important comparator) 
Nature

Your narrative is: JEPA predicts state embeddings; scGPT predicts expression tokens/count-like outputs.

B. Splits (OOD-first, or results will be dismissed)

You should report at least two of:

Hold out perturbations (unseen genes/drugs)

Hold out cellular contexts

donor holdout

cell line holdout (mirrors SC-JEPA’s leave-one-cell-line-out perturbation generalization test) 
OpenReview

Cross-dataset holdout (train on subset of scPerturb datasets, test on held-out dataset)

C. Metrics (state-space and expression-space)

State-space metrics (JEPA-native):

cosine distance between predicted and true embeddings (SC-JEPA uses cosine distance in its perturbation prediction evaluation) 
OpenReview

kNN retrieval accuracy: does predicted embedding retrieve cells from the correct perturbation condition?

Set-level metrics:

energy distance / E-distance between predicted set and observed perturbed set (aligned with scPerturb’s emphasis) 
PMC

Expression-space metrics (only if you decode):

correlation of predicted vs observed DE

gene set enrichment consistency

calibration checks if you output distributions

D. Ablations to prove “it’s JEPA, not just more parameters”

JEPA vs masked reconstruction objective (same backbone)

EMA teacher on/off

variance/cov regularization on/off (if used)

random masks vs module masks

cell-level JEPA vs condition-level JEPA (if implemented)

1.7 Engineering plan (what makes it “practical”)
Reproducibility and packaging

Minimum bar for “practical and useful”:

a single command to:

download/prepare a dataset subset

train JEPA encoder

train perturbation predictor head

run benchmark suite producing a fixed report

Compute discipline

To avoid the critique “JEPA only works at giant scale”:

train at 2–3 scales (small, medium, larger) and report scaling curves

keep baselines trained with comparable compute where possible [Inference]

Failure mode handling

JEPA-like methods can collapse without careful design. You should pre-plan:

collapse diagnostics (embedding variance over batch, predictor outputs)

early stopping triggers

fallback to stronger regularization (variance/cov, predictor capacity changes) [Inference]

1.8 Later additions (brief, as requested)
Branch 1: Morphology integration

JUMP Cell Painting pilot dataset exists for chemical and genetic perturbations in U2OS and A549. 
Virtual Cell Models

CPJUMP1 resource provides large-scale morphological profiles of perturbed genes, designed around links between genetic and chemical perturbations targeting the same proteins. 
Nature

LINCS L1000 provides perturbation gene-expression signatures. 
Lincs Portal
+1

JEPA role: align a shared “perturbation state” embedding across morphology and expression, then test cross-modal perturbation prediction. [Speculation]

Branch 2: Spatial perturbation context

SDMap is described as a database of spatial drug perturbation maps. 
OUP Academic
+1

CONCERT is a niche-aware model for spatial perturbation transcriptomics prediction. 
PubMed
+1

JEPA role: learn context/target spatial neighborhood embeddings and predict withheld spatial regions in latent space, then connect to perturbation response prediction. [Speculation]

GWAS + survival (later layers)

Not core. Potential uses:

GWAS-enriched gene module masks as “semantic targets” for JEPA [Speculation]

survival association as external validity check for embeddings [Speculation]

2) Systematic critique of the plan
Novelty

Strengths

JEPA applied explicitly to perturbation prediction with set-level matching losses (energy distance / E-distance style) is plausibly novel as a complete system, even if elements exist separately. [Inference]

Multi-modal JEPA (RNA+protein) focused on predicting latent state transitions is a clean differentiator from reconstruction-heavy pipelines.

Risks

SC-JEPA already demonstrates perturbation generalization in embedding space using a held-out cell line setup. 
OpenReview

GeneJEPA publicly claims perturbation reasoning and provides a JEPA recipe for scRNA. 
GitHub

So novelty may be incremental unless you:

show new OOD regimes (cross-dataset, cross-modality)

show robust wins over linear controls on carefully chosen metrics

Accuracy and benchmark defensibility

Strengths

You plan to anchor evaluation in rigorous baselines, which is mandatory given evidence that deep methods often do not beat simple linear baselines. 
Nature
+1

Using JEPA-native metrics (embedding prediction, retrieval) avoids pretending to predict precise counts when measurement noise dominates. [Inference]

Risks

Reviewers may argue embedding metrics are “moving the goalposts” unless you connect embeddings to practical decisions (perturbation ranking, signature matching, multi-modal agreement). [Inference]

OT-based training can easily leak information if transport maps are computed using test data. [Inference]

Impact

High-impact if you achieve:

clear OOD generalization win (held-out perturbations + held-out contexts)

robust multi-modal gains on Perturb-CITE-seq 
PMC

a reproducible benchmark harness with strong baselines, addressing current field concerns 
Nature
+1

Lower impact if:

improvements are only in-distribution

wins are only on embedding metrics with weak downstream linkage

Feasibility

High feasibility pieces

scPerturb availability and standardized curation make it realistic to build a benchmark suite. 
PMC

Perturb-CITE-seq provides a concrete multi-modal testbed. 
PMC

Medium feasibility / high-risk pieces

condition-level JEPA over sets of cells is more complex but potentially very valuable. [Inference]

diffusion hybrids are speculative in omics and may distract unless introduced late. 
arXiv

Primary feasibility risk

Achieving consistent improvement over simple baselines. This is explicitly nontrivial per Nature Methods and Bioinformatics benchmarks. 
Nature
+1

Mitigation: define success criteria that includes:

OOD generalization wins

robustness across datasets

multi-modal consistency gains

compute-efficiency or stability advantages relative to reconstruction objectives
All require careful measurement. [Inference]

3) Finalized “bulletproof” project report plan
Title (working)

JEPA for Practical Perturbation Prediction in Single-Cell Omics: Latent-State Forecasting with Rigorous Baselines

Core claims to test

JEPA embeddings are better substrates for perturbation prediction under OOD shifts than embeddings learned via reconstruction or contrastive alignment. 
arXiv
+1

Latent-state prediction plus set-level evaluation is more faithful to single-cell perturbation data structure than predicting exact counts. 
PMC
+1

Multi-modal JEPA (RNA+protein) improves robustness and reduces ambiguity in predicted perturbation outcomes. 
PMC

Experimental design
Dataset suite

Core: 2–4 scPerturb datasets spanning genetic and chemical perturbations. 
PMC

Multi-modal: Perturb-CITE-seq for RNA+protein evaluation. 
PMC

Models

Your method (JEPA-Perturb)

JEPA encoder with EMA teacher, predictor, anti-collapse regularization (GeneJEPA-style design choices are a credible starting point). 
GitHub
+1

Perturbation transition head:

v1: prototype transition

v2: set-to-set distribution matching using E-distance style objectives (or justified alternative) 
PMC

Baselines

Simple controls: no-change, mean-shift, linear regressors

scGen 
PubMed

CPA 
BioRxiv
+1

GEARS 
Nature

scGPT as a representative generative foundation model baseline 
Nature

Splits

Minimum required:

Hold-out perturbations

Hold-out context (donor or cell line) to mirror SC-JEPA’s held-out generalization framing 
OpenReview

Stretch:

Cross-dataset holdout across scPerturb subsets

Metrics

Primary (JEPA-native):

embedding prediction cosine distance and retrieval

set-level distance (E-distance inspired) 
PMC
+1

Secondary (if decoding enabled):

DE correlation and pathway-level agreement

Ablations (to isolate JEPA contribution)

JEPA vs masked reconstruction (same backbone)

masking strategy (random vs module masks)

teacher EMA on/off

regularization on/off (variance-cov)

Hybrids policy (to keep the project focused)

Do not introduce diffusion or LLM-JEPA hybrids until you have a stable JEPA baseline beating at least some controls on at least one OOD split.

Then add one hybrid as a capped experiment:

either latent diffusion for stochastic prediction [Speculation]

or LLM-JEPA-style tokenization experiment [Speculation] 
arXiv
+1

Output artifacts (what makes it “practical”)

Reproducible benchmark harness with fixed splits and baselines

Pretrained JEPA checkpoints for each dataset suite

A “model card” style report:

where it works (OOD regime)

where it fails (baseline wins)

compute footprint

recommended use cases and non-use cases

Later expansions (explicitly out of scope for initial paper)

Branch 1: morphology + L1000 alignment 
Virtual Cell Models
+2
Lincs Portal
+2

Branch 2: spatial perturbation context (SDMap, CONCERT) 
OUP Academic
+1

GWAS modules and survival associations as external validity layers [Speculation]

Citations

I-JEPA (masking strategy, context-to-target representation prediction). 
arXiv

SC-JEPA (JEPA objective for set-structured single-cell data; perturbation generalization in embedding space). 
OpenReview
+1

GeneJEPA (public implementation notes: Perceiver encoder, tokenizer, EMA teacher, variance/cov regularization). 
GitHub

scPerturb (44 harmonized perturbation datasets; E-statistics and E-distance). 
PMC

Perturb-CITE-seq (multi-modal RNA+protein perturbation screens). 
PMC

Baseline warnings (deep models not beating linear controls in perturbation prediction). 
Nature
+1

Perturbation prediction baselines: scGen, CPA, GEARS, scGPT. 
Nature
+3
PubMed
+3
BioRxiv
+3

Hybrid pointers: LLM-JEPA, N-JEPA. 
arXiv
+1

Optional branches: JUMP, L1000, SDMap, CONCERT. 
Nature
+4
Virtual Cell Models
+4
Lincs Portal
+4

Claim Check (with tags)

scPerturb contains 44 publicly available single-cell perturbation-response datasets with transcriptomics, proteomics, and epigenomics and introduces E-statistics/E-distance framing. [Verified] 
PMC

Perturb-CITE-seq is a pooled CRISPR screen with multi-modal single-cell RNA and protein readout. [Verified] 
PMC

I-JEPA predicts representations of target blocks from a context block and emphasizes masking strategy. [Verified] 
arXiv

SC-JEPA defines a context/target partition with an encoder and predictor using pointer tokens; it evaluates predicting perturbed-state embeddings from baseline embeddings in a held-out context. [Verified] 
OpenReview
+1

GeneJEPA uses a Perceiver-style encoder, continuous-value tokenizer, EMA teacher targets, and variance-covariance regularization (per its public repo description). [Verified as “repo claims,” not independently validated] 
GitHub

Deep learning models often do not outperform simple linear baselines for gene perturbation prediction, motivating rigorous benchmarking. [Verified] 
Nature
+1

“Set-to-set training using E-distance-style losses will improve OOD perturbation prediction.” [Speculation]

“Latent diffusion conditioned on (pre-state, perturbation) will yield calibrated stochastic perturbation outcomes.” [Speculation]

“LLM-JEPA-style tokenization will improve transcriptomic JEPA training.” [Speculation]

“Condition-level JEPA over sets of cells will be more robust than cell-level JEPA.” [Inference]

Gaps & Next Steps

Pick the initial 2–4 scPerturb datasets for Stage A based on: perturbation type diversity, presence of multiple donors/cell lines, and clean metadata. 
PMC

Define the OOD split protocol first (before modeling) to prevent leakage and to align with the “simple baselines win” benchmark concerns. 
Nature
+1

Decide the JEPA encoder backbone (Perceiver-style vs transformer) based on engineering constraints; both are defensible, but Perceiver-style is more directly aligned with GeneJEPA’s published implementation notes. 
GitHub

Implement prototype-transition first, then graduate to set-to-set losses only after you have stable JEPA training and reproducible baseline numbers. [Inference]