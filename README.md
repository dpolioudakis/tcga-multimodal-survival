# Multimodal Survival Modeling with TCGA

A step-by-step guide to building and evaluating multimodal models for 5-year overall survival prediction using TCGA BRCA RNA-seq and clinical data. Covers leakage control, strong baselines, attention-based fusion, and decision-oriented evaluation.

---

## Results

| Model | Val AUC | Val AP | Test AUC | Test AP |
|---|---|---|---|---|
| Clinical LR | 0.884 | 0.826 | 0.786 | 0.675 |
| RNA LR | 0.709 | 0.575 | 0.624 | 0.479 |
| XGBoost (concat) | — | — | — | — |
| Deep concat MLP | — | — | — | — |
| Attention fusion | — | — | — | — |

Top-20% risk capture (test set):

| Model | High-risk event rate | Low-risk event rate |
|---|---|---|
| Clinical LR | 0.667 | 0.111 |
| RNA LR | 0.556 | 0.222 |
| XGBoost (concat) | — | — |
| Deep concat MLP | — | — |
| Attention fusion | — | — |

---

## Setup

```bash
conda env create -f environment.yml
conda activate tcga-survival
```

Data download (requires GDC access):

```bash
bash scripts/download_data.sh
```

---

## Reproducing the Pipeline

Run steps in order. Each script writes outputs consumed by the next.

```bash
# 1. Create train/val/test split
python scripts/create_split.py

# 2. Preprocess clinical features
python scripts/preprocess_clinical.py

# 3. Preprocess RNA features
python scripts/preprocess_rna.py

# 4. Assemble model-ready datasets
python scripts/assemble_dataset.py

# 5. Train baseline models
python scripts/train_baselines.py \
    --assembled-dir data/processed/assembled \
    --outdir models/baselines

# 6. Train XGBoost benchmark
# python scripts/train_xgboost.py \
#     --assembled-dir data/processed/assembled \
#     --outdir models/xgboost
```

Each script also has a corresponding development notebook in `notebooks/`.

---

## Guide Structure

This project is organized as a teaching guide. Each step introduces a concept, motivates the design decision, and implements it in a notebook before refactoring into a reusable script.

1. **Data preparation** (`01_data_prep.ipynb`) — cohort definition, QC, modality alignment
2. **Split creation** (`02_create_split.ipynb`) — stratified train/val/test split, leakage safeguards
3. **Clinical preprocessing** (`03_preprocess_clinical.ipynb`) — imputation, encoding, leakage checks; all parameters fit on train only
4. **RNA preprocessing** (`04_preprocess_rna.ipynb`) — filtering, log transform, train-only scaling
5. **Dataset assembly** (`05_assemble_dataset.ipynb`) — align modalities, construct targets, validate
6. **Baseline models** (`06_train_baselines.ipynb`) — clinical-only and RNA-only logistic regression
7. **XGBoost benchmark** (`07_train_xgboost.ipynb`) — nonlinear multimodal benchmark with SHAP
8. **Deep learning** (planned) — MLP encoders, concat fusion, attention fusion

### Ablation design

Each model in the results table changes one thing:

- **Clinical LR → RNA LR**: modality value
- **RNA LR → XGBoost (concat)**: nonlinearity + naive fusion
- **XGBoost → Deep concat MLP**: deep representations
- **Deep concat → Attention fusion**: learned modality weighting

The goal is to isolate whether attention meaningfully improves over naive concatenation.

---

## Directory Structure

```
data/
├── raw/                        # Immutable source files (never modified)
├── interim/                    # Cohort definition: sample IDs, feature manifests, preprocessing parameters
├── processed/                  # Model-ready outputs (split applied, preprocessing fit on train)
│   ├── splits/                 # Train/val/test sample ID CSVs
│   ├── clinical/               # Preprocessed clinical feature matrices (train/, val/, test/)
│   ├── rna/                    # Preprocessed RNA feature matrices (train/, val/, test/)
│   └── assembled/              # Aligned, merged feature matrices ready for modeling (train/, val/, test/)
└── tests/                      # Ephemeral outputs from notebook dev/smoke tests

models/
└── baselines/                  # Fitted model artifacts, predictions, and metrics

notebooks/                      # Step-by-step development notebooks (01–07)
scripts/                        # Reusable pipeline modules (CLI entry points)
reports/                        # Figures and analysis outputs
```

---

## Design Notes

### Why BRCA
- Decent cohort size (~1,200 patients before filtering)
- High overlap across RNA, clinical, and survival data
- Sufficient primary tumor samples after QC
- Adequate event count for 5-year survival modeling (~100 events)
- Biologically heterogeneous disease, making multimodal integration meaningful
- Well-characterized TCGA cohort commonly used in benchmarking

### 5-Year Overall Survival Definition
- **Event (label = 1):** death within 5 years
- **Survivor (label = 0):** alive with ≥ 5 years of follow-up
- **Excluded:** alive with < 5 years follow-up (censored before cutoff)

Rationale: minimizes label ambiguity from censoring, provides a clinically meaningful endpoint, maintains adequate event rate (~33%), and preserves sufficient cohort size (n ≈ 301).

### Why TPM for RNA
TCGA log2(count + 1) values are not library-size normalized, so sequencing depth differences remain. TPM is library-size normalized, reducing global shifts across patients. After log transformation and train-only z-scoring, TPM provides a stable input for ML models.

What would be fully optimal (if reprocessing from raw): start from raw counts, normalize with DESeq2/TMM size factors, apply VST, split before any statistical filtering, fit all scaling on train only.

### Leakage Controls
- Train/val/test split is created once and saved; all preprocessing parameters are fit on the train partition only.
- Clinical imputation medians, OHE categories, and RNA scaling parameters are derived from training data and applied unchanged to val and test.
- Target leakage columns (`vital_status`, `treatment_or_therapy`) are explicitly dropped before any feature encoding.
- Formal leakage check in `03_preprocess_clinical.ipynb` correlates candidate features with the outcome label before finalizing the feature set.

---

## Roadmap

### 1. Problem Definition

- Select one TCGA cancer type (e.g., LUAD or BRCA).
- Define binary label: survived ≥ 5 years (1825 days) vs not.
- Exclude samples with insufficient follow-up.
- Clearly document censoring assumptions.
- Frame as a risk stratification problem (not just classification).

Primary metrics:
- ROC-AUC
- Average Precision (AP)
- Calibration (Brier score + reliability curve)
- Performance at fixed high-risk threshold (e.g., top 20%)

### 2. Data Assembly & Leakage Control

Modalities:
- RNA-seq gene expression (normalized)
- Clinical variables (age, stage, sex, etc.)

Steps:
- Align samples across modalities.
- Remove duplicates.
- Handle missing clinical data (imputation strategy documented).
- Standardize gene expression (fit on train only).
- Single train / validation / test split: stratified by outcome, all preprocessing fit only on training set.

Explicitly verify:
- No data leakage from test into training.
- No leakage via normalization or feature filtering.

### 3. Baseline Models (Performance Floor)

1. Clinical-only logistic regression.
2. RNA-only logistic regression (L2 regularization).

Evaluate on validation and test: ROC-AUC, AP, calibration curves, risk-tier separation.

Purpose:
- Establish interpretable baseline.
- Quantify incremental value of RNA.

### 4. Strong Tabular Model (Nonlinear Benchmark)

Model: XGBoost on concatenated RNA + clinical features.

Include:
- Cross-validation on training set.
- Hyperparameter tuning.
- Early stopping.
- Save best parameters.

Evaluate: ROC-AUC, AP, calibration, top-20% risk capture, confusion-style summary at threshold.

Purpose:
- Measure nonlinear gains over logistic regression.
- Provide SHAP-based interpretability baseline.

### 5. Multimodal Deep Learning

**Encoders**
- RNA → MLP encoder → embedding
- Clinical → MLP encoder → embedding

**Fusion Strategies (Ablation Study)**
1. Concatenation + MLP (no attention)
2. Attention / modality-gating fusion: learn per-patient weight between RNA and clinical embeddings.

Train end-to-end for 5-year survival classification.

### 6. Ablation Comparison

Explicitly compare: Clinical-only LR, RNA-only LR, XGBoost (RNA + clinical), Deep concat model, Attention fusion model.

For each: ROC-AUC, AP, calibration, top-20% high-risk capture.

Goal: demonstrate whether attention meaningfully improves over naive fusion.

### 7. Risk Stratification Analysis

On test set:
- Define high-risk group (top 20% predicted risk).
- Generate Kaplan–Meier survival curves.
- Perform log-rank test.
- Compare separation across models.

Purpose: show decision-oriented utility beyond AUC.

### 8. Interpretability

XGBoost:
- SHAP feature importance.
- Confirm biologically plausible drivers (e.g., stage, age).

Attention Model:
- Inspect learned modality weights distribution.
- Show example patients where clinical dominates and where RNA dominates.
- Sanity check against clinical intuition.

### 9. Final Model Selection

Compare models based on: discrimination (AUC, AP), calibration, high-risk capture, interpretability, deployment simplicity.

Explicitly justify final recommendation.

---

## Deliverables

- Data preprocessing notebook (reproducible pipeline).
- Training script (modular, configurable).
- Evaluation notebook/report.
- Clean results table.
- Clear README including problem framing, leakage safeguards, model comparison, limitations, and future extensions (e.g., Cox model, additional modalities).

## What This Demonstrates

- Multimodal modeling capability.
- Practical attention-based fusion.
- Strong ML fundamentals (baselines, CV, calibration).
- Decision-driven evaluation.
- Biotech-relevant applied machine learning expertise.
