# Project Planning Notes

Working notes for future extensions and reference. Not intended as project documentation.

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

**Fusion Strategies**
1. Concatenation + MLP (no attention)
2. Attention / modality-gating fusion: learn per-patient weight between RNA and clinical embeddings.

Train end-to-end for 5-year survival classification.

### 6. Analysis and Model Selection

Compare all models (Clinical LR, RNA LR, XGBoost, Concat MLP, Attention fusion) on ROC-AUC, AP, and top-20% high-risk capture.

Risk stratification on test set: Kaplan–Meier survival curves and log-rank test to show decision-oriented utility beyond AUC.

Interpretability: SHAP feature importance for XGBoost; learned modality weight distribution for the attention model.

Select final model based on discrimination, high-risk capture, interpretability, and deployment simplicity.
