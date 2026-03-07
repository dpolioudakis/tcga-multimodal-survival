# Guide to Multimodal Survival Modeling with TCGA

## Project Roadmap: Multimodal 5-Year Survival Prediction (TCGA)

## Objective

Build and rigorously evaluate multimodal models to predict 5-year overall survival for a single TCGA cancer cohort using RNA-seq and clinical features. Emphasize leakage control, strong baselines, attention-based fusion, calibration, and decision-oriented evaluation.

---

## 1. Problem Definition

- Select one TCGA cancer type (e.g., LUAD or BRCA).
- Define binary label:
  - Survived ≥ 5 years (1825 days) vs not.
- Exclude samples with insufficient follow-up.
- Clearly document censoring assumptions.

Frame as:
- Risk stratification problem (not just classification).

Primary metrics:
- ROC-AUC
- Average Precision (AP)
- Calibration (Brier score + reliability curve)
- Performance at fixed high-risk threshold (e.g., top 20%)

---

## 2. Data Assembly & Leakage Control

Modalities:
- RNA-seq gene expression (normalized)
- Clinical variables (age, stage, sex, etc.)

Steps:
- Align samples across modalities.
- Remove duplicates.
- Handle missing clinical data (imputation strategy documented).
- Standardize gene expression (fit on train only).
- Single train / validation / test split:
  - Stratified by outcome.
  - All preprocessing fit only on training set.

Explicitly verify:
- No data leakage from test into training.
- No leakage via normalization or feature filtering.

---

## 3. Baseline Models (Performance Floor)

1. Clinical-only logistic regression.
2. RNA-only logistic regression (L2 regularization).

Evaluate on validation and test:
- ROC-AUC
- AP
- Calibration curves
- Risk-tier separation

Purpose:
- Establish interpretable baseline.
- Quantify incremental value of RNA.

---

## 4. Strong Tabular Model (Nonlinear Benchmark)

Model:
- XGBoost on concatenated RNA + clinical features.

Include:
- Cross-validation on training set.
- Hyperparameter tuning.
- Early stopping.
- Save best parameters.

Evaluate:
- ROC-AUC
- AP
- Calibration
- Top-20% risk capture
- Confusion-style summary at threshold

Purpose:
- Measure nonlinear gains over logistic regression.
- Provide SHAP-based interpretability baseline.

---

## 5. Multimodal Deep Learning

### Encoders

- RNA → MLP encoder → embedding
- Clinical → MLP encoder → embedding

### Fusion Strategies (Ablation Study)

1. Concatenation + MLP (no attention)
2. Attention / modality-gating fusion:
   - Learn per-patient weight between RNA and clinical embeddings.

Train end-to-end for 5-year survival classification.

---

## 6. Ablation Comparison

Explicitly compare:

- Clinical-only LR
- RNA-only LR
- XGBoost (RNA + clinical)
- Deep concat model
- Attention fusion model

For each:
- ROC-AUC
- AP
- Calibration
- Top-20% high-risk capture

Goal:
Demonstrate whether attention meaningfully improves over naive fusion.

---

## 7. Risk Stratification Analysis

On test set:

- Define high-risk group (top 20% predicted risk).
- Generate Kaplan–Meier survival curves.
- Perform log-rank test.
- Compare separation across models.

Purpose:
Show decision-oriented utility beyond AUC.

---

## 8. Interpretability

XGBoost:
- SHAP feature importance.
- Confirm biologically plausible drivers (e.g., stage, age).

Attention Model:
- Inspect learned modality weights distribution.
- Show example patients where:
  - Clinical dominates.
  - RNA dominates.
- Sanity check against clinical intuition.

---

## 9. Final Model Selection

Compare models based on:

- Discrimination (AUC, AP)
- Calibration
- High-risk capture
- Interpretability
- Deployment simplicity

Explicitly justify final recommendation.

---

## Deliverables

- Data preprocessing notebook (reproducible pipeline).
- Training script (modular, configurable).
- Evaluation notebook/report.
- Clean results table.
- Clear README including:
  - Problem framing
  - Leakage safeguards
  - Model comparison
  - Limitations
  - Future extensions (e.g., Cox model, additional modalities)

---

## What This Demonstrates

- Multimodal modeling capability.
- Practical attention-based fusion.
- Strong ML fundamentals (baselines, CV, calibration).
- Decision-driven evaluation.
- Biotech-relevant applied machine learning expertise.




---


### Why BRCA was chosen
- Decent cohort size (~1,200 patients before filtering)
- High overlap across RNA, clinical, and survival data
- Sufficient number of primary tumor samples after QC
- Adequate event count for 5-year survival modeling (~100 events)
- Biologically heterogeneous disease, making multimodal integration meaningful
- Well-characterized TCGA cohort commonly used in benchmarking


### 5-Year Overall Survival Definition
- Primary endpoint: 5-year overall survival (≥ 1825 days)
- Event (label = 1): death within 5 years
- Survivor (label = 0): alive with ≥ 5 years of follow-up
- Excluded: alive with < 5 years of follow-up (censored before cutoff)
**Rationale**
- Minimizes label ambiguity from censoring
- Provides a clinically meaningful endpoint
- Maintains adequate event rate (~33%)
- Preserves sufficient cohort size (n ≈ 301) for stable modeling


### Why We Use TCGA TPM
- TCGA log2(count + 1) values are **not library-size normalized**, so sequencing depth differences remain.
- TPM is **library-size normalized**, reducing global shifts across patients.
- Gene-length normalization in TPM does not negatively affect within-gene, cross-patient modeling.
- TPM ensures a consistent total expression scale across samples.
- After log transformation and train-only scaling, TPM provides a stable input for ML models.
**What Would Be Fully Optimal (If Reprocessing From Raw Data)**
- Start from raw gene-level counts.
- Perform library-size normalization using size factors (e.g., DESeq2 or TMM).
- Apply a variance-stabilizing transform (VST or log after normalization).
- Split data by patient before any statistical filtering.
- Fit gene filtering and scaling steps on the training set only.





Pre-split (full dataset, label-agnostic)
Structural integrity
Genes x samples orientation
One tumor sample per patient
No duplicated barcodes
Sample-level QC (technical only)
Per-sample mean and median expression
Per-sample SD
Approximate detected genes (genes > 0)
Percent zeros
Remove clear technical failures using predefined thresholds
Distribution sanity
Overlay density plots of expression across samples
Boxplots ordered by median
Confirm no global compression or obvious broken samples





Post-split (fit on train only)
Gene filtering
Low-prevalence filter (e.g., expressed in ≥ X% samples)
Variance filter or top N variable genes
Scaling
Fit gene-wise scaler on train
Apply to val/test
PCA diagnostics
Run PCA on train
Project val/test without refitting





## Project Structure

### 1. `01_data_prep.ipynb`

**Purpose:** Structural cleaning + EDA + rule definition  

**Contains:**
- Raw data loading  
- Sample alignment  
- Structural QC  
- Missingness analysis  
- Distribution checks  
- Cohort definition  
- Explicit filtering rules (constants)

**Ends with:**
- Saved cleaned cohort manifest  
- Saved rule definitions (e.g., JSON or Python constants)  
- No train/test split yet  

**This notebook answers:**  
“Is the dataset coherent and what rules will we apply?”

### 2. `02_split_and_features.py` (script, not notebook)

**Purpose:** Deterministic preprocessing  

- Load cleaned cohort  
- Train/test split  
- Fit imputation on train  
- Fit scaling on train  
- Fit gene filtering on train  
- Apply to val/test  
- Save feature matrices  

No plotting. Pure pipeline.

### 3. `03_modeling.py`

- Baselines  
- Cross-validation tuning  
- Save trained models  

### 4. `04_evaluation.ipynb`

- ROC-AUC / Average Precision  
- Calibration  
- Kaplan–Meier curves  
- SHAP  
- Model comparison


### Directory structure
- data/
    - raw/
        - Immutable source data
    - interim/
        - Dataset definition
            - Cleaned, aligned, QC’d
            - Cohort defined
            - No train/test split yet
    - processed/
        - Experiment definition
            - Split applied
            - Scaling/imputation fit
            - Model-ready matrices
        - splits/
            - train_ids.csv
            - val_ids.csv
            - test_ids.csv
        - model_inputs/
            - train/
            - val/
            - test/




Finalize split as a reproducible artifact
Add split_metadata.json (seed, fractions, stratify col, cohort size, timestamp, input manifest hash/path).
Run create_split.py once to generate the canonical train_ids/val_ids/test_ids.
Clinical missingness handling (train-fit only)
Decide per-column rules: drop vs impute.
Fit imputers on train only, apply to val/test.
Optionally add missingness indicators for key vars.
RNA preprocessing (train-fit only)
If you do gene filtering (low-expression/variance), fit thresholds on train only.
Fit StandardScaler on train only, transform val/test.
Data assembly
Assemble X_clin, X_rna, y strictly by split IDs; assert no ID drift/leakage.