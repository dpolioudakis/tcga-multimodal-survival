# Multimodal Survival Modeling with TCGA

A step-by-step guide to building and evaluating survival models on TCGA BRCA RNA-seq and clinical data — from logistic regression through XGBoost to attention-based deep learning.

---

## Results

| Model | Val AUC | Val AP | Test AUC | Test AP |
|---|---|---|---|---|
| Majority class | 0.500 | 0.326 | 0.500 | 0.318 |
| Clinical LR | 0.884 | 0.826 | 0.786 | 0.675 |
| RNA LR | 0.719 | 0.582 | 0.614 | 0.474 |
| XGBoost (concat) | 0.786 | 0.687 | 0.671 | 0.475 |
| Deep concat MLP | 0.594 | 0.549 | 0.679 | 0.541 |
| Attention fusion | 0.645 | 0.514 | 0.538 | 0.380 |

Top-20% risk capture (test set):

| Model | High-risk event rate | Low-risk event rate |
|---|---|---|
| Majority class | N/A | N/A |
| Clinical LR | 0.667 | 0.111 |
| RNA LR | 0.556 | 0.222 |
| XGBoost (concat) | 0.444 | 0.000 |
| Deep concat MLP | 0.444 | 0.333 |
| Attention fusion | 0.667 | 0.000 |

Clinical LR is the strongest model across discrimination and risk stratification. Deep learning models underperform at this sample size (n=203 training samples), consistent with the broader literature on deep learning in small biomedical cohorts. The deep learning models are included as architectural demonstrations of multimodal fusion rather than optimized production models. The project emphasizes rigorous leakage control, reproducible pipelines, and honest evaluation over maximizing reported metrics.

---

## Setup

```bash
conda env create -f environment.yml  # Python 3.10
conda activate tcga-survival
```

Data download (public, no credentials required):

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

## Project Structure

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
├── baselines/                  # Logistic regression artifacts, predictions, and metrics
├── xgboost/                    # XGBoost artifacts, predictions, SHAP values, and metrics
└── multimodal/                 # Concat MLP and attention fusion artifacts, predictions, and metrics

notebooks/                      # Step-by-step development notebooks (01–09)
scripts/                        # Reusable pipeline modules (CLI entry points)
reports/                        # Figures and analysis outputs
```

Each notebook has a corresponding script in `scripts/` — the notebook is the development and guide artifact, the script is the reproducible CLI entry point.

1. **Data preparation** (`01_data_prep.ipynb`) — cohort definition, QC, modality alignment
2. **Split creation** (`02_create_split.ipynb`) — stratified train/val/test split, leakage safeguards
3. **Clinical preprocessing** (`03_preprocess_clinical.ipynb`) — imputation, encoding, leakage checks; all parameters fit on train only
4. **RNA preprocessing** (`04_preprocess_rna.ipynb`) — filtering, log transform, train-only scaling
5. **Dataset assembly** (`05_assemble_dataset.ipynb`) — align modalities, construct targets, validate
6. **Baseline models** (`06_train_baselines.ipynb`) — clinical-only and RNA-only logistic regression
7. **XGBoost benchmark** (`07_train_xgboost.ipynb`) — nonlinear multimodal benchmark with SHAP
8. **Deep learning** (`08_train_multimodal.ipynb`) — MLP encoders, concatenation fusion, attention fusion
9. **Analysis and model selection** (`09_analysis_and_model_selection.ipynb`) — cross-model comparison, Kaplan–Meier survival curves, SHAP interpretability, final recommendation

---

## Design Decisions

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

### Deep Learning Hyperparameter Tuning
Hyperparameter tuning was applied to XGBoost but not the deep learning models. With n=203 training samples, the primary constraint is sample size — not architecture or regularization choices. A grid search over dropout rates, learning rates, or embedding dimensions would not address the fundamental parameter-to-sample ratio problem, and would be prohibitively slow to run on a local machine. The deep learning models are included as architectural demonstrations of multimodal fusion, not as optimized production models.

### Model Comparison Design

Each model in the results table represents a step up in complexity:

- **Clinical LR**: interpretable baseline using well-established clinical risk factors
- **RNA LR**: tests whether gene expression adds signal beyond clinical features
- **XGBoost (concat)**: nonlinear model on combined modalities
- **Deep concat MLP**: learned representations via neural encoders
- **Attention fusion**: per-patient modality weighting via learned attention

This is a model comparison rather than a strict ablation. Each step changes multiple things at once (architecture, training procedure, feature set), so individual contributions cannot be cleanly isolated. The progression is designed to answer a practical question: at what point does added complexity stop paying off?

### Leakage Controls
- Train/val/test split is created once and saved; all preprocessing parameters are fit on the train partition only.
- Clinical imputation medians, OHE categories, and RNA scaling parameters are derived from training data and applied unchanged to val and test.
- Target leakage columns (`vital_status`, `treatment_or_therapy`) are explicitly dropped before any feature encoding.
- Formal leakage check in `03_preprocess_clinical.ipynb` correlates candidate features with the outcome label before finalizing the feature set.


