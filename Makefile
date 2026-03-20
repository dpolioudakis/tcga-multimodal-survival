# Makefile for TCGA multimodal survival pipeline
# Run `make help` to see available targets.

# --- paths ---
RAW_DIR        := data/raw
INTERIM_DIR    := data/interim
SPLITS_DIR     := data/processed/splits
CLINICAL_DIR   := data/processed/clinical
RNA_DIR        := data/processed/rna
ASSEMBLED_DIR  := data/processed/assembled

MODELS_BASELINES  := models/baselines
MODELS_XGBOOST    := models/xgboost
MODELS_MULTIMODAL := models/multimodal

# --- parameters ---
RANDOM_STATE := 42

.PHONY: help test pipeline split preprocess-clinical preprocess-rna assemble \
        train-baselines train-xgboost train-multimodal clean

help:
	@echo "Available targets:"
	@echo "  make pipeline           Run the full pipeline end to end"
	@echo "  make test               Run unit tests"
	@echo "  make split              Create train/val/test split"
	@echo "  make preprocess-clinical  Preprocess clinical features"
	@echo "  make preprocess-rna     Preprocess RNA features"
	@echo "  make assemble           Assemble model-ready datasets"
	@echo "  make train-baselines    Train logistic regression baselines"
	@echo "  make train-xgboost      Train XGBoost benchmark"
	@echo "  make train-multimodal   Train deep learning fusion models"
	@echo "  make clean              Remove all generated outputs"

# --- run full pipeline ---
pipeline: split \
          preprocess-clinical \
          preprocess-rna \
          assemble \
          train-baselines \
          train-xgboost \
          train-multimodal
	@echo "Pipeline complete."

# --- run tests ---
test:
	python -m pytest tests/ -v

# --- pipeline steps ---
split:
	python scripts/create_split.py \
		--sample-ids-path $(INTERIM_DIR)/sample_ids.csv \
		--survival-path   $(RAW_DIR)/TCGA-BRCA.survival.tsv.gz \
		--event-col       OS \
		--outdir          $(SPLITS_DIR) \
		--seed            $(RANDOM_STATE)

preprocess-clinical:
	python scripts/preprocess_clinical.py \
		--clinical-path    $(RAW_DIR)/TCGA-BRCA.clinical.tsv.gz \
		--feature-path     $(INTERIM_DIR)/clinical_features.csv \
		--sample-ids-path  $(INTERIM_DIR)/sample_ids.csv \
		--params-path      $(INTERIM_DIR)/clinical_preprocessing_params.json \
		--split-dir        $(SPLITS_DIR) \
		--outdir           $(CLINICAL_DIR)

preprocess-rna:
	python scripts/preprocess_rna.py \
		--rna-path         $(RAW_DIR)/TCGA-BRCA.star_tpm.tsv.gz \
		--sample-ids-path  $(INTERIM_DIR)/sample_ids.csv \
		--params-path      $(INTERIM_DIR)/rna_preprocessing_params.json \
		--split-dir        $(SPLITS_DIR) \
		--outdir           $(RNA_DIR)

assemble:
	python scripts/assemble_dataset.py \
		--clinical-dir    $(CLINICAL_DIR) \
		--rna-dir         $(RNA_DIR) \
		--survival-path   $(RAW_DIR)/TCGA-BRCA.survival.tsv.gz \
		--event-col       OS \
		--outdir          $(ASSEMBLED_DIR)

train-baselines:
	python scripts/train_baselines.py \
		--assembled-dir $(ASSEMBLED_DIR) \
		--outdir        $(MODELS_BASELINES) \
		--random-state  $(RANDOM_STATE)

train-xgboost:
	python scripts/train_xgboost.py \
		--assembled-dir $(ASSEMBLED_DIR) \
		--outdir        $(MODELS_XGBOOST) \
		--random-state  $(RANDOM_STATE)

train-multimodal:
	python scripts/train_multimodal.py \
		--assembled-dir $(ASSEMBLED_DIR) \
		--outdir        $(MODELS_MULTIMODAL) \
		--random-state  $(RANDOM_STATE)

# --- clean generated outputs ---
# WARNING: removes all processed data and trained model artifacts.
# Only use to verify the pipeline reproduces from scratch.
clean:
	rm -rf data/processed models/baselines models/xgboost models/multimodal
	@echo "Cleaned generated outputs."
