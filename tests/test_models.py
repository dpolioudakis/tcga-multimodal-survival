"""Tests for model I/O — prediction range, shape, and expected attributes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_multimodal import RNAEncoder, ClinicalEncoder, ConcatFusionModel, AttentionFusionModel
from sklearn.linear_model import LogisticRegression


# --- fixtures ---

N_SAMPLES  = 30
N_RNA      = 50
N_CLIN     = 10


def make_tensors(n_samples=N_SAMPLES, n_rna=N_RNA, n_clin=N_CLIN, seed=42):
    rng = np.random.default_rng(seed)
    rna_t  = torch.tensor(rng.standard_normal((n_samples, n_rna)),  dtype=torch.float32)
    clin_t = torch.tensor(rng.standard_normal((n_samples, n_clin)), dtype=torch.float32)
    y_t    = torch.tensor(rng.integers(0, 2, n_samples),            dtype=torch.float32)
    return rna_t, clin_t, y_t


def make_logistic_model():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10))
    y = rng.integers(0, 2, 50)
    model = LogisticRegression()
    model.fit(X, y)
    return model


def make_concat_model():
    rna_enc  = RNAEncoder(input_dim=N_RNA)
    clin_enc = ClinicalEncoder(input_dim=N_CLIN)
    return ConcatFusionModel(rna_enc, clin_enc)


def make_attn_model():
    rna_enc  = RNAEncoder(input_dim=N_RNA)
    clin_enc = ClinicalEncoder(input_dim=N_CLIN)
    return AttentionFusionModel(rna_enc, clin_enc, embed_dim=64)


# --- logistic regression tests ---

def test_logistic_predictions_in_range():
    model = make_logistic_model()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 10))
    preds = model.predict_proba(X)[:, 1]
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def test_logistic_prediction_count_matches_input():
    model = make_logistic_model()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 10))
    preds = model.predict_proba(X)[:, 1]
    assert len(preds) == 20


def test_logistic_has_predict_proba():
    model = make_logistic_model()
    assert hasattr(model, "predict_proba")


# --- concat fusion model tests ---

def test_concat_model_predictions_in_range():
    model = make_concat_model()
    rna_t, clin_t, _ = make_tensors()
    model.eval()
    with torch.no_grad():
        preds = model(rna_t, clin_t).numpy()
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def test_concat_model_prediction_count_matches_input():
    model = make_concat_model()
    rna_t, clin_t, _ = make_tensors()
    model.eval()
    with torch.no_grad():
        preds = model(rna_t, clin_t).numpy()
    assert len(preds) == N_SAMPLES


def test_concat_model_has_head():
    model = make_concat_model()
    assert hasattr(model, "head")


# --- attention fusion model tests ---

def test_attn_model_predictions_in_range():
    model = make_attn_model()
    rna_t, clin_t, _ = make_tensors()
    model.eval()
    with torch.no_grad():
        preds = model(rna_t, clin_t).numpy()
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def test_attn_model_prediction_count_matches_input():
    model = make_attn_model()
    rna_t, clin_t, _ = make_tensors()
    model.eval()
    with torch.no_grad():
        preds = model(rna_t, clin_t).numpy()
    assert len(preds) == N_SAMPLES


def test_attn_model_has_attention():
    model = make_attn_model()
    assert hasattr(model, "attention")


def test_attn_weights_sum_to_one():
    """Softmax attention weights over two modalities should sum to 1 per patient."""
    model = make_attn_model()
    rna_t, clin_t, _ = make_tensors()
    model.eval()
    with torch.no_grad():
        rna_emb  = model.rna_encoder(rna_t)
        clin_emb = model.clin_proj(model.clin_encoder(clin_t))
        stacked  = torch.stack([rna_emb, clin_emb], dim=1)
        weights  = torch.softmax(model.attention(stacked), dim=1).squeeze(-1)
    np.testing.assert_allclose(weights.sum(dim=1).numpy(), np.ones(N_SAMPLES), atol=1e-5)
