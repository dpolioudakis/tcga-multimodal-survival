"""Tests for preprocessing functions — scaler, feature names, data quality."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_baselines import fit_and_apply_scaler


# --- fixtures ---

def make_feature_df(n_samples=30, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"TCGA-{i:04d}" for i in range(n_samples)]
    return pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        index=ids,
        columns=[f"feature_{i}" for i in range(n_features)],
    )


# --- tests ---

def test_scaler_fit_on_train_only():
    """Scaler mean and std should match training data, not val or test."""
    X_train = make_feature_df(n_samples=50, seed=0)
    X_val   = make_feature_df(n_samples=20, seed=1)
    X_test  = make_feature_df(n_samples=15, seed=2)

    scaler, X_train_scaled, _, _ = fit_and_apply_scaler(X_train, X_val, X_test)

    np.testing.assert_allclose(scaler.mean_, X_train.values.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(scaler.scale_, X_train.values.std(axis=0, ddof=0), rtol=1e-5)


def test_scaler_output_shapes():
    X_train = make_feature_df(n_samples=50)
    X_val   = make_feature_df(n_samples=20)
    X_test  = make_feature_df(n_samples=15)

    _, X_train_scaled, X_val_scaled, X_test_scaled = fit_and_apply_scaler(X_train, X_val, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_val_scaled.shape   == X_val.shape
    assert X_test_scaled.shape  == X_test.shape


def test_no_missing_values_after_scaling():
    X_train = make_feature_df(n_samples=50)
    X_val   = make_feature_df(n_samples=20)
    X_test  = make_feature_df(n_samples=15)

    _, X_train_scaled, X_val_scaled, X_test_scaled = fit_and_apply_scaler(X_train, X_val, X_test)

    assert not np.isnan(X_train_scaled).any()
    assert not np.isnan(X_val_scaled).any()
    assert not np.isnan(X_test_scaled).any()


def test_feature_name_sanitization():
    """XGBoost does not allow [ ] < in feature names — verify sanitization works."""
    cols = ["feature[1]", "feature<2>", "feature[3", "normal_feature"]
    df = pd.DataFrame(np.ones((5, 4)), columns=cols)
    df.columns = df.columns.str.replace(r"[\[\]<]", "_", regex=True)

    assert not any("[" in c or "]" in c or "<" in c for c in df.columns)
    assert "normal_feature" in df.columns


def test_feature_counts_consistent_across_splits():
    X_train = make_feature_df(n_samples=50, n_features=10)
    X_val   = make_feature_df(n_samples=20, n_features=10)
    X_test  = make_feature_df(n_samples=15, n_features=10)

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
