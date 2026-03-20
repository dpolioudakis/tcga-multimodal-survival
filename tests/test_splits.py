"""Tests for split integrity — leakage and sample ID checks."""

import numpy as np
import pandas as pd
import pytest


# --- fixtures ---

def make_split_ids(n_train=50, n_val=20, n_test=15):
    train_ids = [f"TCGA-{i:04d}" for i in range(n_train)]
    val_ids   = [f"TCGA-{i:04d}" for i in range(n_train, n_train + n_val)]
    test_ids  = [f"TCGA-{i:04d}" for i in range(n_train + n_val, n_train + n_val + n_test)]
    return train_ids, val_ids, test_ids


def make_feature_df(ids, n_features=10):
    return pd.DataFrame(
        np.random.rand(len(ids), n_features),
        index=ids,
        columns=[f"feature_{i}" for i in range(n_features)],
    )


def make_target(ids, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 2, len(ids)), index=ids, name="y")


# --- tests ---

def test_no_train_val_overlap():
    train_ids, val_ids, _ = make_split_ids()
    assert len(set(train_ids) & set(val_ids)) == 0


def test_no_train_test_overlap():
    train_ids, _, test_ids = make_split_ids()
    assert len(set(train_ids) & set(test_ids)) == 0


def test_no_val_test_overlap():
    _, val_ids, test_ids = make_split_ids()
    assert len(set(val_ids) & set(test_ids)) == 0


def test_split_sizes_sum_to_total():
    n_train, n_val, n_test = 50, 20, 15
    train_ids, val_ids, test_ids = make_split_ids(n_train, n_val, n_test)
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_train + n_val + n_test


def test_no_duplicate_sample_ids():
    train_ids, val_ids, test_ids = make_split_ids()
    all_ids = train_ids + val_ids + test_ids
    assert len(all_ids) == len(set(all_ids))


def test_feature_index_matches_target():
    train_ids, _, _ = make_split_ids()
    X = make_feature_df(train_ids)
    y = make_target(train_ids)
    assert X.index.equals(y.index)


def test_feature_counts_consistent_across_splits():
    train_ids, val_ids, test_ids = make_split_ids()
    X_train = make_feature_df(train_ids, n_features=10)
    X_val   = make_feature_df(val_ids,   n_features=10)
    X_test  = make_feature_df(test_ids,  n_features=10)
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
