"""Train XGBoost model on concatenated RNA and clinical features for TCGA survival prediction.

This module tunes and trains an XGBoost classifier on the pre-assembled concatenated
feature matrix, evaluates discrimination on validation and test splits, and saves
model artifacts, predictions, metrics, and run metadata.

Pipeline:
1. Load assembled train/val/test concatenated feature matrices and targets.
2. Sanitize feature names (XGBoost does not allow [ ] < in column names).
3. Tune hyperparameters via randomized search with stratified CV on train only.
4. Train final model using best hyperparameters with early stopping on the val set.
5. Evaluate on val and test (ROC-AUC, AP).
6. Save model artifacts, predictions, metrics, and metadata.

Inputs:
- Assembled parquet files from assemble_dataset.py output directory.

Outputs:
- `xgboost_model.pkl`
- `predictions_val.parquet`
- `predictions_test.parquet`
- `metrics.json`
- `train_xgboost_metadata.json`
"""

from __future__ import annotations

import argparse
import json
import pickle
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    n_iter: int = 5,
    n_splits: int = 3,
    random_state: int = 42,
) -> dict:
    """Tune XGBoost hyperparameters via randomized search with stratified CV on train data only.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        param_grid: Hyperparameter search grid.
        n_iter: Number of random parameter combinations to try.
        n_splits: Number of CV folds.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary of best hyperparameters.
    """
    xgb = XGBClassifier(
        n_estimators=50,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )

    search = RandomizedSearchCV(
        xgb,
        param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best CV AUC: {search.best_score_:.3f}")
    print(f"Best params: {search.best_params_}")

    return search.best_params_


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: dict,
    random_state: int = 42,
) -> XGBClassifier:
    """Train final XGBoost model using best hyperparameters with early stopping on the val set.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix for early stopping.
        y_val: Validation labels for early stopping.
        best_params: Best hyperparameters from CV search.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted XGBClassifier.
    """
    model = XGBClassifier(
        **best_params,
        n_estimators=1000,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"Best n_estimators: {model.best_iteration}")
    print(f"Best val AUC (early stopping): {model.best_score:.3f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model on concatenated features.")
    parser.add_argument("--assembled-dir", required=True, help="Path to assembled dataset directory.")
    parser.add_argument("--outdir", required=True, help="Output directory for artifacts.")
    parser.add_argument("--n-iter", type=int, default=5, help="Number of hyperparameter search iterations.")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of CV folds for hyperparameter search.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    assembled_dir = Path(args.assembled_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train_df = pd.read_parquet(assembled_dir / "train/X_concat.parquet")
    X_val_df   = pd.read_parquet(assembled_dir / "val/X_concat.parquet")
    X_test_df  = pd.read_parquet(assembled_dir / "test/X_concat.parquet")

    y_train = pd.read_parquet(assembled_dir / "train/y.parquet")["y"]
    y_val   = pd.read_parquet(assembled_dir / "val/y.parquet")["y"]
    y_test  = pd.read_parquet(assembled_dir / "test/y.parquet")["y"]

    for split, X, y in [("train", X_train_df, y_train), ("val", X_val_df, y_val), ("test", X_test_df, y_test)]:
        assert X.index.equals(y.index), f"{split}: index mismatch between X and y"
        print(f"{split}: n={len(y)}, n_events={y.sum()}, features={X.shape[1]}")

    # Sanitize feature names — XGBoost does not allow [ ] < in column names
    X_train_df.columns = X_train_df.columns.str.replace(r"[\[\]<]", "_", regex=True)
    X_val_df.columns   = X_val_df.columns.str.replace(r"[\[\]<]", "_", regex=True)
    X_test_df.columns  = X_test_df.columns.str.replace(r"[\[\]<]", "_", regex=True)

    n_features = X_train_df.shape[1]

    # Tune hyperparameters
    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "colsample_bytree": [0.05, 0.1],
    }
    print("\n--- Hyperparameter search ---")
    best_params = tune_hyperparameters(
        X_train_df, y_train, param_grid,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    # Train final model
    print("\n--- Training final model ---")
    xgb_final = train_final_model(
        X_train_df, y_train, X_val_df, y_val,
        best_params, random_state=args.random_state,
    )

    # Generate predictions
    y_val_pred  = xgb_final.predict_proba(X_val_df)[:, 1]
    y_test_pred = xgb_final.predict_proba(X_test_df)[:, 1]

    # Validation checks
    assert len(y_val_pred)  == len(y_val),  f"val size mismatch: {len(y_val_pred)} vs {len(y_val)}"
    assert len(y_test_pred) == len(y_test), f"test size mismatch: {len(y_test_pred)} vs {len(y_test)}"
    assert len(set(X_train_df.index) & set(X_val_df.index))  == 0, "train/val overlap"
    assert len(set(X_train_df.index) & set(X_test_df.index)) == 0, "train/test overlap"
    assert len(set(X_val_df.index)   & set(X_test_df.index)) == 0, "val/test overlap"

    # Compute metrics
    val_roc_auc  = roc_auc_score(y_val, y_val_pred)
    val_ap       = average_precision_score(y_val, y_val_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred)
    test_ap      = average_precision_score(y_test, y_test_pred)

    print(f"\n--- Validation ---")
    print(f"ROC-AUC: {val_roc_auc:.3f}  AP: {val_ap:.3f}")
    print(f"\n--- Test ---")
    print(f"ROC-AUC: {test_roc_auc:.3f}  AP: {test_ap:.3f}")

    # Save model
    with open(outdir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_final, f)

    # Save predictions
    pd.DataFrame({
        "sample":    X_val_df.index,
        "y_true":    y_val.values,
        "y_pred_xgb": y_val_pred,
    }).to_parquet(outdir / "predictions_val.parquet", index=False)

    pd.DataFrame({
        "sample":    X_test_df.index,
        "y_true":    y_test.values,
        "y_pred_xgb": y_test_pred,
    }).to_parquet(outdir / "predictions_test.parquet", index=False)

    # Save metrics
    metrics = {
        "val":  {"roc_auc": round(val_roc_auc,  4), "ap": round(val_ap,  4)},
        "test": {"roc_auc": round(test_roc_auc, 4), "ap": round(test_ap, 4)},
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save metadata
    metadata = {
        "script_name": "train_xgboost.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_command": shlex.join(sys.argv),
        "input_files": {
            "assembled_dir": str(assembled_dir.resolve()),
        },
        "key_parameters_used": {
            "random_state": args.random_state,
            "model_type": "XGBClassifier",
            "param_grid": param_grid,
            "best_params": best_params,
            "n_iter": args.n_iter,
            "n_splits": args.n_splits,
            "n_estimators_max": 1000,
            "early_stopping_rounds": 50,
            "best_n_estimators": int(xgb_final.best_iteration),
        },
        "dataset_statistics": {
            "n_samples_train": int(len(y_train)),
            "n_samples_val": int(len(y_val)),
            "n_samples_test": int(len(y_test)),
            "n_events_train": int(y_train.sum()),
            "n_events_val": int(y_val.sum()),
            "n_events_test": int(y_test.sum()),
            "n_features": int(n_features),
            "val_roc_auc": round(val_roc_auc, 4),
            "val_ap": round(val_ap, 4),
            "test_roc_auc": round(test_roc_auc, 4),
            "test_ap": round(test_ap, 4),
        },
        "output_files": {
            "xgboost_model": str(outdir / "xgboost_model.pkl"),
            "predictions_val": str(outdir / "predictions_val.parquet"),
            "predictions_test": str(outdir / "predictions_test.parquet"),
            "metrics": str(outdir / "metrics.json"),
        },
    }
    (outdir / "train_xgboost_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nArtifacts saved to {outdir}")


if __name__ == "__main__":
    main()
