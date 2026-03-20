"""Train baseline logistic regression models for TCGA multimodal survival prediction.

This module fits L2-regularized logistic regression models on clinical and RNA
features separately, evaluates discrimination on validation and test splits, and
saves model artifacts, predictions, metrics, and run metadata.

Pipeline:
1. Load assembled train/val/test feature matrices and targets.
2. Scale clinical features (fit on train only). RNA is pre-scaled in preprocessing.
3. Fit clinical-only logistic regression with C selected via 5-fold CV on train.
4. Fit RNA-only logistic regression with C selected via 5-fold CV on train.
5. Evaluate both models on val and test (ROC-AUC, AP).
6. Save model artifacts, predictions, metrics, and metadata.

Inputs:
- Assembled parquet files from assemble_dataset.py output directory.

Outputs:
- `clinical_model.pkl`
- `rna_model.pkl`
- `clinical_scaler.pkl`
- `predictions_val.parquet`
- `predictions_test.parquet`
- `metrics.json`
- `train_baselines_metadata.json`
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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def load_assembled_datasets(
    assembled_dir: Path,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """Load assembled train/val/test feature matrices and targets from parquet files.

    Parameters:
    - assembled_dir: path to the assembled dataset directory

    Returns:
    - X_clin_train_df, X_clin_val_df, X_clin_test_df: clinical feature matrices
    - X_rna_train_df, X_rna_val_df, X_rna_test_df: RNA feature matrices
    - y_train, y_val, y_test: binary outcome vectors
    """
    assembled_dir = Path(assembled_dir)

    X_clin_train_df = pd.read_parquet(assembled_dir / "train/X_clinical.parquet")
    X_clin_val_df   = pd.read_parquet(assembled_dir / "val/X_clinical.parquet")
    X_clin_test_df  = pd.read_parquet(assembled_dir / "test/X_clinical.parquet")

    X_rna_train_df  = pd.read_parquet(assembled_dir / "train/X_rna.parquet")
    X_rna_val_df    = pd.read_parquet(assembled_dir / "val/X_rna.parquet")
    X_rna_test_df   = pd.read_parquet(assembled_dir / "test/X_rna.parquet")

    y_train = pd.read_parquet(assembled_dir / "train/y.parquet")["y"]
    y_val   = pd.read_parquet(assembled_dir / "val/y.parquet")["y"]
    y_test  = pd.read_parquet(assembled_dir / "test/y.parquet")["y"]

    for split, X_clin, X_rna, y in [
        ("train", X_clin_train_df, X_rna_train_df, y_train),
        ("val",   X_clin_val_df,   X_rna_val_df,   y_val),
        ("test",  X_clin_test_df,  X_rna_test_df,  y_test),
    ]:
        assert X_clin.index.equals(X_rna.index), f"{split}: clinical/RNA index mismatch"
        assert X_clin.index.equals(y.index),     f"{split}: clinical/y index mismatch"
        print(f"{split}: n={len(y)}, n_events={y.sum()}, clin={X_clin.shape[1]} features, rna={X_rna.shape[1]} features")

    return (
        X_clin_train_df, X_clin_val_df, X_clin_test_df,
        X_rna_train_df, X_rna_val_df, X_rna_test_df,
        y_train, y_val, y_test,
    )


def fit_and_apply_scaler(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a StandardScaler on training data and apply to all splits.

    Parameters:
    - X_train_df, X_val_df, X_test_df: feature matrices for each split

    Returns:
    - scaler: fitted StandardScaler
    - X_train_scaled, X_val_scaled, X_test_scaled: scaled arrays
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def fit_logistic_regression(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[LogisticRegressionCV, float, float]:
    """Fit logistic regression with L2 regularization, selecting C via stratified 5-fold CV.

    Parameters:
    - X_train_scaled: scaled training feature matrix
    - y_train: binary outcome vector
    - random_state: random seed for reproducibility

    Returns:
    - model: fitted LogisticRegressionCV
    - best_c: selected regularization strength
    - best_cv_auc: mean ROC-AUC at best C across CV folds
    """
    model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l2",
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    best_c_idx = model.Cs_.tolist().index(model.C_[0])
    best_cv_auc = model.scores_[1].mean(axis=0)[best_c_idx]

    print(f"Best C: {model.C_[0]:.4f}")
    print(f"Train AUC (CV mean at best C): {best_cv_auc:.3f}")

    return model, model.C_[0], best_cv_auc


def evaluate_predictions(
    lr_clin: LogisticRegressionCV,
    lr_rna: LogisticRegressionCV,
    X_clin_scaled: np.ndarray,
    X_rna_scaled: np.ndarray,
    y_true: pd.Series,
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate predictions and compute discrimination metrics for both models.

    Parameters:
    - lr_clin, lr_rna: fitted clinical and RNA logistic regression models
    - X_clin_scaled, X_rna_scaled: scaled feature matrices
    - y_true: binary outcome vector
    - split: split name used for column prefixes (e.g. "val", "test")

    Returns:
    - y_pred_clin, y_pred_rna: predicted probabilities
    - metrics_df: ROC-AUC and AP for each model
    """
    y_pred_clin = lr_clin.predict_proba(X_clin_scaled)[:, 1]
    y_pred_rna = lr_rna.predict_proba(X_rna_scaled)[:, 1]

    metrics_df = pd.DataFrame({
        "model": ["clinical", "rna"],
        f"{split}_roc_auc": [
            roc_auc_score(y_true, y_pred_clin),
            roc_auc_score(y_true, y_pred_rna),
        ],
        f"{split}_ap": [
            average_precision_score(y_true, y_pred_clin),
            average_precision_score(y_true, y_pred_rna),
        ],
    })

    print(metrics_df.to_string(index=False))
    return y_pred_clin, y_pred_rna, metrics_df


def risk_tier_summary(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Bin patients into risk tiers by predicted probability: top 20%, middle 60%, bottom 20%.

    Parameters:
    - y_true: binary outcome vector
    - y_pred: predicted probabilities
    - model_name: label for the model column

    Returns:
    - summary: dataframe with event counts and rates per tier
    """
    thresholds = np.percentile(y_pred, [80, 20])
    tiers = np.where(y_pred >= thresholds[0], "high",
             np.where(y_pred <= thresholds[1], "low", "mid"))

    df = pd.DataFrame({"y": y_true.values, "tier": tiers})
    summary = (
        df.groupby("tier")["y"]
        .agg(n="count", events="sum", event_rate="mean")
        .reindex(["high", "mid", "low"])
        .assign(model=model_name)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline logistic regression models.")
    parser.add_argument("--assembled-dir", required=True, help="Path to assembled dataset directory.")
    parser.add_argument("--outdir", required=True, help="Output directory for artifacts.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    assembled_dir = Path(args.assembled_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    (
        X_clin_train_df, X_clin_val_df, X_clin_test_df,
        X_rna_train_df, X_rna_val_df, X_rna_test_df,
        y_train, y_val, y_test,
    ) = load_assembled_datasets(assembled_dir)

    # Scale clinical features (RNA is pre-scaled in preprocessing)
    scaler_clin, X_clin_train_scaled, X_clin_val_scaled, X_clin_test_scaled = fit_and_apply_scaler(
        X_clin_train_df, X_clin_val_df, X_clin_test_df
    )
    X_rna_train_scaled = X_rna_train_df.values
    X_rna_val_scaled   = X_rna_val_df.values
    X_rna_test_scaled  = X_rna_test_df.values

    # Fit models
    print("\n--- Clinical model ---")
    lr_clin, best_c_clin, best_cv_auc_clin = fit_logistic_regression(
        X_clin_train_scaled, y_train, random_state=args.random_state
    )
    print("\n--- RNA model ---")
    lr_rna, best_c_rna, best_cv_auc_rna = fit_logistic_regression(
        X_rna_train_scaled, y_train, random_state=args.random_state
    )

    # Evaluate on val and test
    print("\n--- Validation ---")
    y_val_pred_clin, y_val_pred_rna, val_metrics_df = evaluate_predictions(
        lr_clin, lr_rna, X_clin_val_scaled, X_rna_val_scaled, y_val, split="val"
    )
    print("\n--- Test ---")
    y_test_pred_clin, y_test_pred_rna, test_metrics_df = evaluate_predictions(
        lr_clin, lr_rna, X_clin_test_scaled, X_rna_test_scaled, y_test, split="test"
    )

    # Majority class baseline: predicts training positive rate for all patients
    majority_prob = float(y_train.mean())
    y_val_pred_majority  = np.full(len(y_val),  majority_prob)
    y_test_pred_majority = np.full(len(y_test), majority_prob)

    # Save model artifacts
    with open(outdir / "clinical_model.pkl", "wb") as f:
        pickle.dump(lr_clin, f)
    with open(outdir / "rna_model.pkl", "wb") as f:
        pickle.dump(lr_rna, f)
    with open(outdir / "clinical_scaler.pkl", "wb") as f:
        pickle.dump(scaler_clin, f)

    # Save predictions
    pd.DataFrame({
        "sample": X_clin_val_df.index,
        "y_true": y_val.values,
        "y_pred_clin": y_val_pred_clin,
        "y_pred_rna": y_val_pred_rna,
        "y_pred_majority": y_val_pred_majority,
    }).to_parquet(outdir / "predictions_val.parquet", index=False)

    pd.DataFrame({
        "sample": X_clin_test_df.index,
        "y_true": y_test.values,
        "y_pred_clin": y_test_pred_clin,
        "y_pred_rna": y_test_pred_rna,
        "y_pred_majority": y_test_pred_majority,
    }).to_parquet(outdir / "predictions_test.parquet", index=False)

    # Save metrics
    metrics = {
        "val": val_metrics_df.set_index("model").to_dict(),
        "test": test_metrics_df.set_index("model").to_dict(),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save metadata
    metadata = {
        "script_name": "train_baselines.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_command": shlex.join(sys.argv),
        "input_files": {
            "assembled_dir": str(assembled_dir.resolve()),
        },
        "key_parameters_used": {
            "random_state": args.random_state,
            "model_type": "LogisticRegressionCV",
            "penalty": "l2",
            "cv_folds": 5,
            "Cs": 10,
            "solver": "lbfgs",
            "clinical_best_c": best_c_clin,
            "rna_best_c": best_c_rna,
        },
        "dataset_statistics": {
            "n_samples_train": int(len(y_train)),
            "n_samples_val": int(len(y_val)),
            "n_samples_test": int(len(y_test)),
            "n_events_train": int(y_train.sum()),
            "n_events_val": int(y_val.sum()),
            "n_events_test": int(y_test.sum()),
            "n_clin_features": int(X_clin_train_df.shape[1]),
            "n_rna_features": int(X_rna_train_df.shape[1]),
            "clinical_cv_auc": round(float(best_cv_auc_clin), 4),
            "rna_cv_auc": round(float(best_cv_auc_rna), 4),
        },
        "output_files": {
            "clinical_model": str(outdir / "clinical_model.pkl"),
            "rna_model": str(outdir / "rna_model.pkl"),
            "clinical_scaler": str(outdir / "clinical_scaler.pkl"),
            "predictions_val": str(outdir / "predictions_val.parquet"),
            "predictions_test": str(outdir / "predictions_test.parquet"),
            "metrics": str(outdir / "metrics.json"),
        },
    }
    (outdir / "train_baselines_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nArtifacts saved to {outdir}")


if __name__ == "__main__":
    main()
