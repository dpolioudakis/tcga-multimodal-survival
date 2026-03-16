"""Preprocess cohort-level clinical features for downstream TCGA modeling.

This module loads the raw clinical table, restricts it to the cohort sample IDs
and feature set, aligns it to the saved train/validation/test split IDs,
applies train-fit preprocessing rules, and writes split-specific parquet files
plus run metadata. The preprocessing is designed to preserve split integrity:
parameters such as numeric medians are fit on the training partition only and
then applied unchanged to validation and test data.

Pipeline:
1. Load cohort sample IDs.
2. Load cohort clinical feature names.
3. Subset the raw clinical table to those samples and features.
4. Apply configured preprocessing rules for missing-value standardization.
5. Align the cohort table to the saved train/validation/test splits.
6. Fit train-only preprocessing artifacts.
7. Transform train, validation, and test splits.
8. Validate outputs and write processed artifacts plus metadata.

Inputs:
- Raw clinical table in TSV or TSV.GZ format.
- Cohort sample ID manifest CSV.
- Cohort clinical feature manifest CSV.
- Clinical preprocessing parameter JSON.
- Saved split ID CSV files for train, validation, and test.

Outputs:
- `train/X_clinical.parquet`
- `val/X_clinical.parquet`
- `test/X_clinical.parquet`
- `clinical_preprocess_metadata.json`
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_clinical_cohort(
    clinical_path: str | Path,
    feature_path: str | Path,
    sample_ids_path: str | Path,
) -> pd.DataFrame:
    """Load the raw clinical table and subset it to the cohort samples and features."""
    sample_ids = pd.read_csv(sample_ids_path)["sample"].astype(str).tolist()
    clinical_features = pd.read_csv(feature_path)["feature"].astype(str).tolist()

    clin_df = pd.read_csv(clinical_path, sep="\t").assign(sample=lambda df: df["sample"].astype(str)).set_index(
        "sample"
    )

    missing_sample_ids = sorted(set(sample_ids) - set(clin_df.index))
    if missing_sample_ids:
        raise ValueError(f"Cohort sample IDs missing from clinical table: {missing_sample_ids[:5]}")

    missing_features = sorted(set(clinical_features) - set(clin_df.columns))
    if missing_features:
        raise ValueError(f"Cohort features missing from clinical table: {missing_features[:5]}")

    clin_df = clin_df.reindex(index=sample_ids, columns=clinical_features).copy()

    return clin_df


def load_split_ids_and_partition_clin_df(
    clin_df: pd.DataFrame,
    sample_ids_path: str | Path,
    split_dir: str | Path,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align the cohort dataframe to saved split IDs and return split-specific views.

    This function verifies that the saved split files cover the cohort exactly,
    checks that all cohort samples are present in `clin_df`, drops the requested
    columns, and materializes train, validation, and test dataframes in saved
    split order.
    """
    sample_ids_df = pd.read_csv(sample_ids_path)
    train_ids_df = pd.read_csv(Path(split_dir) / "train_ids.csv")
    val_ids_df = pd.read_csv(Path(split_dir) / "val_ids.csv")
    test_ids_df = pd.read_csv(Path(split_dir) / "test_ids.csv")

    sample_ids = sample_ids_df["sample"].astype(str)
    train_ids = train_ids_df["sample"].astype(str)
    val_ids = val_ids_df["sample"].astype(str)
    test_ids = test_ids_df["sample"].astype(str)

    if set(sample_ids) != set(train_ids) | set(val_ids) | set(test_ids):
        raise ValueError("Saved splits do not match the cohort sample ID manifest")

    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in clin_df.index]
    if missing_ids:
        raise ValueError(f"Some filtered samples are missing from clin_df: {missing_ids[:5]}")

    clin_df = clin_df.loc[sample_ids].copy()

    missing_drop_cols = sorted(set(drop_cols) - set(clin_df.columns))
    assert not missing_drop_cols, f"Missing drop columns in clin_df: {missing_drop_cols}"

    clin_df = clin_df.drop(columns=drop_cols)

    X_train_df = clin_df.loc[train_ids].copy()
    X_val_df = clin_df.loc[val_ids].copy()
    X_test_df = clin_df.loc[test_ids].copy()

    return clin_df, train_ids, val_ids, test_ids, X_train_df, X_val_df, X_test_df


def fit_numeric_median_imputation_parameters(
    X_train_df: pd.DataFrame,
    features_numeric_median_impute: list[str],
) -> pd.DataFrame:
    """Compute training-set medians for the configured numeric features.

    Returns a two-column dataframe with one row per feature so the fitted
    values can be serialized and reused across splits.
    """
    missing_numeric_impute_cols = sorted(set(features_numeric_median_impute) - set(X_train_df.columns))
    assert not missing_numeric_impute_cols, (
        f"FEATURES_NUMERIC_MEDIAN_IMPUTE columns missing from X_train_df: {missing_numeric_impute_cols}"
    )

    features_numeric_medians = (
        X_train_df[features_numeric_median_impute]
        .median()
        .rename("median")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    return features_numeric_medians


def apply_numeric_median_imputations(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    features_numeric_median_impute: list[str],
    features_numeric_medians: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Impute configured numeric features in all splits using training-set medians."""
    numeric_median_map = features_numeric_medians.set_index("feature")["median"]

    X_train_df = X_train_df.copy()
    X_val_df = X_val_df.copy()
    X_test_df = X_test_df.copy()

    X_train_df.loc[:, features_numeric_median_impute] = X_train_df[features_numeric_median_impute].fillna(
        numeric_median_map
    )
    X_val_df.loc[:, features_numeric_median_impute] = X_val_df[features_numeric_median_impute].fillna(
        numeric_median_map
    )
    X_test_df.loc[:, features_numeric_median_impute] = X_test_df[features_numeric_median_impute].fillna(
        numeric_median_map
    )

    return X_train_df, X_val_df, X_test_df


def fill_unknown_for_selected_categorical_features(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    features_categorical_fill_unknown: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace missing values with `"unknown"` for selected categorical features."""
    missing_categorical_fill_cols = sorted(set(features_categorical_fill_unknown) - set(X_train_df.columns))
    assert not missing_categorical_fill_cols, (
        f"FEATURES_CATEGORICAL_FILL_UNKNOWN columns missing from X_train_df: {missing_categorical_fill_cols}"
    )

    X_train_df = X_train_df.copy()
    X_val_df = X_val_df.copy()
    X_test_df = X_test_df.copy()

    X_train_df.loc[:, features_categorical_fill_unknown] = (
        X_train_df[features_categorical_fill_unknown].fillna("unknown")
    )
    X_val_df.loc[:, features_categorical_fill_unknown] = (
        X_val_df[features_categorical_fill_unknown].fillna("unknown")
    )
    X_test_df.loc[:, features_categorical_fill_unknown] = (
        X_test_df[features_categorical_fill_unknown].fillna("unknown")
    )

    return X_train_df, X_val_df, X_test_df


def validate_imputed_outputs(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    train_ids: pd.Series,
    val_ids: pd.Series,
    test_ids: pd.Series,
) -> None:
    """Validate processed split outputs for completeness, ordering, and leakage."""
    for split_name, split_df, split_ids in [
        ("train", X_train_df, train_ids),
        ("val", X_val_df, val_ids),
        ("test", X_test_df, test_ids),
    ]:
        missing_total = int(split_df.isna().sum().sum())
        if missing_total != 0:
            raise ValueError(f"{split_name} split still has {missing_total} missing values")

        if len(split_df) != len(split_ids):
            raise ValueError(
                f"{split_name} row count mismatch: expected {len(split_ids)}, got {len(split_df)}"
            )

        if split_df.index.duplicated().any():
            raise ValueError(f"{split_name} split contains duplicate sample IDs")

        if list(split_df.index.astype(str)) != list(split_ids.astype(str)):
            raise ValueError(f"{split_name} sample IDs do not match the saved split order")

    train_id_set = set(X_train_df.index.astype(str))
    val_id_set = set(X_val_df.index.astype(str))
    test_id_set = set(X_test_df.index.astype(str))

    if not train_id_set.isdisjoint(val_id_set):
        raise ValueError("Leakage detected between train and val")
    if not train_id_set.isdisjoint(test_id_set):
        raise ValueError("Leakage detected between train and test")
    if not val_id_set.isdisjoint(test_id_set):
        raise ValueError("Leakage detected between val and test")


def main() -> None:
    """Run the clinical preprocessing pipeline from CLI inputs and write outputs."""
    parser = argparse.ArgumentParser(description="Preprocess clinical data using train-fit imputations.")
    parser.add_argument(
        "--clinical-path",
        type=Path,
        default=Path("data/raw/TCGA-BRCA.clinical.tsv.gz"),
        help="Path to the raw clinical TSV/TSV.GZ file.",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path("data/interim/clinical_features_cohort.csv"),
        help="Path to the cohort clinical feature manifest CSV.",
    )
    parser.add_argument(
        "--sample-ids-path",
        type=Path,
        default=Path("data/interim/sample_ids_cohort.csv"),
        help="Path to the cohort sample IDs CSV.",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=Path("data/interim/clinical_preprocess_parameters.json"),
        help="Path to the clinical preprocessing rules JSON.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/splits/os_seed42_v15_t15"),
        help="Directory containing train_ids.csv, val_ids.csv, and test_ids.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/processed/clinical"),
        help="Base output directory for processed clinical split parquet files.",
    )
    args = parser.parse_args()

    clinical_preprocess_parameters = json.loads(args.params_path.read_text(encoding="utf-8"))
    params = clinical_preprocess_parameters["parameters"]

    clin_df = load_clinical_cohort(
        clinical_path=args.clinical_path,
        feature_path=args.feature_path,
        sample_ids_path=args.sample_ids_path,
    )
    n_features_before_drop = int(clin_df.shape[1])

    object_cols = clin_df.select_dtypes(include=["object", "string"]).columns
    # Standardize string-coded missing values before split-specific processing.
    clin_df.loc[:, object_cols] = clin_df[object_cols].apply(
        lambda col: col.str.strip().mask(
            col.str.strip().str.lower().isin(params["MISSING_STRINGS_TO_STANDARDIZE"]),
            pd.NA,
        )
    )

    clin_df, train_ids, val_ids, test_ids, X_train_df, X_val_df, X_test_df = load_split_ids_and_partition_clin_df(
        clin_df=clin_df,
        sample_ids_path=args.sample_ids_path,
        split_dir=args.split_dir,
        drop_cols=params["DROP_COLS"],
    )

    features_numeric_medians = fit_numeric_median_imputation_parameters(
        X_train_df=X_train_df,
        features_numeric_median_impute=params["FEATURES_NUMERIC_MEDIAN_IMPUTE"],
    )

    X_train_df, X_val_df, X_test_df = apply_numeric_median_imputations(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        features_numeric_median_impute=params["FEATURES_NUMERIC_MEDIAN_IMPUTE"],
        features_numeric_medians=features_numeric_medians,
    )

    X_train_df, X_val_df, X_test_df = fill_unknown_for_selected_categorical_features(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        features_categorical_fill_unknown=params["FEATURES_CATEGORICAL_FILL_UNKNOWN"],
    )

    validate_imputed_outputs(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    train_outdir = args.outdir / "train"
    val_outdir = args.outdir / "val"
    test_outdir = args.outdir / "test"
    train_outdir.mkdir(parents=True, exist_ok=True)
    val_outdir.mkdir(parents=True, exist_ok=True)
    test_outdir.mkdir(parents=True, exist_ok=True)

    train_path = train_outdir / "X_clinical.parquet"
    val_path = val_outdir / "X_clinical.parquet"
    test_path = test_outdir / "X_clinical.parquet"

    X_train_df.to_parquet(train_path)
    X_val_df.to_parquet(val_path)
    X_test_df.to_parquet(test_path)

    metadata = {
        "script_name": "preprocess_clinical.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "input_files": {
            "clinical_path": str(args.clinical_path.resolve()),
            "feature_path": str(args.feature_path.resolve()),
            "sample_ids_path": str(args.sample_ids_path.resolve()),
            "split_dir": str(args.split_dir.resolve()),
            "params_path": str(args.params_path.resolve()),
        },
        "output_files": {
            "train_path": str(train_path.resolve()),
            "val_path": str(val_path.resolve()),
            "test_path": str(test_path.resolve()),
        },
        "config_path": str(args.params_path.resolve()),
        "random_seed": None,
        "key_parameters_used": params,
        "dataset_statistics": {
            "n_samples_total": int(clin_df.shape[0]),
            "n_samples_train": int(X_train_df.shape[0]),
            "n_samples_val": int(X_val_df.shape[0]),
            "n_samples_test": int(X_test_df.shape[0]),
            "n_features_before_drop": n_features_before_drop,
            "n_features_after_drop": int(X_train_df.shape[1]),
            "n_drop_cols": int(len(params["DROP_COLS"])),
            "n_numeric_median_impute": int(len(params["FEATURES_NUMERIC_MEDIAN_IMPUTE"])),
            "n_categorical_fill_unknown": int(len(params["FEATURES_CATEGORICAL_FILL_UNKNOWN"])),
        },
    }

    metadata_path = args.outdir / "clinical_preprocess_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
