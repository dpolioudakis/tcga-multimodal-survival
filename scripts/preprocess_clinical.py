"""Preprocess TCGA clinical features into train/val/test splits.

Purpose:
- load the raw clinical table and filtered feature list
- standardize missing-like strings
- partition by saved split IDs
- fit train-only imputation parameters
- apply imputations to all splits
- save processed splits and learned parameters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_preprocess_config(config_path: str | Path) -> dict:
    """Load preprocessing rules from JSON.

    Parameters:
    - config_path: path to the clinical preprocessing config JSON

    Returns:
    - dict with preprocessing rule lists
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required_keys = [
        "MISSING_STRINGS_TO_STANDARDIZE",
        "FEATURES_NUMERIC_MEDIAN_IMPUTE",
        "FEATURES_CATEGORICAL_FILL_UNKNOWN",
        "MANUAL_DROP_COLS",
        "DROP_COLS",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing config keys: {missing_keys}")
    return config


def load_filtered_clinical_df(
    clinical_path: str | Path,
    feature_path: str | Path,
) -> pd.DataFrame:
    """Load raw clinical data and keep only selected feature columns.

    Parameters:
    - clinical_path: path to the raw clinical TSV/TSV.GZ
    - feature_path: path to the filtered feature manifest CSV

    Returns:
    - clinical dataframe indexed by sample
    """
    feature_df = pd.read_csv(feature_path)
    if "feature" not in feature_df.columns:
        raise KeyError("Feature manifest must contain a 'feature' column")

    clinical_features = feature_df["feature"].astype(str).tolist()
    clin_df = pd.read_csv(clinical_path, sep="\t")

    if "sample" not in clin_df.columns:
        raise KeyError("Raw clinical table must contain a 'sample' column")

    clin_df = clin_df.set_index("sample")

    missing_features = [feature for feature in clinical_features if feature not in clin_df.columns]
    if missing_features:
        raise KeyError(f"Filtered features missing from raw clinical data: {missing_features}")

    return clin_df.loc[:, clinical_features].copy()


def standardize_missing_values(
    df: pd.DataFrame,
    missing_strings_to_standardize: list[str],
) -> pd.DataFrame:
    """Convert common missing-like strings to pd.NA.

    Parameters:
    - df: input dataframe
    - missing_strings_to_standardize: lowercased string values to treat as missing

    Returns:
    - dataframe with standardized missing values
    """
    df = df.copy()
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    df.loc[:, object_cols] = df[object_cols].apply(
        lambda col: col.str.strip().mask(
            col.str.strip().str.lower().isin(missing_strings_to_standardize),
            pd.NA,
        )
    )
    return df


def load_split_ids_and_partition_clin_df(
    clin_df: pd.DataFrame,
    sample_ids_path: str | Path,
    split_dir: str | Path,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align the clinical dataframe to the saved cohort and split IDs.

    Parameters:
    - clin_df: full clinical dataframe indexed by sample
    - sample_ids_path: path to filtered cohort sample IDs
    - split_dir: directory containing train_ids.csv, val_ids.csv, test_ids.csv
    - drop_cols: columns to drop before creating split dataframes

    Returns:
    - filtered clinical dataframe plus split IDs and split dataframes
    """
    split_dir = Path(split_dir)
    sample_ids_df = pd.read_csv(sample_ids_path)
    train_ids_df = pd.read_csv(split_dir / "train_ids.csv")
    val_ids_df = pd.read_csv(split_dir / "val_ids.csv")
    test_ids_df = pd.read_csv(split_dir / "test_ids.csv")

    sample_ids = sample_ids_df["sample"].astype(str)
    train_ids = train_ids_df["sample"].astype(str)
    val_ids = val_ids_df["sample"].astype(str)
    test_ids = test_ids_df["sample"].astype(str)

    if set(sample_ids) != set(train_ids) | set(val_ids) | set(test_ids):
        raise ValueError("Saved splits do not match sample_ids_filtered.csv")

    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in clin_df.index]
    if missing_ids:
        raise ValueError(f"Some filtered samples are missing from clin_df: {missing_ids[:5]}")

    clin_df = clin_df.loc[sample_ids].copy()

    missing_drop_cols = sorted(set(drop_cols) - set(clin_df.columns))
    if missing_drop_cols:
        raise KeyError(f"Missing drop columns in clin_df: {missing_drop_cols}")

    clin_df = clin_df.drop(columns=drop_cols)

    X_train_df = clin_df.loc[train_ids].copy()
    X_val_df = clin_df.loc[val_ids].copy()
    X_test_df = clin_df.loc[test_ids].copy()

    return clin_df, train_ids, val_ids, test_ids, X_train_df, X_val_df, X_test_df


def fit_numeric_median_imputation_parameters(
    X_train_df: pd.DataFrame,
    features_numeric_median_impute: list[str],
) -> pd.DataFrame:
    """Fit train-only medians for selected numeric features.

    Parameters:
    - X_train_df: training split dataframe
    - features_numeric_median_impute: numeric columns to impute with the median

    Returns:
    - dataframe with columns feature and median
    """
    missing_cols = sorted(set(features_numeric_median_impute) - set(X_train_df.columns))
    if missing_cols:
        raise KeyError(f"Numeric imputation columns missing from X_train_df: {missing_cols}")

    return (
        X_train_df[features_numeric_median_impute]
        .median()
        .rename("median")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "feature"})
    )


def apply_numeric_median_imputations(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    features_numeric_median_impute: list[str],
    features_numeric_medians: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fill selected numeric columns using train-fit medians.

    Parameters:
    - X_train_df: training split dataframe
    - X_val_df: validation split dataframe
    - X_test_df: test split dataframe
    - features_numeric_median_impute: numeric columns to fill
    - features_numeric_medians: dataframe with feature and median columns

    Returns:
    - transformed train, validation, and test dataframes
    """
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
    """Fill selected categorical columns with 'unknown'.

    Parameters:
    - X_train_df: training split dataframe
    - X_val_df: validation split dataframe
    - X_test_df: test split dataframe
    - features_categorical_fill_unknown: categorical columns to fill with 'unknown'

    Returns:
    - transformed train, validation, and test dataframes
    """
    missing_cols = sorted(set(features_categorical_fill_unknown) - set(X_train_df.columns))
    if missing_cols:
        raise KeyError(f"Categorical fill columns missing from X_train_df: {missing_cols}")

    X_train_df = X_train_df.copy()
    X_val_df = X_val_df.copy()
    X_test_df = X_test_df.copy()

    X_train_df.loc[:, features_categorical_fill_unknown] = X_train_df[features_categorical_fill_unknown].fillna(
        "unknown"
    )
    X_val_df.loc[:, features_categorical_fill_unknown] = X_val_df[features_categorical_fill_unknown].fillna(
        "unknown"
    )
    X_test_df.loc[:, features_categorical_fill_unknown] = X_test_df[features_categorical_fill_unknown].fillna(
        "unknown"
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
    """Validate split shapes, IDs, and missingness after preprocessing.

    Parameters:
    - X_train_df: processed training split
    - X_val_df: processed validation split
    - X_test_df: processed test split
    - train_ids: saved training sample IDs
    - val_ids: saved validation sample IDs
    - test_ids: saved test sample IDs

    Returns:
    - None
    """
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


def save_clinical_preprocess_parameters(
    params_path: str | Path,
    missing_strings_to_standardize: list[str],
    features_numeric_median_impute: list[str],
    features_categorical_fill_unknown: list[str],
    manual_drop_cols: list[str],
    drop_cols: list[str],
    features_numeric_medians: pd.DataFrame,
) -> dict:
    """Save preprocessing rules and learned medians to JSON.

    Parameters:
    - params_path: output JSON path
    - missing_strings_to_standardize: strings standardized to missing
    - features_numeric_median_impute: numeric median-impute feature names
    - features_categorical_fill_unknown: categorical fill-with-unknown feature names
    - manual_drop_cols: manually dropped feature names
    - drop_cols: full dropped feature list
    - features_numeric_medians: dataframe with feature and median columns

    Returns:
    - parameter dictionary that was written to disk
    """
    params_path = Path(params_path)
    params_path.parent.mkdir(parents=True, exist_ok=True)

    clinical_preprocess_parameters = {
        "MISSING_STRINGS_TO_STANDARDIZE": sorted(missing_strings_to_standardize),
        "FEATURES_NUMERIC_MEDIAN_IMPUTE": features_numeric_median_impute,
        "FEATURES_CATEGORICAL_FILL_UNKNOWN": features_categorical_fill_unknown,
        "MANUAL_DROP_COLS": manual_drop_cols,
        "DROP_COLS": drop_cols,
        "FEATURES_NUMERIC_MEDIANS": dict(
            zip(features_numeric_medians["feature"], features_numeric_medians["median"])
        ),
    }

    params_path.write_text(json.dumps(clinical_preprocess_parameters, indent=2), encoding="utf-8")
    return clinical_preprocess_parameters


def save_processed_splits(
    outdir: str | Path,
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
) -> None:
    """Save processed clinical splits to CSV.

    Parameters:
    - outdir: output directory
    - X_train_df: processed training split
    - X_val_df: processed validation split
    - X_test_df: processed test split

    Returns:
    - None
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_train_df.to_csv(outdir / "X_train_clinical.csv", index=True, index_label="sample")
    X_val_df.to_csv(outdir / "X_val_clinical.csv", index=True, index_label="sample")
    X_test_df.to_csv(outdir / "X_test_clinical.csv", index=True, index_label="sample")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Parameters:
    - None

    Returns:
    - configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Preprocess clinical data for train/val/test modeling splits.")
    parser.add_argument(
        "--clinical-path",
        type=Path,
        default=Path("data/raw/TCGA-BRCA.clinical.tsv.gz"),
        help="Path to the raw clinical TSV/TSV.GZ file.",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path("data/interim/clinical_features_filtered.csv"),
        help="Path to the filtered clinical feature manifest CSV.",
    )
    parser.add_argument(
        "--sample-ids-path",
        type=Path,
        default=Path("data/interim/sample_ids_filtered.csv"),
        help="Path to the filtered cohort sample IDs CSV.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("data/interim/clinical_preprocess_parameters.json"),
        help="Path to the clinical preprocessing config JSON.",
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
        help="Directory to save processed clinical split CSVs.",
    )
    parser.add_argument(
        "--params-out",
        type=Path,
        default=Path("data/interim/clinical_preprocess_parameters.json"),
        help="Path to save learned preprocessing parameters JSON.",
    )
    return parser


def main() -> None:
    """Run the clinical preprocessing pipeline from CLI arguments.

    Parameters:
    - None

    Returns:
    - None
    """
    args = build_arg_parser().parse_args()
    config = load_preprocess_config(args.config_path)

    clin_df = load_filtered_clinical_df(
        clinical_path=args.clinical_path,
        feature_path=args.feature_path,
    )
    clin_df = standardize_missing_values(
        clin_df,
        missing_strings_to_standardize=config["MISSING_STRINGS_TO_STANDARDIZE"],
    )

    clin_df, train_ids, val_ids, test_ids, X_train_df, X_val_df, X_test_df = load_split_ids_and_partition_clin_df(
        clin_df=clin_df,
        sample_ids_path=args.sample_ids_path,
        split_dir=args.split_dir,
        drop_cols=config["DROP_COLS"],
    )

    features_numeric_medians = fit_numeric_median_imputation_parameters(
        X_train_df=X_train_df,
        features_numeric_median_impute=config["FEATURES_NUMERIC_MEDIAN_IMPUTE"],
    )

    X_train_df, X_val_df, X_test_df = apply_numeric_median_imputations(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        features_numeric_median_impute=config["FEATURES_NUMERIC_MEDIAN_IMPUTE"],
        features_numeric_medians=features_numeric_medians,
    )

    X_train_df, X_val_df, X_test_df = fill_unknown_for_selected_categorical_features(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        features_categorical_fill_unknown=config["FEATURES_CATEGORICAL_FILL_UNKNOWN"],
    )

    validate_imputed_outputs(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    save_processed_splits(
        outdir=args.outdir,
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
    )

    save_clinical_preprocess_parameters(
        params_path=args.params_out,
        missing_strings_to_standardize=config["MISSING_STRINGS_TO_STANDARDIZE"],
        features_numeric_median_impute=config["FEATURES_NUMERIC_MEDIAN_IMPUTE"],
        features_categorical_fill_unknown=config["FEATURES_CATEGORICAL_FILL_UNKNOWN"],
        manual_drop_cols=config["MANUAL_DROP_COLS"],
        drop_cols=config["DROP_COLS"],
        features_numeric_medians=features_numeric_medians,
    )


if __name__ == "__main__":
    main()
