"""Apply train-fit RNA preprocessing to the TCGA modeling splits.

This module is the pipeline implementation of the RNA preprocessing decisions
developed in the notebook workflow. It assumes the cohort, sample manifest,
split IDs, and RNA preprocessing rules JSON already exist. The script derives
the final retained gene list from the training split only, fits a
``StandardScaler`` on those training genes, applies the unchanged feature set
and scaler to validation and test, validates split integrity, and saves both
processed matrices and run metadata for downstream modeling.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_rna_preprocessing_parameters(
    X_train_df: pd.DataFrame,
    params_path: str | Path,
) -> list[str]:
    """Derive the final retained RNA gene list from the training split.

    Parameters:
    - X_train_df: training RNA expression dataframe with samples as rows and genes as columns.
    - params_path: path to the saved RNA preprocessing parameter JSON.

    Returns:
    - RNA_FINAL_GENE_LIST: gene names retained after applying the saved
      prevalence and variance thresholds to the training split.
    """
    if X_train_df.empty:
        raise ValueError("X_train_df is empty")

    params_path = Path(params_path)
    rna_preprocess_parameters = json.loads(params_path.read_text(encoding="utf-8"))

    RNA_EXPRESSION_THRESHOLD = rna_preprocess_parameters["parameters"]["RNA_EXPRESSION_THRESHOLD"]
    RNA_PREVALENCE_THRESHOLD = rna_preprocess_parameters["parameters"]["RNA_PREVALENCE_THRESHOLD"]
    RNA_VARIANCE_THRESHOLD = rna_preprocess_parameters["parameters"]["RNA_VARIANCE_THRESHOLD"]

    gene_detection_rate_df = pd.DataFrame({
        "detection_rate": (X_train_df > RNA_EXPRESSION_THRESHOLD).sum(axis=0) / X_train_df.shape[0]
    })

    genes_after_prevalence_filter = gene_detection_rate_df.index[
        gene_detection_rate_df["detection_rate"] >= RNA_PREVALENCE_THRESHOLD
    ].tolist()

    X_train_prevalence_filtered_df = X_train_df.loc[:, genes_after_prevalence_filter].copy()

    gene_variance_df = pd.DataFrame({
        "variance": X_train_prevalence_filtered_df.var(axis=0)
    })

    RNA_FINAL_GENE_LIST = gene_variance_df.index[
        gene_variance_df["variance"] > RNA_VARIANCE_THRESHOLD
    ].tolist()

    return RNA_FINAL_GENE_LIST


def apply_rna_preprocessing_to_splits(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    RNA_FINAL_GENE_LIST: list[str],
    scaler: StandardScaler,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter and scale all RNA splits using training-derived artifacts.

    Parameters:
    - X_train_df: training RNA dataframe with samples as rows and genes as columns.
    - X_val_df: validation RNA dataframe with samples as rows and genes as columns.
    - X_test_df: test RNA dataframe with samples as rows and genes as columns.
    - RNA_FINAL_GENE_LIST: training-derived final gene list to retain in all splits.
    - scaler: ``StandardScaler`` fit on the training split only.

    Returns:
    - X_train_scaled_df: filtered and scaled training dataframe.
    - X_val_scaled_df: filtered and scaled validation dataframe.
    - X_test_scaled_df: filtered and scaled test dataframe.
    """
    missing_genes_df = pd.DataFrame({
        "split": ["train", "val", "test"],
        "n_missing_genes": [
            len(set(RNA_FINAL_GENE_LIST) - set(X_train_df.columns)),
            len(set(RNA_FINAL_GENE_LIST) - set(X_val_df.columns)),
            len(set(RNA_FINAL_GENE_LIST) - set(X_test_df.columns)),
        ],
    })

    if (missing_genes_df["n_missing_genes"] > 0).any():
        raise ValueError("Some training-selected genes are missing from one or more splits")

    X_train_filtered_df = X_train_df.loc[:, RNA_FINAL_GENE_LIST].copy()
    X_val_filtered_df = X_val_df.loc[:, RNA_FINAL_GENE_LIST].copy()
    X_test_filtered_df = X_test_df.loc[:, RNA_FINAL_GENE_LIST].copy()

    X_train_scaled_df = pd.DataFrame(
        scaler.transform(X_train_filtered_df),
        index=X_train_filtered_df.index,
        columns=RNA_FINAL_GENE_LIST,
    )
    X_val_scaled_df = pd.DataFrame(
        scaler.transform(X_val_filtered_df),
        index=X_val_filtered_df.index,
        columns=RNA_FINAL_GENE_LIST,
    )
    X_test_scaled_df = pd.DataFrame(
        scaler.transform(X_test_filtered_df),
        index=X_test_filtered_df.index,
        columns=RNA_FINAL_GENE_LIST,
    )

    return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df


def validate_rna_preprocessing_outputs(
    X_train_scaled_df: pd.DataFrame,
    X_val_scaled_df: pd.DataFrame,
    X_test_scaled_df: pd.DataFrame,
    train_ids: pd.Series,
    val_ids: pd.Series,
    test_ids: pd.Series,
    RNA_FINAL_GENE_LIST: list[str],
) -> pd.DataFrame:
    """Validate processed RNA outputs against the saved split definitions.

    Parameters:
    - X_train_scaled_df: processed training RNA dataframe.
    - X_val_scaled_df: processed validation RNA dataframe.
    - X_test_scaled_df: processed test RNA dataframe.
    - train_ids: saved training sample IDs.
    - val_ids: saved validation sample IDs.
    - test_ids: saved test sample IDs.
    - RNA_FINAL_GENE_LIST: final training-derived gene list.

    Returns:
    - validation_summary_df: per-split summary of observed and expected row and
      feature counts.
    """
    expected_feature_count = len(RNA_FINAL_GENE_LIST)

    validation_summary_df = pd.DataFrame({
        "split": ["train", "val", "test"],
        "n_rows": [X_train_scaled_df.shape[0], X_val_scaled_df.shape[0], X_test_scaled_df.shape[0]],
        "expected_rows": [len(train_ids), len(val_ids), len(test_ids)],
        "n_features": [X_train_scaled_df.shape[1], X_val_scaled_df.shape[1], X_test_scaled_df.shape[1]],
        "expected_features": [expected_feature_count, expected_feature_count, expected_feature_count],
    })

    if not (validation_summary_df["n_rows"] == validation_summary_df["expected_rows"]).all():
        raise ValueError("Row counts do not match saved split sizes")

    if not (validation_summary_df["n_features"] == validation_summary_df["expected_features"]).all():
        raise ValueError("Feature counts do not match the final training-derived gene list")

    if list(X_train_scaled_df.index.astype(str)) != list(train_ids.astype(str)):
        raise ValueError("Train sample IDs do not match saved split order")
    if list(X_val_scaled_df.index.astype(str)) != list(val_ids.astype(str)):
        raise ValueError("Validation sample IDs do not match saved split order")
    if list(X_test_scaled_df.index.astype(str)) != list(test_ids.astype(str)):
        raise ValueError("Test sample IDs do not match saved split order")

    train_id_set = set(X_train_scaled_df.index.astype(str))
    val_id_set = set(X_val_scaled_df.index.astype(str))
    test_id_set = set(X_test_scaled_df.index.astype(str))

    if not train_id_set.isdisjoint(val_id_set):
        raise ValueError("Leakage detected between train and val")
    if not train_id_set.isdisjoint(test_id_set):
        raise ValueError("Leakage detected between train and test")
    if not val_id_set.isdisjoint(test_id_set):
        raise ValueError("Leakage detected between val and test")

    return validation_summary_df


def main() -> None:
    """Run the RNA preprocessing pipeline from the command line.

    The CLI loads the raw RNA matrix, aligns it to the saved cohort and split
    IDs, derives train-fit preprocessing artifacts, applies them to all splits,
    writes processed parquet files, and records run metadata alongside the
    outputs.
    """
    parser = argparse.ArgumentParser(description="Preprocess RNA-seq data using train-fit filtering and scaling.")
    parser.add_argument(
        "--rna-path",
        type=Path,
        default=Path("data/raw/TCGA-BRCA.star_tpm.tsv.gz"),
        help="Path to the raw RNA expression TSV/TSV.GZ file.",
    )
    parser.add_argument(
        "--sample-ids-path",
        type=Path,
        default=Path("data/interim/sample_ids_filtered.csv"),
        help="Path to the filtered cohort sample IDs CSV.",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=Path("data/interim/rna_preprocess_parameters.json"),
        help="Path to the RNA preprocessing rules JSON.",
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
        default=Path("data/processed/rna"),
        help="Base output directory for processed RNA split parquet files.",
    )
    args = parser.parse_args()

    # Load saved cohort and split manifests.
    sample_ids_df = pd.read_csv(args.sample_ids_path)
    train_ids_df = pd.read_csv(args.split_dir / "train_ids.csv")
    val_ids_df = pd.read_csv(args.split_dir / "val_ids.csv")
    test_ids_df = pd.read_csv(args.split_dir / "test_ids.csv")

    sample_ids = sample_ids_df["sample"].astype(str)
    train_ids = train_ids_df["sample"].astype(str)
    val_ids = val_ids_df["sample"].astype(str)
    test_ids = test_ids_df["sample"].astype(str)

    # Load the raw RNA matrix and align it to the filtered cohort.
    rna_df = pd.read_csv(args.rna_path, sep="\t")
    if "Ensembl_ID" not in rna_df.columns:
        raise KeyError("Expected 'Ensembl_ID' column in RNA expression table")

    missing_sample_ids = sorted(set(sample_ids) - set(rna_df.columns))
    if missing_sample_ids:
        raise KeyError(f"Filtered sample IDs missing from RNA table: {missing_sample_ids[:5]}")

    rna_df = rna_df.set_index("Ensembl_ID").loc[:, sample_ids].T
    rna_df.columns.name = None
    rna_df.index.name = "sample"

    missing_ids = sorted((set(train_ids) | set(val_ids) | set(test_ids)) - set(rna_df.index))
    if missing_ids:
        raise KeyError(f"Split sample IDs missing from rna_df: {missing_ids[:5]}")

    X_train_df = rna_df.loc[train_ids].copy()
    X_val_df = rna_df.loc[val_ids].copy()
    X_test_df = rna_df.loc[test_ids].copy()

    # Fit train-only artifacts, then apply them unchanged to all splits.
    RNA_FINAL_GENE_LIST = fit_rna_preprocessing_parameters(
        X_train_df=X_train_df,
        params_path=args.params_path,
    )

    X_train_filtered_df = X_train_df.loc[:, RNA_FINAL_GENE_LIST].copy()
    scaler = StandardScaler().fit(X_train_filtered_df)

    X_train_scaled_df, X_val_scaled_df, X_test_scaled_df = apply_rna_preprocessing_to_splits(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        RNA_FINAL_GENE_LIST=RNA_FINAL_GENE_LIST,
        scaler=scaler,
    )

    validation_summary_df = validate_rna_preprocessing_outputs(
        X_train_scaled_df=X_train_scaled_df,
        X_val_scaled_df=X_val_scaled_df,
        X_test_scaled_df=X_test_scaled_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        RNA_FINAL_GENE_LIST=RNA_FINAL_GENE_LIST,
    )

    train_outdir = args.outdir / "train"
    val_outdir = args.outdir / "val"
    test_outdir = args.outdir / "test"
    train_outdir.mkdir(parents=True, exist_ok=True)
    val_outdir.mkdir(parents=True, exist_ok=True)
    test_outdir.mkdir(parents=True, exist_ok=True)

    train_path = train_outdir / "X_rna.parquet"
    val_path = val_outdir / "X_rna.parquet"
    test_path = test_outdir / "X_rna.parquet"

    X_train_scaled_df.to_parquet(train_path)
    X_val_scaled_df.to_parquet(val_path)
    X_test_scaled_df.to_parquet(test_path)

    # Save a compact metadata record describing inputs, outputs, and fitted dimensions.
    rna_preprocess_parameters = json.loads(args.params_path.read_text(encoding="utf-8"))
    metadata = {
        "script_name": "preprocess_rna.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "rna_path": str(args.rna_path.resolve()),
            "sample_ids_path": str(args.sample_ids_path.resolve()),
            "split_dir": str(args.split_dir.resolve()),
            "params_path": str(args.params_path.resolve()),
        },
        "output_files": {
            "train_path": str(train_path.resolve()),
            "val_path": str(val_path.resolve()),
            "test_path": str(test_path.resolve()),
        },
        "command": shlex.join(sys.argv),
        "config_path": str(args.params_path.resolve()),
        "random_seed": None,
        "key_parameters_used": rna_preprocess_parameters["parameters"],
        "dataset_statistics": {
            "n_samples_total": int(rna_df.shape[0]),
            "n_samples_train": int(X_train_scaled_df.shape[0]),
            "n_samples_val": int(X_val_scaled_df.shape[0]),
            "n_samples_test": int(X_test_scaled_df.shape[0]),
            "n_features_before_filtering": int(X_train_df.shape[1]),
            "n_features_after_filtering": int(len(RNA_FINAL_GENE_LIST)),
        },
        "validation_summary": validation_summary_df.to_dict(orient="records"),
    }

    metadata_path = args.outdir / "rna_preprocess_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
