"""Assemble split-specific multimodal datasets for downstream TCGA modeling.

This module is the final data assembly step after split creation and
modality-specific preprocessing. It loads the saved clinical and RNA feature
matrices for each split, enforces within-split sample alignment across
modalities, constructs split-specific outcome vectors, validates assembled
dataset integrity, and writes the final model-ready parquet files plus run
metadata.

Pipeline:
1. Load split-specific processed clinical and RNA matrices.
2. Align RNA sample ordering to the clinical sample ordering within each split.
3. Build train, validation, and test target vectors from the survival table.
4. Validate counts, ordering, duplicate IDs, missingness, and feature
   dimensionality across assembled outputs.
5. Save modality-specific inputs, targets, and concatenated baseline features.
6. Write a metadata record describing inputs, outputs, and dataset dimensions.

Inputs:
- Processed clinical parquet files for train, validation, and test splits.
- Processed RNA parquet files for train, validation, and test splits.
- Survival TSV or TSV.GZ file containing the requested outcome column.
- CLI arguments describing output location and target column.

Outputs:
- `train/X_clinical.parquet`
- `train/X_rna.parquet`
- `train/X_concat.parquet`
- `train/y.parquet`
- `val/X_clinical.parquet`
- `val/X_rna.parquet`
- `val/X_concat.parquet`
- `val/y.parquet`
- `test/X_clinical.parquet`
- `test/X_rna.parquet`
- `test/X_concat.parquet`
- `test/y.parquet`
- `assemble_dataset_metadata.json`
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_processed_modality_matrices(
    clinical_dir: str | Path,
    rna_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed clinical and RNA feature matrices for each split.

    Parameters
    ----------
    clinical_dir : str | Path
        Base directory containing split-specific clinical parquet files.
    rna_dir : str | Path
        Base directory containing split-specific RNA parquet files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `X_clin_train_df`, `X_clin_val_df`, `X_clin_test_df`,
        `X_rna_train_df`, `X_rna_val_df`, and `X_rna_test_df`.
    """
    clinical_dir = Path(clinical_dir)
    rna_dir = Path(rna_dir)

    # Load the split-specific processed matrices produced by the preprocessing scripts.
    X_clin_train_df = pd.read_parquet(clinical_dir / "train" / "X_clinical.parquet")
    X_clin_val_df = pd.read_parquet(clinical_dir / "val" / "X_clinical.parquet")
    X_clin_test_df = pd.read_parquet(clinical_dir / "test" / "X_clinical.parquet")

    X_rna_train_df = pd.read_parquet(rna_dir / "train" / "X_rna.parquet")
    X_rna_val_df = pd.read_parquet(rna_dir / "val" / "X_rna.parquet")
    X_rna_test_df = pd.read_parquet(rna_dir / "test" / "X_rna.parquet")

    # Basic sanity checks: each split should contain rows and be indexed by sample ID.
    assert X_clin_train_df.shape[0] > 0 and X_rna_train_df.shape[0] > 0
    assert X_clin_train_df.index.name == "sample" and X_rna_train_df.index.name == "sample"

    print(
        f"clinical: train={X_clin_train_df.shape}, val={X_clin_val_df.shape}, test={X_clin_test_df.shape}"
    )
    print(
        f"rna: train={X_rna_train_df.shape}, val={X_rna_val_df.shape}, test={X_rna_test_df.shape}"
    )

    return (
        X_clin_train_df,
        X_clin_val_df,
        X_clin_test_df,
        X_rna_train_df,
        X_rna_val_df,
        X_rna_test_df,
    )


def align_modalities_within_splits(
    X_clin_train_df: pd.DataFrame,
    X_clin_val_df: pd.DataFrame,
    X_clin_test_df: pd.DataFrame,
    X_rna_train_df: pd.DataFrame,
    X_rna_val_df: pd.DataFrame,
    X_rna_test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align RNA sample ordering to clinical sample ordering within each split.

    Parameters
    ----------
    X_clin_train_df, X_clin_val_df, X_clin_test_df : pd.DataFrame
        Clinical feature matrices for the train, validation, and test splits.
    X_rna_train_df, X_rna_val_df, X_rna_test_df : pd.DataFrame
        RNA feature matrices for the train, validation, and test splits.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `X_clin_train_df`, `X_clin_val_df`, `X_clin_test_df`,
        `X_rna_train_df`, `X_rna_val_df`, `X_rna_test_df`, with RNA reindexed
        to the clinical ordering only when needed.
    """
    if not X_clin_train_df.index.equals(X_rna_train_df.index):
        X_rna_train_df = X_rna_train_df.reindex(X_clin_train_df.index)

    if not X_clin_val_df.index.equals(X_rna_val_df.index):
        X_rna_val_df = X_rna_val_df.reindex(X_clin_val_df.index)

    if not X_clin_test_df.index.equals(X_rna_test_df.index):
        X_rna_test_df = X_rna_test_df.reindex(X_clin_test_df.index)

    # Validate exact sample alignment after any needed reindexing.
    assert X_clin_train_df.index.equals(X_rna_train_df.index)
    assert X_clin_val_df.index.equals(X_rna_val_df.index)
    assert X_clin_test_df.index.equals(X_rna_test_df.index)

    return (
        X_clin_train_df,
        X_clin_val_df,
        X_clin_test_df,
        X_rna_train_df,
        X_rna_val_df,
        X_rna_test_df,
    )


def build_target_vectors(
    survival_path: str | Path,
    event_col: str,
    X_clin_train_df: pd.DataFrame,
    X_clin_val_df: pd.DataFrame,
    X_clin_test_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build split-specific target vectors aligned to the clinical sample order.

    Parameters
    ----------
    survival_path : str | Path
        Path to the survival table containing `sample` and the outcome column.
    event_col : str
        Name of the outcome column to extract.
    X_clin_train_df, X_clin_val_df, X_clin_test_df : pd.DataFrame
        Clinical feature matrices whose sample indices define the target ordering.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        `y_train`, `y_val`, and `y_test`, indexed by sample ID in split order.
    """
    survival_path = Path(survival_path)

    # Load outcome labels once and index by sample ID for split-specific target construction.
    surv_df = pd.read_csv(survival_path, sep="\t")[["sample", event_col]].astype({"sample": str}).set_index("sample")

    y_train = surv_df.loc[X_clin_train_df.index, event_col].astype(int).copy()
    y_val = surv_df.loc[X_clin_val_df.index, event_col].astype(int).copy()
    y_test = surv_df.loc[X_clin_test_df.index, event_col].astype(int).copy()

    # Validate that target ordering matches the aligned feature matrices exactly.
    assert y_train.index.equals(X_clin_train_df.index)
    assert y_val.index.equals(X_clin_val_df.index)
    assert y_test.index.equals(X_clin_test_df.index)

    return y_train, y_val, y_test


def validate_dataset_invariants(
    X_clin_train_df: pd.DataFrame,
    X_clin_val_df: pd.DataFrame,
    X_clin_test_df: pd.DataFrame,
    X_rna_train_df: pd.DataFrame,
    X_rna_val_df: pd.DataFrame,
    X_rna_test_df: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Validate split-level consistency across modalities and target vectors.

    Parameters
    ----------
    X_clin_train_df, X_clin_val_df, X_clin_test_df : pd.DataFrame
        Clinical feature matrices for the train, validation, and test splits.
    X_rna_train_df, X_rna_val_df, X_rna_test_df : pd.DataFrame
        RNA feature matrices for the train, validation, and test splits.
    y_train, y_val, y_test : pd.Series
        Target vectors aligned to the split-specific feature matrices.

    Returns
    -------
    pd.DataFrame
        Split-level summary of sample counts and per-modality feature counts.
    """
    # Check split-by-split consistency across modalities and targets.
    for split_name, X_clin_df, X_rna_df, y_split in [
        ("train", X_clin_train_df, X_rna_train_df, y_train),
        ("val", X_clin_val_df, X_rna_val_df, y_val),
        ("test", X_clin_test_df, X_rna_test_df, y_test),
    ]:
        assert X_clin_df.shape[0] == X_rna_df.shape[0] == y_split.shape[0]
        assert X_clin_df.index.equals(X_rna_df.index)
        assert X_clin_df.index.equals(y_split.index)
        assert not X_clin_df.index.duplicated().any()
        assert not X_rna_df.index.duplicated().any()
        assert not y_split.index.duplicated().any()
        assert X_clin_df.isna().sum().sum() == 0
        assert X_rna_df.isna().sum().sum() == 0
        assert y_split.isna().sum() == 0

    # Check that feature dimensions are stable across splits within each modality.
    assert X_clin_train_df.shape[1] == X_clin_val_df.shape[1] == X_clin_test_df.shape[1]
    assert X_rna_train_df.shape[1] == X_rna_val_df.shape[1] == X_rna_test_df.shape[1]

    validation_summary_df = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "n_samples": [len(y_train), len(y_val), len(y_test)],
            "n_clin_features": [X_clin_train_df.shape[1], X_clin_val_df.shape[1], X_clin_test_df.shape[1]],
            "n_rna_features": [X_rna_train_df.shape[1], X_rna_val_df.shape[1], X_rna_test_df.shape[1]],
        }
    )

    print("All dataset invariant checks passed.")

    return validation_summary_df


def save_final_model_inputs(
    outdir: str | Path,
    X_clin_train_df: pd.DataFrame,
    X_clin_val_df: pd.DataFrame,
    X_clin_test_df: pd.DataFrame,
    X_rna_train_df: pd.DataFrame,
    X_rna_val_df: pd.DataFrame,
    X_rna_test_df: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save split-specific modality inputs, targets, and concatenated baseline features.

    Parameters
    ----------
    outdir : str | Path
        Base output directory for assembled split-specific parquet files.
    X_clin_train_df, X_clin_val_df, X_clin_test_df : pd.DataFrame
        Clinical feature matrices for the train, validation, and test splits.
    X_rna_train_df, X_rna_val_df, X_rna_test_df : pd.DataFrame
        RNA feature matrices for the train, validation, and test splits.
    y_train, y_val, y_test : pd.Series
        Target vectors for the train, validation, and test splits.

    Returns
    -------
    tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Output directory plus concatenated train, validation, and test feature
        matrices that were written to disk.
    """
    outdir = Path(outdir)

    # Save split-specific modality inputs and targets for downstream modeling.
    for split_name in ["train", "val", "test"]:
        (outdir / split_name).mkdir(parents=True, exist_ok=True)

    X_clin_train_df.to_parquet(outdir / "train" / "X_clinical.parquet")
    X_clin_val_df.to_parquet(outdir / "val" / "X_clinical.parquet")
    X_clin_test_df.to_parquet(outdir / "test" / "X_clinical.parquet")

    X_rna_train_df.to_parquet(outdir / "train" / "X_rna.parquet")
    X_rna_val_df.to_parquet(outdir / "val" / "X_rna.parquet")
    X_rna_test_df.to_parquet(outdir / "test" / "X_rna.parquet")

    y_train.to_frame(name="y").to_parquet(outdir / "train" / "y.parquet")
    y_val.to_frame(name="y").to_parquet(outdir / "val" / "y.parquet")
    y_test.to_frame(name="y").to_parquet(outdir / "test" / "y.parquet")

    # Optionally save concatenated feature matrices for unimodal-vs-fusion baselines.
    X_concat_train_df = pd.concat([X_clin_train_df, X_rna_train_df], axis=1)
    X_concat_val_df = pd.concat([X_clin_val_df, X_rna_val_df], axis=1)
    X_concat_test_df = pd.concat([X_clin_test_df, X_rna_test_df], axis=1)

    X_concat_train_df.to_parquet(outdir / "train" / "X_concat.parquet")
    X_concat_val_df.to_parquet(outdir / "val" / "X_concat.parquet")
    X_concat_test_df.to_parquet(outdir / "test" / "X_concat.parquet")

    print(f"Saved assembled datasets to: {outdir}")

    return outdir, X_concat_train_df, X_concat_val_df, X_concat_test_df


def main() -> None:
    """Run the multimodal dataset assembly pipeline from command-line inputs."""
    parser = argparse.ArgumentParser(description="Assemble split-specific multimodal datasets for downstream modeling.")
    parser.add_argument(
        "--clinical-dir",
        type=Path,
        default=Path("data/processed/clinical"),
        help="Base directory containing split-specific processed clinical parquet files.",
    )
    parser.add_argument(
        "--rna-dir",
        type=Path,
        default=Path("data/processed/rna"),
        help="Base directory containing split-specific processed RNA parquet files.",
    )
    parser.add_argument(
        "--survival-path",
        type=Path,
        default=Path("data/raw/TCGA-BRCA.survival.tsv.gz"),
        help="Path to the survival TSV or TSV.GZ file.",
    )
    parser.add_argument(
        "--event-col",
        type=str,
        default="OS",
        help="Outcome column to assemble as the target vector.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/processed/assembled"),
        help="Base output directory for assembled split-specific parquet files.",
    )
    args = parser.parse_args()

    (
        X_clin_train_df,
        X_clin_val_df,
        X_clin_test_df,
        X_rna_train_df,
        X_rna_val_df,
        X_rna_test_df,
    ) = load_processed_modality_matrices(
        clinical_dir=args.clinical_dir,
        rna_dir=args.rna_dir,
    )

    (
        X_clin_train_df,
        X_clin_val_df,
        X_clin_test_df,
        X_rna_train_df,
        X_rna_val_df,
        X_rna_test_df,
    ) = align_modalities_within_splits(
        X_clin_train_df=X_clin_train_df,
        X_clin_val_df=X_clin_val_df,
        X_clin_test_df=X_clin_test_df,
        X_rna_train_df=X_rna_train_df,
        X_rna_val_df=X_rna_val_df,
        X_rna_test_df=X_rna_test_df,
    )

    y_train, y_val, y_test = build_target_vectors(
        survival_path=args.survival_path,
        event_col=args.event_col,
        X_clin_train_df=X_clin_train_df,
        X_clin_val_df=X_clin_val_df,
        X_clin_test_df=X_clin_test_df,
    )

    validation_summary_df = validate_dataset_invariants(
        X_clin_train_df=X_clin_train_df,
        X_clin_val_df=X_clin_val_df,
        X_clin_test_df=X_clin_test_df,
        X_rna_train_df=X_rna_train_df,
        X_rna_val_df=X_rna_val_df,
        X_rna_test_df=X_rna_test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    outdir, X_concat_train_df, X_concat_val_df, X_concat_test_df = save_final_model_inputs(
        outdir=args.outdir,
        X_clin_train_df=X_clin_train_df,
        X_clin_val_df=X_clin_val_df,
        X_clin_test_df=X_clin_test_df,
        X_rna_train_df=X_rna_train_df,
        X_rna_val_df=X_rna_val_df,
        X_rna_test_df=X_rna_test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    metadata = {
        "script_name": "assemble_dataset.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "input_files": {
            "clinical_dir": str(args.clinical_dir.resolve()),
            "rna_dir": str(args.rna_dir.resolve()),
            "survival_path": str(args.survival_path.resolve()),
            "clinical_train_path": str((args.clinical_dir / "train" / "X_clinical.parquet").resolve()),
            "clinical_val_path": str((args.clinical_dir / "val" / "X_clinical.parquet").resolve()),
            "clinical_test_path": str((args.clinical_dir / "test" / "X_clinical.parquet").resolve()),
            "rna_train_path": str((args.rna_dir / "train" / "X_rna.parquet").resolve()),
            "rna_val_path": str((args.rna_dir / "val" / "X_rna.parquet").resolve()),
            "rna_test_path": str((args.rna_dir / "test" / "X_rna.parquet").resolve()),
        },
        "output_files": {
            "train_clinical_path": str((outdir / "train" / "X_clinical.parquet").resolve()),
            "val_clinical_path": str((outdir / "val" / "X_clinical.parquet").resolve()),
            "test_clinical_path": str((outdir / "test" / "X_clinical.parquet").resolve()),
            "train_rna_path": str((outdir / "train" / "X_rna.parquet").resolve()),
            "val_rna_path": str((outdir / "val" / "X_rna.parquet").resolve()),
            "test_rna_path": str((outdir / "test" / "X_rna.parquet").resolve()),
            "train_target_path": str((outdir / "train" / "y.parquet").resolve()),
            "val_target_path": str((outdir / "val" / "y.parquet").resolve()),
            "test_target_path": str((outdir / "test" / "y.parquet").resolve()),
            "train_concat_path": str((outdir / "train" / "X_concat.parquet").resolve()),
            "val_concat_path": str((outdir / "val" / "X_concat.parquet").resolve()),
            "test_concat_path": str((outdir / "test" / "X_concat.parquet").resolve()),
        },
        "config_path": None,
        "random_seed": None,
        "key_parameters_used": {
            "event_col": args.event_col,
        },
        "dataset_statistics": {
            "n_samples_train": int(X_clin_train_df.shape[0]),
            "n_samples_val": int(X_clin_val_df.shape[0]),
            "n_samples_test": int(X_clin_test_df.shape[0]),
            "n_clin_features_train": int(X_clin_train_df.shape[1]),
            "n_clin_features_val": int(X_clin_val_df.shape[1]),
            "n_clin_features_test": int(X_clin_test_df.shape[1]),
            "n_rna_features_train": int(X_rna_train_df.shape[1]),
            "n_rna_features_val": int(X_rna_val_df.shape[1]),
            "n_rna_features_test": int(X_rna_test_df.shape[1]),
            "n_concat_features_train": int(X_concat_train_df.shape[1]),
            "n_concat_features_val": int(X_concat_val_df.shape[1]),
            "n_concat_features_test": int(X_concat_test_df.shape[1]),
        },
        "validation_summary": validation_summary_df.to_dict(orient="records"),
    }

    metadata_path = outdir / "assemble_dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
