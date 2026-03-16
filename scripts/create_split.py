"""Create cohort-level train/validation/test splits for TCGA survival modeling.

This module loads the cohort sample manifest and survival labels, derives a
stratified three-way split on the requested event column, validates split
integrity, and writes split ID files plus split metadata for downstream
preprocessing and modeling.

Pipeline:
1. Load cohort sample IDs and survival outcomes.
2. Inner-join the manifests to form the labeled cohort table.
3. Create stratified train, validation, and test splits.
4. Validate split sizes, ID disjointness, and event balance summary.
5. Write split ID files and split metadata.

Inputs:
- Cohort sample ID CSV.
- Survival TSV or TSV.GZ file.
- Stratification column name.
- Split fractions and random seed.

Outputs:
- `train_ids.csv`
- `val_ids.csv`
- `test_ids.csv`
- `split_metadata.json`
"""

from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
import argparse
import hashlib
import json
import shlex

import pandas as pd
from sklearn.model_selection import train_test_split


def load_inputs(
    sample_ids_path: str | Path,
    survival_path: str | Path,
    event_col: str,
    id_col: str = "sample",
) -> pd.DataFrame:
    """Load cohort sample IDs and survival labels, then inner-join on sample ID."""
    ids = pd.read_csv(sample_ids_path)[id_col].astype(str)
    surv = pd.read_csv(survival_path, sep="\t")[[id_col, event_col]].astype({id_col: str})
    id_outcome_df = pd.DataFrame({id_col: ids}).merge(surv, on=id_col, how="inner")
    return id_outcome_df


def make_splits(
    df: pd.DataFrame,
    event_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    id_col: str = "sample",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Create stratified train, validation, and test sample ID splits."""
    y = df[event_col].astype(int)

    train_ids, temp_ids = train_test_split(
        df[id_col],
        test_size=val_size + test_size,
        stratify=y,
        random_state=seed,
    )

    temp = df.set_index(id_col).loc[temp_ids]
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_size / (val_size + test_size),
        stratify=temp[event_col].astype(int),
        random_state=seed,
    )

    return train_ids, val_ids, test_ids


def validate_and_summarize_splits(
    df: pd.DataFrame,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    event_col: str,
    id_col: str = "sample",
    val_size: float = 0.15,
    test_size: float = 0.15,
    size_tol: int = 1,
) -> None:
    """Validate split integrity and print split size and event-rate summaries."""
    # 1) Verify unique source sample IDs.
    assert not df[id_col].duplicated().any(), "Duplicate sample IDs found in input dataframe"

    # 2) Verify required columns are complete.
    key_cols = [id_col, event_col]
    missing_counts = df[key_cols].isna().sum()
    assert (missing_counts == 0).all(), f"Missing values detected in key columns: {missing_counts.to_dict()}"

    # 3) Verify split membership is disjoint.
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert train_set.isdisjoint(val_set), "Overlap found between train and val IDs"
    assert train_set.isdisjoint(test_set), "Overlap found between train and test IDs"
    assert val_set.isdisjoint(test_set), "Overlap found between val and test IDs"

    # 4) Verify observed split sizes are consistent with requested fractions.
    n_total = len(df)
    expected_test = round(n_total * test_size)
    expected_val = round(n_total * val_size)
    expected_train = n_total - expected_val - expected_test

    actual_train, actual_val, actual_test = len(train_ids), len(val_ids), len(test_ids)
    assert actual_train + actual_val + actual_test == n_total, "Split sizes do not sum to total dataset size"
    assert abs(actual_train - expected_train) <= size_tol, f"Train size mismatch: expected ~{expected_train}, got {actual_train}"
    assert abs(actual_val - expected_val) <= size_tol, f"Val size mismatch: expected ~{expected_val}, got {actual_val}"
    assert abs(actual_test - expected_test) <= size_tol, f"Test size mismatch: expected ~{expected_test}, got {actual_test}"

    y = df.set_index(id_col)[event_col].astype(int)
    print("All validation checks passed.")
    print(f"n_total={n_total}, train={actual_train}, val={actual_val}, test={actual_test}")
    print("event rates:", y.loc[train_ids].mean(), y.loc[val_ids].mean(), y.loc[test_ids].mean())


def save_splits(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    outdir: str | Path,
    id_col: str = "sample",
) -> None:
    """Write train, validation, and test sample ID files to the output directory."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.Series(train_ids, name=id_col).to_csv(outdir / "train_ids.csv", index=False)
    pd.Series(val_ids, name=id_col).to_csv(outdir / "val_ids.csv", index=False)
    pd.Series(test_ids, name=id_col).to_csv(outdir / "test_ids.csv", index=False)


def write_split_metadata(
    outdir: str | Path,
    sample_ids_path: str | Path,
    survival_path: str | Path,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    seed: int,
    val_size: float,
    test_size: float,
    stratify_col: str,
    command: str,
    id_col: str = "sample",
) -> Path:
    """Write split metadata, input provenance, and the CLI command to JSON."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample_ids_path = Path(sample_ids_path)
    survival_path = Path(survival_path)

    # Hash raw inputs and split membership to record the split provenance.
    hash_builder = hashlib.sha256()
    hash_builder.update(sample_ids_path.read_bytes())
    hash_builder.update(survival_path.read_bytes())
    hash_builder.update("\n".join(map(str, train_ids)).encode("utf-8"))
    hash_builder.update("\n".join(map(str, val_ids)).encode("utf-8"))
    hash_builder.update("\n".join(map(str, test_ids)).encode("utf-8"))

    input_manifest = {
        "sample_ids_path": str(sample_ids_path.resolve()),
        "survival_path": str(survival_path.resolve()),
    }

    cohort_size = len(train_ids) + len(val_ids) + len(test_ids)
    metadata = {
        "seed": seed,
        "fractions": {
            "val": val_size,
            "test": test_size,
            "train": 1.0 - val_size - test_size,
        },
        "stratify_col": stratify_col,
        "id_col": id_col,
        "cohort_size": cohort_size,
        "split_sizes": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": input_manifest,
        "input_manifest_hash_sha256": hash_builder.hexdigest(),
        "command": command,
    }

    metadata_path = outdir / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    """Run the split creation pipeline from command-line inputs."""
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits for TCGA survival modeling.")
    parser.add_argument("--sample-ids-path", type=Path, required=True, help="Path to CSV with sample IDs.")
    parser.add_argument("--survival-path", type=Path, required=True, help="Path to TSV/TSV.GZ survival file.")
    parser.add_argument("--event-col", type=str, required=True, help="Name of binary event column in survival file.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for split ID CSV files.")
    parser.add_argument("--id-col", type=str, default="sample", help="Sample ID column name.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    parser.add_argument("--size-tol", type=int, default=1, help="Allowed mismatch from expected split sizes.")
    args = parser.parse_args()

    id_outcome_df = load_inputs(
        sample_ids_path=args.sample_ids_path,
        survival_path=args.survival_path,
        event_col=args.event_col,
        id_col=args.id_col,
    )

    train_ids, val_ids, test_ids = make_splits(
        df=id_outcome_df,
        event_col=args.event_col,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        id_col=args.id_col,
    )

    validate_and_summarize_splits(
        df=id_outcome_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        event_col=args.event_col,
        id_col=args.id_col,
        val_size=args.val_size,
        test_size=args.test_size,
        size_tol=args.size_tol,
    )

    save_splits(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        outdir=args.outdir,
        id_col=args.id_col,
    )

    write_split_metadata(
        outdir=args.outdir,
        sample_ids_path=args.sample_ids_path,
        survival_path=args.survival_path,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        stratify_col=args.event_col,
        command=shlex.join(
            [
                "python",
                "scripts/create_split.py",
                "--sample-ids-path",
                str(args.sample_ids_path),
                "--survival-path",
                str(args.survival_path),
                "--event-col",
                args.event_col,
                "--outdir",
                str(args.outdir),
                "--id-col",
                args.id_col,
                "--val-size",
                str(args.val_size),
                "--test-size",
                str(args.test_size),
                "--seed",
                str(args.seed),
                "--size-tol",
                str(args.size_tol),
            ]
        ),
        id_col=args.id_col,
    )


if __name__ == "__main__":
    main()
