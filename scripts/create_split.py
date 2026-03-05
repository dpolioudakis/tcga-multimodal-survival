"""Create stratified train/validation/test splits for TCGA survival modeling.

This module provides reusable utilities to:
1) load aligned sample IDs and outcomes,
2) generate stratified three-way splits,
3) validate split integrity, and
4) persist split IDs to disk.
"""

from pathlib import Path
from collections.abc import Sequence
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


def load_inputs(
    sample_ids_path: str | Path,
    survival_path: str | Path,
    event_col: str,
    id_col: str = "sample",
) -> pd.DataFrame:
    """Load sample IDs and survival labels, then inner-join on the sample ID column."""
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
    """Create stratified train/validation/test ID splits from a labeled dataframe."""
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
    """Validate split integrity and print split sizes plus event rates."""
    # 1) No duplicate sample IDs in source frame
    assert not df[id_col].duplicated().any(), "Duplicate sample IDs found in input dataframe"

    # 2) No missing values in key columns
    key_cols = [id_col, event_col]
    missing_counts = df[key_cols].isna().sum()
    assert (missing_counts == 0).all(), f"Missing values detected in key columns: {missing_counts.to_dict()}"

    # 3) No overlap between train/val/test split IDs
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert train_set.isdisjoint(val_set), "Overlap found between train and val IDs"
    assert train_set.isdisjoint(test_set), "Overlap found between train and test IDs"
    assert val_set.isdisjoint(test_set), "Overlap found between val and test IDs"

    # 4) Dataset sizes match expectations
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
    """Save train, validation, and test sample IDs to CSV files."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.Series(train_ids, name=id_col).to_csv(outdir / "train_ids.csv", index=False)
    pd.Series(val_ids, name=id_col).to_csv(outdir / "val_ids.csv", index=False)
    pd.Series(test_ids, name=id_col).to_csv(outdir / "test_ids.csv", index=False)


def main() -> None:
    """Run the split creation pipeline from CLI arguments."""
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


if __name__ == "__main__":
    main()
