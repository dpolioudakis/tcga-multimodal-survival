"""Train multimodal deep learning models for TCGA survival prediction.

This module trains two fusion models — a concatenation MLP and an attention fusion model —
on separate RNA and clinical feature matrices, evaluates discrimination on validation and
test splits, and saves model artifacts, predictions, metrics, and run metadata.

Pipeline:
1. Load assembled train/val/test clinical and RNA feature matrices and targets.
2. Convert to PyTorch tensors and build a training DataLoader.
3. Train concatenation fusion model with BCE loss and Adam optimizer.
4. Train attention fusion model with BCE loss and Adam optimizer.
5. Evaluate both models on val and test (ROC-AUC, AP).
6. Save model artifacts, predictions, metrics, and metadata.

Inputs:
- Assembled parquet files from assemble_dataset.py output directory.

Outputs:
- `concat_model.pkl`
- `attn_model.pkl`
- `predictions_val.parquet`
- `predictions_test.parquet`
- `metrics.json`
- `train_multimodal_metadata.json`
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
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def load_datasets(
    assembled_dir: Path,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """Load assembled train/val/test feature matrices and targets from parquet files.

    Args:
        assembled_dir: Path to the assembled dataset directory.

    Returns:
        X_clin_train_df, X_clin_val_df, X_clin_test_df: clinical feature matrices.
        X_rna_train_df, X_rna_val_df, X_rna_test_df: RNA feature matrices.
        y_train, y_val, y_test: binary outcome vectors.
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
        X_rna_train_df,  X_rna_val_df,  X_rna_test_df,
        y_train, y_val, y_test,
    )


def to_tensors(
    X_rna_df: pd.DataFrame,
    X_clin_df: pd.DataFrame,
    y: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert RNA, clinical, and label DataFrames to float32 PyTorch tensors.

    Args:
        X_rna_df: RNA feature matrix.
        X_clin_df: Clinical feature matrix.
        y: Binary outcome vector.

    Returns:
        rna_tensor, clin_tensor, y_tensor as float32 tensors.
    """
    return (
        torch.tensor(X_rna_df.values,  dtype=torch.float32),
        torch.tensor(X_clin_df.values, dtype=torch.float32),
        torch.tensor(y.values,         dtype=torch.float32),
    )


class RNAEncoder(nn.Module):
    """MLP encoder for RNA features: 25k → 128 → 64."""

    def __init__(self, input_dim: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClinicalEncoder(nn.Module):
    """MLP encoder for clinical features: 98 → 64 → 64."""

    def __init__(self, input_dim: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 64),        nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConcatFusionModel(nn.Module):
    """Concatenate RNA and clinical embeddings, pass through classification head."""

    def __init__(self, rna_encoder: nn.Module, clin_encoder: nn.Module, dropout: float = 0.5):
        super().__init__()
        self.rna_encoder  = rna_encoder
        self.clin_encoder = clin_encoder
        fused_dim = 64 + 64
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x_rna: torch.Tensor, x_clin: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.rna_encoder(x_rna), self.clin_encoder(x_clin)], dim=1)
        return self.head(emb).squeeze(1)


class AttentionFusionModel(nn.Module):
    """Modality-level attention fusion: learns per-patient weights over RNA and clinical embeddings.

    Both embeddings are projected to the same dimension before attention is applied.
    """

    def __init__(
        self,
        rna_encoder: nn.Module,
        clin_encoder: nn.Module,
        embed_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.rna_encoder  = rna_encoder
        self.clin_encoder = clin_encoder
        self.clin_proj    = nn.Linear(64, embed_dim)
        self.attention    = nn.Linear(embed_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x_rna: torch.Tensor, x_clin: torch.Tensor) -> torch.Tensor:
        rna_emb  = self.rna_encoder(x_rna)
        clin_emb = self.clin_proj(self.clin_encoder(x_clin))

        stacked = torch.stack([rna_emb, clin_emb], dim=1)         # (B, 2, 64)
        weights = torch.softmax(self.attention(stacked), dim=1)    # (B, 2, 1)
        fused   = (stacked * weights).sum(dim=1)                   # (B, 64)

        return self.head(fused).squeeze(1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    rna_val: torch.Tensor,
    clin_val: torch.Tensor,
    y_val_t: torch.Tensor,
    n_epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[list[float], list[float]]:
    """Train a fusion model and return per-epoch train and val losses.

    Args:
        model: Fusion model to train.
        train_loader: DataLoader for training batches.
        rna_val: RNA validation tensor for loss monitoring.
        clin_val: Clinical validation tensor for loss monitoring.
        y_val_t: Validation labels tensor.
        n_epochs: Number of training epochs.
        lr: Adam learning rate.
        weight_decay: L2 regularization strength.

    Returns:
        train_losses, val_losses: per-epoch loss lists.
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses, val_losses = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        batch_losses = []
        for rna_batch, clin_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(rna_batch, clin_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(rna_val, clin_val), y_val_t).item()

        train_losses.append(sum(batch_losses) / len(batch_losses))
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} — train loss: {train_losses[-1]:.4f}  val loss: {val_loss:.4f}")

    return train_losses, val_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal deep learning models.")
    parser.add_argument("--assembled-dir", required=True, help="Path to assembled dataset directory.")
    parser.add_argument("--outdir", required=True, help="Output directory for artifacts.")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for encoders.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    assembled_dir = Path(args.assembled_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    (
        X_clin_train_df, X_clin_val_df, X_clin_test_df,
        X_rna_train_df,  X_rna_val_df,  X_rna_test_df,
        y_train, y_val, y_test,
    ) = load_datasets(assembled_dir)

    n_clin_features = X_clin_train_df.shape[1]
    n_rna_features  = X_rna_train_df.shape[1]

    # Convert to tensors
    rna_train, clin_train, y_train_t = to_tensors(X_rna_train_df, X_clin_train_df, y_train)
    rna_val,   clin_val,   y_val_t   = to_tensors(X_rna_val_df,   X_clin_val_df,   y_val)
    rna_test,  clin_test,  _         = to_tensors(X_rna_test_df,  X_clin_test_df,  y_test)

    train_loader = DataLoader(
        TensorDataset(rna_train, clin_train, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Train concatenation fusion model
    print("\n--- Concatenation fusion model ---")
    concat_model = ConcatFusionModel(
        RNAEncoder(n_rna_features, dropout=args.dropout),
        ClinicalEncoder(n_clin_features, dropout=args.dropout),
        dropout=args.dropout,
    )
    train_model(concat_model, train_loader, rna_val, clin_val, y_val_t, args.n_epochs, args.lr, args.weight_decay)

    # Train attention fusion model
    print("\n--- Attention fusion model ---")
    attn_model = AttentionFusionModel(
        RNAEncoder(n_rna_features, dropout=args.dropout),
        ClinicalEncoder(n_clin_features, dropout=args.dropout),
    )
    train_model(attn_model, train_loader, rna_val, clin_val, y_val_t, args.n_epochs, args.lr, args.weight_decay)

    # Generate predictions
    concat_model.eval()
    attn_model.eval()
    with torch.no_grad():
        y_val_pred_concat  = concat_model(rna_val,  clin_val).detach().cpu().numpy()
        y_val_pred_attn    = attn_model(rna_val,   clin_val).detach().cpu().numpy()
        y_test_pred_concat = concat_model(rna_test, clin_test).detach().cpu().numpy()
        y_test_pred_attn   = attn_model(rna_test,  clin_test).detach().cpu().numpy()

    # Validation checks
    assert len(y_val_pred_concat)  == len(y_val),  "concat val size mismatch"
    assert len(y_val_pred_attn)    == len(y_val),  "attn val size mismatch"
    assert len(y_test_pred_concat) == len(y_test), "concat test size mismatch"
    assert len(y_test_pred_attn)   == len(y_test), "attn test size mismatch"
    assert len(set(X_rna_train_df.index) & set(X_rna_val_df.index))  == 0, "train/val overlap"
    assert len(set(X_rna_train_df.index) & set(X_rna_test_df.index)) == 0, "train/test overlap"
    assert len(set(X_rna_val_df.index)   & set(X_rna_test_df.index)) == 0, "val/test overlap"

    # Compute metrics
    val_roc_auc_concat  = roc_auc_score(y_val,  y_val_pred_concat)
    val_ap_concat       = average_precision_score(y_val,  y_val_pred_concat)
    val_roc_auc_attn    = roc_auc_score(y_val,  y_val_pred_attn)
    val_ap_attn         = average_precision_score(y_val,  y_val_pred_attn)
    test_roc_auc_concat = roc_auc_score(y_test, y_test_pred_concat)
    test_ap_concat      = average_precision_score(y_test, y_test_pred_concat)
    test_roc_auc_attn   = roc_auc_score(y_test, y_test_pred_attn)
    test_ap_attn        = average_precision_score(y_test, y_test_pred_attn)

    print(f"\n--- Validation ---")
    print(f"Concat MLP  — ROC-AUC: {val_roc_auc_concat:.3f}  AP: {val_ap_concat:.3f}")
    print(f"Attention   — ROC-AUC: {val_roc_auc_attn:.3f}  AP: {val_ap_attn:.3f}")
    print(f"\n--- Test ---")
    print(f"Concat MLP  — ROC-AUC: {test_roc_auc_concat:.3f}  AP: {test_ap_concat:.3f}")
    print(f"Attention   — ROC-AUC: {test_roc_auc_attn:.3f}  AP: {test_ap_attn:.3f}")

    # Save model artifacts
    with open(outdir / "concat_model.pkl", "wb") as f:
        pickle.dump(concat_model, f)
    with open(outdir / "attn_model.pkl", "wb") as f:
        pickle.dump(attn_model, f)

    # Save predictions
    pd.DataFrame({
        "sample":          X_clin_val_df.index,
        "y_true":          y_val.values,
        "y_pred_concat":   y_val_pred_concat,
        "y_pred_attn":     y_val_pred_attn,
    }).to_parquet(outdir / "predictions_val.parquet", index=False)

    pd.DataFrame({
        "sample":          X_clin_test_df.index,
        "y_true":          y_test.values,
        "y_pred_concat":   y_test_pred_concat,
        "y_pred_attn":     y_test_pred_attn,
    }).to_parquet(outdir / "predictions_test.parquet", index=False)

    # Save metrics
    metrics = {
        "val": {
            "concat": {"roc_auc": round(val_roc_auc_concat, 4), "ap": round(val_ap_concat, 4)},
            "attn":   {"roc_auc": round(val_roc_auc_attn,   4), "ap": round(val_ap_attn,   4)},
        },
        "test": {
            "concat": {"roc_auc": round(test_roc_auc_concat, 4), "ap": round(test_ap_concat, 4)},
            "attn":   {"roc_auc": round(test_roc_auc_attn,   4), "ap": round(test_ap_attn,   4)},
        },
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save metadata
    metadata = {
        "script_name": "train_multimodal.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_command": shlex.join(sys.argv),
        "input_files": {
            "assembled_dir": str(assembled_dir.resolve()),
        },
        "key_parameters_used": {
            "random_state": args.random_state,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "optimizer": "Adam",
            "loss": "BCELoss",
        },
        "dataset_statistics": {
            "n_samples_train": int(len(y_train)),
            "n_samples_val":   int(len(y_val)),
            "n_samples_test":  int(len(y_test)),
            "n_events_train":  int(y_train.sum()),
            "n_events_val":    int(y_val.sum()),
            "n_events_test":   int(y_test.sum()),
            "n_clin_features": int(n_clin_features),
            "n_rna_features":  int(n_rna_features),
            "val_roc_auc_concat":  round(val_roc_auc_concat,  4),
            "val_roc_auc_attn":    round(val_roc_auc_attn,    4),
            "test_roc_auc_concat": round(test_roc_auc_concat, 4),
            "test_roc_auc_attn":   round(test_roc_auc_attn,   4),
        },
        "output_files": {
            "concat_model":      str(outdir / "concat_model.pkl"),
            "attn_model":        str(outdir / "attn_model.pkl"),
            "predictions_val":   str(outdir / "predictions_val.parquet"),
            "predictions_test":  str(outdir / "predictions_test.parquet"),
            "metrics":           str(outdir / "metrics.json"),
        },
    }
    (outdir / "train_multimodal_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nArtifacts saved to {outdir}")


if __name__ == "__main__":
    main()
