"""
dataset.py
==========
Preprocessing pipeline: Polars journey DataFrame → PyTorch Dataset

Expected input schema
---------------------
shape: (N, 2)
┌────────────────────────┬──────────────────────────────────┐
│ id                     ┆ journey                          │
│ str                    ┆ list[struct[timestamp, value]]   │
└────────────────────────┴──────────────────────────────────┘

Each row is one sequence (one journey). The 'journey' column is a
variable-length list of structs — each struct is one time step.

Adjust STRUCT_TIMESTAMP_COL and STRUCT_VALUE_COL to match your
actual struct field names (inspect with df['journey'][0] in Polars).
"""

import torch
import numpy as np
import polars as pl
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from typing import Optional


# ── Field names inside the struct ───────────────────────────────────────────
# Change these to match your actual schema.
STRUCT_TIMESTAMP_COL = "event_timestamp"   # datetime field name inside struct
STRUCT_VALUE_COL     = "ed_id"       # integer/numeric field name inside struct


# ---------------------------------------------------------------------------
# Step 1 — Unnest and extract features from each journey
# ---------------------------------------------------------------------------

def extract_features_from_journey(df: pl.DataFrame) -> pl.DataFrame:
    """
    Explode the 'journey' list column, unnest the struct, and derive
    numeric features from the timestamp and value fields.

    Input:  (N rows, 2 cols: id, journey)
    Output: (M rows, several feature cols + id) where M = sum of journey lengths
    """
    return (
        df
        # Explode list → one row per time step, 'id' repeated
        .explode("journey")

        # Unnest struct → separate columns: id, timestamp, value
        .unnest("journey")

        # Cast timestamp to datetime if not already
        .with_columns(
            pl.col(STRUCT_TIMESTAMP_COL).cast(pl.Datetime("us", "UTC"))
        )

        # Sort within each journey chronologically
        .sort(["id", STRUCT_TIMESTAMP_COL])

        # ── Timestamp features ───────────────────────────────────────────
        # These encode *when* each step happened. Add/remove as needed.
        .with_columns([
            pl.col(STRUCT_TIMESTAMP_COL).dt.hour()
              .alias("hour_of_day"),

            pl.col(STRUCT_TIMESTAMP_COL).dt.weekday()
              .alias("day_of_week"),                    # 0 = Monday … 6 = Sunday

            pl.col(STRUCT_TIMESTAMP_COL).dt.month()
              .alias("month"),

            # Time-delta to previous step (seconds) — captures event spacing
            (
                pl.col(STRUCT_TIMESTAMP_COL)
                  .diff()
                  .over("id")
                  .dt.total_seconds()
                  .fill_null(0)
                  .alias("delta_seconds")
            ),

            # Step index within journey — positional context
            pl.col(STRUCT_TIMESTAMP_COL)
              .cum_count()
              .over("id")
              .alias("step_index"),
        ])

        # ── Cyclic encoding of hour and day ─────────────────────────────
        # Wraps 23:59 → 00:00 smoothly. Always better than raw integers
        # for periodic features.
        .with_columns([
            (2 * np.pi * pl.col("hour_of_day") / 24).sin().alias("hour_sin"),
            (2 * np.pi * pl.col("hour_of_day") / 24).cos().alias("hour_cos"),
            (2 * np.pi * pl.col("day_of_week") / 7).sin().alias("dow_sin"),
            (2 * np.pi * pl.col("day_of_week") / 7).cos().alias("dow_cos"),
        ])

        .select([
            "id",
            STRUCT_TIMESTAMP_COL,
            STRUCT_VALUE_COL,     # raw value — will be one-hot or embedded later
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
            "month",
            "delta_seconds",
            "step_index",
        ])
    )


# ---------------------------------------------------------------------------
# Step 2 — Build per-journey feature matrices
# ---------------------------------------------------------------------------

# These are the numeric columns fed into the model.
# Adjust: add columns you extracted above, remove ones you don't want.
FEATURE_COLS = [
    "hour_sin","hour_cos",
    "dow_sin", "dow_cos",
    "month",
    "delta_seconds",
    "step_index",
    STRUCT_VALUE_COL,
]

def build_sequences(exploded) -> dict[str, np.ndarray]:
    if isinstance(exploded, pl.LazyFrame):
        exploded = exploded.collect()

    # partition_by creates a list of sub-DataFrames split by 'id'
    # as_dict=True gives us { (id_val,): sub_df }
    parts = exploded.partition_by("id", as_dict=True, maintain_order=True)
    
    # Use a dictionary comprehension for speed
    return {
        id_tuple[0]: group.select(FEATURE_COLS).to_numpy().astype(np.float32)
        for id_tuple, group in parts.items()
    }


# ---------------------------------------------------------------------------
# Step 3 — Fit a scaler on all training sequences
# ---------------------------------------------------------------------------

def fit_scaler(sequences: dict[str, np.ndarray]) -> StandardScaler:
    """
    Fit a StandardScaler across all time steps from all sequences.
    Call this ONLY on the training split, then apply to val/test.
    """
    all_steps = np.vstack(list(sequences.values()))   # (total_steps, num_features)
    scaler = StandardScaler()
    scaler.fit(all_steps)
    return scaler


def apply_scaler(
    sequences: dict[str, np.ndarray],
    scaler: StandardScaler,
) -> dict[str, np.ndarray]:
    return {
        k: scaler.transform(v).astype(np.float32)
        for k, v in sequences.items()
    }


# ---------------------------------------------------------------------------
# Step 4 — PyTorch Dataset
# ---------------------------------------------------------------------------

class JourneyDataset(Dataset):
    """
    Converts variable-length journey sequences into fixed-length padded tensors.

    Parameters
    ----------
    sequences : dict[id → (seq_len, num_features) np.ndarray]
    labels    : dict[id → int]  — class label for each journey
    max_seq_len : int
        Sequences longer than this are *truncated from the right*.
        Sequences shorter are *right-padded with zeros*.
        A padding mask is returned so the model ignores pad positions.
    """

    def __init__(
        self,
        sequences: dict[str, np.ndarray],
        labels: dict[str, int],
        max_seq_len: int = 512,
    ):
        self.max_seq_len = max_seq_len
        self.ids = list(sequences.keys())
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        journey_id = self.ids[idx]
        seq = self.sequences[journey_id]          # (raw_len, num_features)

        # Truncate
        seq = seq[: self.max_seq_len]
        actual_len = len(seq)

        # Pad to max_seq_len
        pad_len = self.max_seq_len - actual_len
        if pad_len > 0:
            padding = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, padding], axis=0)

        # Padding mask: True = position should be ignored by attention
        padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        if pad_len > 0:
            padding_mask[actual_len:] = True

        return {
            "src":          torch.from_numpy(seq),         # (max_seq_len, num_features)
            "padding_mask": padding_mask,                  # (max_seq_len,)
            "label":        torch.tensor(self.labels[journey_id], dtype=torch.long),
            "id":           journey_id,
        }


# ---------------------------------------------------------------------------
# Step 5 — DataLoader with custom collate
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict[str, Tensor]:
    """Stack items into batch tensors."""
    return {
        "src":          torch.stack([b["src"] for b in batch]),           # (B, T, F)
        "padding_mask": torch.stack([b["padding_mask"] for b in batch]),  # (B, T)
        "label":        torch.stack([b["label"] for b in batch]),         # (B,)
        "id":           [b["id"] for b in batch],
    }


def make_dataloader(
    dataset: JourneyDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Step 6 — End-to-end helper
# ---------------------------------------------------------------------------

def prepare_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_labels: dict[str, int],
    val_labels: dict[str, int],
    test_labels: dict[str, int],
    max_seq_len: int = 512,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Full pipeline from raw Polars DataFrames → train/val/test DataLoaders.

    Parameters
    ----------
    train_df / val_df / test_df : Polars DataFrames with 'id' and 'journey' columns
    *_labels : dict mapping id string → integer class label
    max_seq_len : fixed sequence length (truncate / pad to this)
    batch_size  : samples per batch

    Returns
    -------
    train_loader, val_loader, test_loader, scaler
    """
    # Feature extraction
    train_exp = extract_features_from_journey(train_df)
    val_exp   = extract_features_from_journey(val_df)
    test_exp  = extract_features_from_journey(test_df)

    # Build sequence dicts
    train_seqs = build_sequences(train_exp)
    val_seqs   = build_sequences(val_exp)
    test_seqs  = build_sequences(test_exp)

    # Fit scaler on training data only, apply everywhere
    scaler     = fit_scaler(train_seqs)
    train_seqs = apply_scaler(train_seqs, scaler)
    val_seqs   = apply_scaler(val_seqs, scaler)
    test_seqs  = apply_scaler(test_seqs, scaler)

    # Build Datasets
    train_ds = JourneyDataset(train_seqs, train_labels, max_seq_len)
    val_ds   = JourneyDataset(val_seqs,   val_labels,   max_seq_len)
    test_ds  = JourneyDataset(test_seqs,  test_labels,  max_seq_len)

    return (
        make_dataloader(train_ds, batch_size, shuffle=True),
        make_dataloader(val_ds,   batch_size, shuffle=False),
        make_dataloader(test_ds,  batch_size, shuffle=False),
        scaler,
    )


# ---------------------------------------------------------------------------
# Wiring to the Transformer model
# ---------------------------------------------------------------------------

def train_step(model, batch: dict, optimizer, criterion, device):
    """
    One training step. Shows how the batch dict maps to model inputs.
    """
    src   = batch["src"].to(device)            # (B, T, num_features)
    mask  = batch["padding_mask"].to(device)   # (B, T)  True = ignore
    label = batch["label"].to(device)          # (B,)

    logits = model(src, src_padding_mask=mask)  # (B, num_classes)
    loss   = criterion(logits, label)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Quick diagnostic — run to inspect your data before training
# ---------------------------------------------------------------------------

def inspect(df: pl.DataFrame, n: int = 3):
    """Print the first n rows after feature extraction to spot issues."""
    exp = extract_features_from_journey(df.head(n * 20))
    seqs = build_sequences(exp)
    for i, (k, v) in enumerate(seqs.items()):
        print(f"\nid: {k}")
        print(f"  raw sequence length : {v.shape[0]} steps")
        print(f"  num_features        : {v.shape[1]}")
        print(f"  feature means       : {v.mean(axis=0).round(3)}")
        print(f"  feature stds        : {v.std(axis=0).round(3)}")
        if i >= n - 1:
            break
    print(f"\nFeature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")