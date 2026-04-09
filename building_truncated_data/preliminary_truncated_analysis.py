from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


DATA_PATH = Path("/Users/derek/UCLA_stats/statM148proj/truncated_data_codex.parquet")

NUMERIC_SUMMARY_COLS = [
    "sampling_weight_days",
    "n_events_observed",
    "n_unique_event_types_observed",
    "observed_duration_days",
    "days_since_last_event",
    "cutoff_hour_utc",
    "cutoff_dayofweek_utc",
    "cutoff_month_utc",
    "cutoff_dayofyear_utc",
    "max_milestone_seen",
]

CATEGORICAL_SUMMARY_COLS = [
    "journey_terminal_reason",
    "first_event_name",
    "first_stage",
    "last_event_name",
    "last_stage",
]


def read_sample_rows(path: Path, n_rows: int = 200_000) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    collected = []
    rows_left = n_rows

    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx)
        batch = table.to_pandas()
        if len(batch) >= rows_left:
            collected.append(batch.iloc[:rows_left].copy())
            break
        collected.append(batch)
        rows_left -= len(batch)

    return pd.concat(collected, ignore_index=True)


def print_numeric_summary(df: pd.DataFrame):
    print("\n--- Numeric Summary For New Truncation Variables ---")
    summary = df[NUMERIC_SUMMARY_COLS].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).T
    print(summary[["mean", "std", "min", "25%", "50%", "75%", "90%", "max"]].round(2))


def print_categorical_summary(df: pd.DataFrame):
    print("\n--- Categorical Snapshot Summary ---")
    for col in CATEGORICAL_SUMMARY_COLS:
        print(f"\nTop values for {col}:")
        print(df[col].fillna("MISSING").value_counts().head(10).to_string())


def print_event_feature_summary(df: pd.DataFrame):
    seen_cols = [col for col in df.columns if col.startswith("seen_ed_")]
    count_cols = [col for col in df.columns if col.startswith("count_ed_")]

    seen_rates = (
        df[seen_cols]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .rename("share_of_snapshots_seen")
    )
    count_means = (
        df[count_cols]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .rename("avg_count_per_snapshot")
    )

    print("\n--- Most Common Event Indicators In Truncated Snapshots ---")
    print(seen_rates.round(4).to_string())

    print("\n--- Highest Average Event Counts Per Snapshot ---")
    print(count_means.round(2).to_string())


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find parquet file: {DATA_PATH}")

    parquet_file = pq.ParquetFile(DATA_PATH)
    total_rows = parquet_file.metadata.num_rows
    total_columns = len(parquet_file.schema_arrow.names)

    sample_df = read_sample_rows(DATA_PATH)

    print("--- Truncated Data Preliminary Analysis ---")
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {total_columns}")
    print(f"Sample used for summaries: {len(sample_df):,} rows")
    print()
    print(f"Positive rate (sample): {sample_df['label_order_shipped'].mean():.4f}")
    print("Terminal reason distribution (sample):")
    print(
        sample_df["journey_terminal_reason"]
        .value_counts(normalize=True)
        .round(4)
        .mul(100)
        .rename("percent")
        .to_string()
    )

    print_numeric_summary(sample_df)
    print_categorical_summary(sample_df)
    print_event_feature_summary(sample_df)



if __name__ == "__main__":
    main()
