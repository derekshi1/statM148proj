import polars as pl

def flatten_journeys_parquet(input_csv_path, output_parquet_path):
    q = pl.scan_csv(input_csv_path)

    q = q.unique(subset=["id", "ed_id", "event_timestamp"])

    df = (
        q
        .with_columns(
            pl.col("event_timestamp").str.to_datetime(time_zone="UTC")
        )
        .group_by("id")
        .agg(
            # 1. Create the nested journey list
            pl.struct(["event_timestamp", "ed_id"])
            .sort_by(["event_timestamp", "ed_id"])
            .alias("journey"),
            
            # 2. Mark as successful if ed_id 28 exists anywhere in the journey
            (pl.col("ed_id") == 28).any().alias("is_success")
        )
    )

    df.sink_parquet(output_parquet_path)
    
    
training_csv_path = "/Users/emiliodulay/Documents/1. UCLA/2. Year 2/3. Spring 2026/STAT M148/statM148proj/train.csv"
output_path = "/Users/emiliodulay/Documents/1. UCLA/2. Year 2/3. Spring 2026/STAT M148/statM148proj/training_labeled_flattened.parquet"

flatten_journeys_parquet(training_csv_path, output_path)