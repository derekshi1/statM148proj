#This is for the groups using Python. One of the groups last quarter shared code for ram efficient polars/pyarrow code to create nested journeys. 
# Use this instead of pandas for better performance:


import polars as pl

training_csv_path = "/Users/emiliodulay/Documents/1. UCLA/2. Year 2/3. Spring 2026/STAT M148/statM148proj/test.csv"
output_path = "/Users/emiliodulay/Documents/1. UCLA/2. Year 2/3. Spring 2026/STAT M148/statM148proj/testing_new_data.parquet"

def flatten_journeys_parquet(input_csv_path, output_parquet_path):
    """
    Removes duplicates and flattens a event csv file with journey stored as structs ["event_timestamp", "ed_id"]
    """
    q = pl.scan_csv(input_csv_path)

    q = q.unique(subset=["id", "ed_id", "event_timestamp"])

    df = (
        q
        .with_columns(
            pl.col("event_timestamp").str.to_datetime(time_zone="UTC")
        )
        .group_by("id")
        .agg(
            pl.struct(["event_timestamp", "ed_id"])
            .sort_by(["event_timestamp", "ed_id"])
            .alias("journey")
        )
    )

    df.sink_parquet(output_parquet_path)

flatten_journeys_parquet(training_csv_path, output_path)

df = pl.read_parquet(output_path) 