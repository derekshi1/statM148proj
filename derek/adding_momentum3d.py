import polars as pl

# 1. Load your parquet file
input_file = "/Users/derek/UCLA_stats/statM148proj/data/truncated_data_codex.parquet"
output_file = "/Users/derek/UCLA_stats/statM148proj/data/truncated_data_with_momentum.parquet"

df = pl.read_parquet(input_file)

# 2. Define the window (3 days in seconds)
SECONDS_IN_3_DAYS = 3 * 24 * 3600

# 3. Calculate momentum_3d
# This counts how many events occur within 72 hours of the very first event in the journey
df_with_feature = df.with_columns(
    momentum_3d = pl.col("journey").list.eval(
        (pl.element().struct.field("event_timestamp") - 
         pl.element().struct.field("event_timestamp").first())
        .dt.total_seconds() <= SECONDS_IN_3_DAYS
    ).list.sum()
)

# 4. Save the result
df_with_feature.write_parquet(output_file)

# Preview results
print("Feature added successfully!")
print(df_with_feature.select(["id", "momentum_3d"]).head())