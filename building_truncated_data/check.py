import polars as pl

# Load your featurized file
DATA_PATH = "/Users/derek/UCLA_stats/statM148proj/data/truncated_data_momentum.parquet"
df = pl.read_parquet(DATA_PATH)

# 1. Check for column existence and nulls
momentum_cols = [ "momentum_5h", "momentum_1d", "momentum_3d"]
print("--- Column Presence & Data Quality ---")
stats = df.select([
    pl.col(c).null_count().alias(f"{c}_nulls") for c in momentum_cols
])
print(stats)

# 2. Verify values aren't all zeros
print("\n--- Value Distribution (Top 10) ---")
print(df.select(["id", "n_events_observed"] + momentum_cols).head(10))

# 3. Logical Sanity Check
# Momentum should be less than or equal to the total events observed
print("\n--- Logical Sanity Check ---")
check = df.select([
    (pl.col("momentum_3d") <= pl.col("n_events_observed")).all().alias("all_momentum_valid"),
    pl.col("momentum_3d").mean().alias("avg_3d"),
    pl.col("n_events_observed").mean().alias("avg_total")
])
print(check)