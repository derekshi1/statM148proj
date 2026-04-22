import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Testing Set
# Note: Ensure the file path is correct for your local environment
file_path = "/Users/derek/UCLA_stats/statM148proj/data/derek_xgb_test_set.parquet" 
df_test = pl.read_parquet(file_path)

# 2. Print Structure and Features
print("--- Test Set Structure ---")
print(df_test.head())
print("\n--- Schema (Features) ---")
print(df_test.schema)

# 3. Basic Journey Analysis
# We calculate the number of actions and the total duration for each journey
df_stats = df_test.with_columns([
    pl.col("journey").list.len().alias("n_actions"),
    ((pl.col("journey").list.last().struct.field("event_timestamp") - 
      pl.col("journey").list.first().struct.field("event_timestamp")).dt.total_seconds() / 3600).alias("duration_hours")
])

print("\n--- Descriptive Statistics ---")
print(df_stats.select([
    pl.col("n_actions").mean().alias("avg_actions"),
    pl.col("n_actions").median().alias("median_actions"),
    pl.col("duration_hours").mean().alias("avg_duration_hrs"),
    pl.col("duration_hours").max().alias("max_duration_hrs")
]))

# 4. Momentum Feature Engineering (1h, 5h, 24h, 72h)
# This calculates how many actions happened within specific time windows from the start
windows = {"1h": 3600, "5h": 18000, "1d": 86400, "3d": 259200}

momentum_exprs = [
    pl.col("journey").list.eval(
        (pl.element().struct.field("event_timestamp") - 
         pl.element().struct.field("event_timestamp").first()).dt.total_seconds() <= seconds
    ).list.sum().alias(f"actions_{name}")
    for name, seconds in windows.items()
]

df_momentum = df_stats.with_columns(momentum_exprs)

# 5. Visualizing Action Distribution
# This helps see if most users are "short-term" or "long-term" browsers
plt.figure(figsize=(10, 5))
sns.histplot(df_momentum['n_actions'].to_pandas(), bins=50, kde=True, color='blue')
plt.title("Distribution of Action Counts per Journey (Test Set)")
plt.xlabel("Number of Actions")
plt.ylabel("Frequency")
plt.show()

# Print a preview of the new momentum features
print("\n--- Momentum Features Preview ---")
print(df_momentum.select(["n_actions", "actions_1h", "actions_5h", "actions_1d"]).head())