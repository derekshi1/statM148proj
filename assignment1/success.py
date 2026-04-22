import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
# 1. Load the data
df = pl.read_parquet("/Users/derek/UCLA_stats/statM148proj/data/training_new_data.parquet")
SUCCESS_ID = 28  # order_shipped

print("Analyzing success cases...")

# 2. Create the 'Label' and calculate journey metrics
# We calculate these for the whole dataset since Polars is fast enough
df_stats = df.with_columns([
    # Did they ever have the success ID?
    pl.col("journey").list.eval(pl.element().struct.field("ed_id") == SUCCESS_ID).list.any().alias("is_success"),
    
    # Journey length in actions
    pl.col("journey").list.len().alias("n_actions"),
    
    # Journey duration in days
    ((pl.col("journey").list.last().struct.field("event_timestamp") - 
      pl.col("journey").list.first().struct.field("event_timestamp")).dt.total_seconds() / 86400).alias("duration_days")
])

# 3. Calculate Overall Conversion Rate
windows = {
    "1h": 3600,
    "5h": 5 * 3600,
    "1d": 24 * 3600,
    "3d": 72 * 3600
}

# 1. Calculate momentum for each window
momentum_expressions = []
for name, seconds in windows.items():
    expr = pl.col("journey").list.eval(
        (pl.element().struct.field("event_timestamp") - 
         pl.element().struct.field("event_timestamp").first()).dt.total_seconds() <= seconds
    ).list.sum().alias(f"actions_{name}")
    momentum_expressions.append(expr)

df_momentum = df_stats.with_columns(momentum_expressions)

# 2. Reshape for Visualization
# We want to see the average actions per window split by success
vis_data = df_momentum.select([
    "is_success", 
    *[f"actions_{name}" for name in windows.keys()]
]).group_by("is_success").mean().to_pandas()

# 3. Plotting
vis_data_melted = vis_data.melt(id_vars="is_success", var_name="window", value_name="avg_actions")

plt.figure(figsize=(10, 6))
sns.barplot(data=vis_data_melted, x="window", y="avg_actions", hue="is_success")
plt.title("Action Momentum: Success vs. Non-Success")
plt.ylabel("Average Number of Actions")
plt.xlabel("Time Since First Event")
plt.show()