import polars as pl

# 1. Load the data
df = pl.read_parquet("training_new_data.parquet")
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
total_users = df_stats.height
success_users = df_stats.filter(pl.col("is_success")).height
conversion_rate = (success_users / total_users) * 100

print(f"--- Global Conversion Stats ---")
print(f"Total Unique Users: {total_users:,}")
print(f"Successful Users (Orders): {success_users:,}")
print(f"Conversion Rate: {conversion_rate:.2f}%")

# 4. Compare Success vs. Non-Success Behavior
comparison = df_stats.group_by("is_success").agg([
    pl.col("n_actions").median().alias("median_actions"),
    pl.col("n_actions").mean().alias("avg_actions"),
    pl.col("duration_days").median().alias("median_days"),
    pl.col("duration_days").mean().alias("avg_days")
]).sort("is_success")

print("\n--- Success vs. Non-Success Comparison ---")
print(comparison)

# 5. What are the most common actions for SUCCESSFUL users only?
print("\nTop actions for Successful Users...")
event_defs = pl.read_csv("Event Definitions.csv")

top_success_actions = (
    df_stats.filter(pl.col("is_success"))
    .select(pl.col("journey").list.explode())
    .unnest("journey")
    .group_by("ed_id")
    .len()
    .join(event_defs, left_on="ed_id", right_on="event_definition_id")
    .sort("len", descending=True)
    .head(10)

    
)

print(top_success_actions.select(["event_name", "len", "stage"]))