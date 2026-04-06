import polars as pl

# 1. Load data
df = pl.read_parquet("training_new_data.parquet")

# 2. Grab 10 random users and calculate their specific start, end, and duration
random_10 = (
    df.sample(n=10, seed=42) # seed=42 ensures you get the same 10 if you run it again
    .with_columns([
        pl.col("journey").list.first().struct.field("event_timestamp").alias("start_time"),
        pl.col("journey").list.last().struct.field("event_timestamp").alias("end_time"),
        pl.col("journey").list.len().alias("total_actions")
    ])
    .with_columns([
        # Calculate duration in days for readability
        ((pl.col("end_time") - pl.col("start_time")).dt.total_seconds() / 86400).alias("duration_days")
    ])
)

# 3. Display the results
print("--- Check of 10 Random User Durations ---")
print(random_10.select(["id", "total_actions", "start_time", "end_time", "duration_days"]))

# 4. Summary of these 10
print(f"\nAverage duration of these 10 users: {random_10['duration_days'].mean():.2f} days")