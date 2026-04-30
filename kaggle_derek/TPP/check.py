import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
# Load raw journey data
DATA_PATH = "/Users/derek/UCLA_stats/statM148proj/data/training_new_data.parquet"
df = pl.read_parquet(DATA_PATH)

SUCCESS_ID = 28

# 1. Create label and extract TPP features
processed_df = df.with_columns([
    # Create the label: Does the journey list contain ID 28?
    pl.col("journey").list.eval(
        pl.element().struct.field("ed_id") == SUCCESS_ID
    ).list.any().alias("label_order_shipped")
]).select([
    pl.col("id"),
    pl.col("label_order_shipped"),
    # Sequence of event IDs
    pl.col("journey").list.eval(pl.element().struct.field("ed_id")).alias("event_seq"),
    # Sequence of time deltas (seconds between events)
    pl.col("journey").list.eval(
        (pl.element().struct.field("event_timestamp").diff().dt.total_seconds().fill_null(0))
    ).alias("delta_t_seq"),
    # Absolute time for intensity calculation
    pl.col("journey").list.eval(
        (pl.element().struct.field("event_timestamp") - pl.element().struct.field("event_timestamp").first()).dt.total_seconds()
    ).alias("abs_time_seq")
])
# Using your processed_df from the TPP preprocessing
# Check the last element of every 'event_seq'
lengths = processed_df.select([
    pl.col("event_seq").list.len().alias("journey_length"),
    pl.col("label_order_shipped")
])

# 2. Get Percentiles
percentiles = [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
stats = lengths.select([
    pl.col("journey_length").quantile(p).alias(f"{int(p*100)}th") for p in percentiles
])

print("Journey Length Percentiles:")
print(stats)

# 3. Visualize
plt.figure(figsize=(10, 6))
sns.histplot(lengths.to_pandas(), x="journey_length", hue="label_order_shipped", kde=True, bins=50)
plt.title("Distribution of Journey Lengths")
plt.xlabel("Number of Events")
plt.ylabel("Count")
plt.show()