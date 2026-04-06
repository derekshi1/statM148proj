import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SETUP AND LOADING ---
input_parquet = "training_new_data.parquet"
event_defs_path = "Event Definitions.csv"

print("Loading data...")
df = pl.read_parquet(input_parquet)
event_defs = pl.read_csv(event_defs_path)

# --- 2. SAMPLING ---
# We take a 10% random sample (approx 143k users) 
# This is statistically significant but 10x faster for plotting.
print("Creating 10% sample for analysis...")
df_sample = df.sample(fraction=0.1, seed=42)

# --- 3. TASK 3.1: JOURNEY LENGTHS (ACTIONS & TIME) ---
print("Calculating journey lengths...")
analysis_df = df_sample.with_columns([
    # Count how many actions are in the journey list
    pl.col("journey").list.len().alias("n_actions"),
    
    # Calculate duration in minutes (Last - First)
    ((pl.col("journey").list.last().struct.field("event_timestamp") - 
      pl.col("journey").list.first().struct.field("event_timestamp"))
      .dt.total_seconds() / (60 * 60 * 24)).alias("duration_days") # Div by 86400
])

# Summary Stats
print("\n--- Typical Journey Statistics (Median) ---")
print(f"Median Actions: {analysis_df['n_actions'].median()}")
print(f"Median Duration: {analysis_df['duration_days'].median():.2f} days")

# Visualization: Journey Length (Actions)
plt.figure(figsize=(10, 5))
sns.histplot(analysis_df['n_actions'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Actions per Journey (10% Sample)")
plt.xlabel("Number of Actions")
plt.xlim(0, analysis_df['n_actions'].quantile(0.95)) # Zoom in on the bulk of users
plt.show()

# --- 4. TASK 3.2: COMMON ACTIONS (FIXED) ---
print("\nIdentifying common actions...")

# 1. Explode and unnest to get individual actions
exploded_df = (
    df_sample.select(pl.col("journey").list.explode())
    .unnest("journey")
)

# 2. Group by ed_id and count (using .len() instead of .count())
common_actions = (
    exploded_df.group_by("ed_id")
    .len() # This replaces .count() to fix the DeprecationWarning
    .rename({"len": "count"})
)

# 3. Join with Event Definitions
# We use 'left_on' and 'right_on' to handle different column names
event_defs = pl.read_csv("Event Definitions.csv")

# Identify the correct column name in your CSV (likely 'event_definition_id')
final_actions = common_actions.join(
    event_defs, 
    left_on="ed_id", 
    right_on="event_definition_id" # This matches the error message's suggestion
).sort("count", descending=True)

print("--- Top 10 Most Common Actions ---")
print(final_actions.head(10))

# 4. Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
top_10 = final_actions.head(10).to_pandas()
sns.barplot(data=top_10, x='count', y='event_name', palette='viridis')
plt.title("Top 10 Most Frequent Actions")
plt.tight_layout()
plt.show()

# --- 5. TASK 3.3: TIME BETWEEN ACTIONS ---
print("\nCalculating time between individual actions...")
# We use .diff() inside the list to find the gap between timestamps
time_gap_df = df_sample.with_columns(
    pl.col("journey")
    .list.eval(
        pl.element().struct.field("event_timestamp").diff().dt.total_seconds()
    )
    .alias("gaps_sec")
)

# Explode the gaps to get a single list of every time difference in the data
all_gaps = time_gap_df.select(pl.col("gaps_sec").list.explode()).drop_nulls()

median_gap = all_gaps["gaps_sec"].median()
print(f"Median time between actions: {median_gap:.2f} seconds")

# Optional: Visualization of gaps (Log scale because some gaps are huge)
plt.figure(figsize=(10, 5))
sns.histplot(all_gaps['gaps_sec'], bins=100, color='salmon')
plt.xscale('log')
plt.title("Time Between Actions (Log Scale)")
plt.xlabel("Seconds")
plt.show()

print("\nAnalysis Complete!")