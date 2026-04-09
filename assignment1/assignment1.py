import polars as pl

# Load your processed file
df = pl.read_parquet("/Users/derek/UCLA_stats/statM148proj/training_new_data.parquet")

# 1. How many rows (Users)?
n_rows = df.height

# 2. How many unique IDs?
# Since we grouped by ID, every row is already a unique ID
n_unique_ids = df["id"].n_unique()

# 3. Earliest and Latest Timestamps
# We reach into the journey list to find the start of the first journey 
# and the end of the last journey across the whole dataset.
time_stats = df.select([
    pl.col("journey").list.first().struct.field("event_timestamp").min().alias("earliest"),
    pl.col("journey").list.last().struct.field("event_timestamp").max().alias("latest")
])

print(f"1. Total Rows: {n_rows}")
print(f"2. Unique IDs: {n_unique_ids}")
print(f"3. Time Range: {time_stats.row(0)[0]} to {time_stats.row(0)[1]}")



# 1. Start a Lazy Scan (Doesn't load into RAM yet)
csv_path = "/Users/derek/UCLA_stats/statM148proj/training_data.csv"
q = pl.scan_csv(csv_path)

# 2. Get the Total Row Count (Raw)
total_count = q.select(pl.len()).collect().item()

# 3. Define the Duplicates 
# We define a duplicate as same ID, same Action (ed_id), and same Timestamp
# We keep the first occurrence
q_unique = q.unique(subset=["id", "ed_id", "event_timestamp"], keep="first")

# 4. Execute the deduplication and get the New Count
df_cleaned = q_unique.collect()
unique_count = df_cleaned.height

# 5. Calculate Statistics
duplicate_count = total_count - unique_count
proportion_duplicates = (duplicate_count / total_count) * 100

# 6. Summary Statistics Evidence
print("--- Deduplication Summary Statistics ---")
print(f"1. Total entries (Raw): {total_count:,}")
print(f"2. Total duplicates found: {duplicate_count:,}")
print(f"3. Proportion of duplicates: {proportion_duplicates:.2f}%")
print(f"4. Rows remaining after cleaning: {unique_count:,}")

# 7. Optional: Save the cleaned data to Parquet for your next steps
# (This is much better than saving back to a massive CSV)
df_cleaned.write_parquet("new_cleaned_training_data.parquet")



