import pandas as pd

# Update the path to point to your .parquet file
file_path = '/Users/derek/UCLA_stats/statM148proj/kaggle/testing_submission.parquet'

# 1. Load the Parquet file
# Parquet handles encoding and separators automatically, so no 'sep' or 'encoding' needed!
df = pd.read_parquet(file_path)

# 2. Check the shape
print(f"The shape of the dataset is: {df.shape}")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")

# 3. Quick peek at the ID column to ensure it loaded correctly
print("\nFirst 5 IDs:")
print(df['id'].head())