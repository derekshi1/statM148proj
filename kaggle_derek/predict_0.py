import pandas as pd

file_path = '/Users/derek/UCLA_stats/statM148proj/kaggle/testing_submission.parquet'

# 1. Load data
# sep='\s+' tells pandas the separator is 'one or more spaces'
# encoding='latin1' handles that decode error you saw
df = pd.read_parquet(file_path)
# 2. Create the ID Kaggle wants
# Based on your example, the ID is "customer_id account_id"
# We convert to string to ensure they stay formatted correctly

# 3. Create the prediction column
df['order_shipped'] = 0.0

# 4. Filter to just the required columns
submission = df[['id', 'order_shipped']]

# 5. Save (Kaggle requires a comma-separated file for submission)
submission.to_csv('submission.csv', index=False)

print("Created submission.csv with shape:", submission.shape)
print("\nFirst few rows of submission:")
print(submission.head())