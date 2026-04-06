import polars as pl

# 1. Load the Parquet file (Use scan for efficiency)
output_path = "training_new_data.parquet" # Make sure this matches your script
df = pl.read_parquet(output_path)

# 2. Check the "Shape" 
# (This will show how many unique IDs you have now)
print(f"Dataset Shape: {df.shape}")

# 3. Look at the first 5 rows
print("--- First 5 User Journeys ---")
print(df.head())

# 4. Peek inside the 'journey' struct of the first user
first_user_journey = df[0, "journey"]
print("\n--- First User's First 3 Actions ---")
print(first_user_journey[:3])


first_user_journey = df[1, "journey"]
print("\n--- Second User's First 3 Actions ---")
print(first_user_journey[:3])


first_user_journey = df[2, "journey"]
print("\n--- Third User's First 3 Actions ---")
print(first_user_journey[:3])