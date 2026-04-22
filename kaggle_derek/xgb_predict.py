import polars as pl
import xgboost as xgb
import pandas as pd
import numpy as np

MODEL_PATH = "/Users/derek/UCLA_stats/statM148proj/derek_models/xgb_momentum_model.json"
TEST_DATA_PATH = "/Users/derek/UCLA_stats/statM148proj/data/derek_xgb_test_set.parquet"
OUTPUT_CSV = "/Users/derek/UCLA_stats/statM148proj/kaggle_derek/derek_submission_momentum.csv"


model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
df_test = pl.read_parquet(TEST_DATA_PATH)

# 3. Get Features and Predict
# Use model.get_booster().feature_names to ensure alignment with training
features = model.get_booster().feature_names
X_test = df_test.select(features).to_pandas()

# Get the raw probabilities for the 'Success' class
probs = model.predict_proba(X_test)[:, 1]

final_preds = np.zeros_like(probs)

# Fill the "Unsure" bucket
final_preds[(probs >= 0.6) & (probs < 0.7)] = 0.15
final_preds[(probs >= 0.7) & (probs < 0.8)] = 0.5


# Fill the "High Confidence" bucket
final_preds[probs >= 0.8] = 0.8

# 4. Create and Save Submission
submission = pd.DataFrame({
    'id': df_test['id'].to_list(),
    'order_shipped': final_preds
})

submission.to_csv('submission_tiered.csv', index=False)

# 5. Output Stats
print(f"Total Tier 1 (0.0): {(final_preds == 0.0).sum():,}")
print(f"Total Tier 2 (0.5): {(final_preds == 0.15).sum():,}")
print(f"Total Tier 2 (0.5): {(final_preds == 0.5).sum():,}")

print(f"Total Tier 3 (1.0): {(final_preds == 0.8).sum():,}")