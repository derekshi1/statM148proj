import polars as pl
import pandas as pd
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# 1. Load the featurized data (the one with momentum and counts)
# Use the file you just created with the build script
df = pl.read_parquet("/Users/derek/UCLA_stats/statM148proj/data/truncated_data_momentum.parquet")

# 2. Get the feature names the model was trained on
model = xgb.XGBClassifier()
model.load_model("/Users/derek/UCLA_stats/statM148proj/derek_models/xgb_momentum_model.json")
features = model.get_booster().feature_names

# 3. Create a sample for the plot (X_plot)
# We sample 2,000 rows so it runs fast on your Air
X_plot = df.select(features).sample(2000, seed=42).to_pandas()

# IMPORTANT: PDP cannot handle NaNs. If users never saw a milestone, 
# fill with 0 so the plotter doesn't crash.
X_plot['max_milestone_seen'] = X_plot['max_milestone_seen'].fillna(0)

# 4. Generate the Plot
print("Plotting PDP and ICE for max_milestone_seen...")
fig, ax = plt.subplots(figsize=(10, 8))

display = PartialDependenceDisplay.from_estimator(
    model,
    X_plot,
    features=['max_milestone_seen'],
    kind='both',      # 'both' gives you PDP (average) and ICE (individual lines)
    subsample=50,      # Number of ICE lines to draw
    n_jobs=-1,         # Use all CPU cores
    ax=ax
)

plt.title("PDP and ICE: Effect of Max Milestone on Success Probability")
plt.show()