import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 1. Load the featurized data
DATA_PATH = "/Users/derek/UCLA_stats/statM148proj/data/truncated_data_momentum.parquet"
df = pl.read_parquet(DATA_PATH)

# 2. Separate Target, Weights, and Features
# We drop IDs, timestamps, and the terminal reason which would be "leaking" the answer
target = "label_order_shipped"
weight_col = "sampling_weight_days"
drop_cols = [
    "id", "cutoff_time", "journey_terminal_time", 
    "journey_terminal_reason", target, weight_col,
    "first_event_name", "first_stage", "last_event_name", "last_stage" # Drop text, keep IDs
]

features = [c for c in df.columns if c not in drop_cols]

X = df.select(features).to_pandas()
y = df[target].to_pandas()
weights = df[weight_col].to_pandas()

# 3. Train/Test Split
# We split the weights along with X and y
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# 4. Initialize and Train XGBoost
# We use scale_pos_weight if your success rate is very low (e.g., < 5%)
success_rate = y.mean()
scale_pos_weight = (1 - success_rate) / success_rate

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",  # Fast training
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

# Crucial: Apply the sampling_weight_days here
model.fit(
    X_train, y_train, 
    sample_weight=w_train,
    eval_set=[(X_val, y_val)],
    sample_weight_eval_set=[w_val],
    verbose=50
)

# 5. Evaluate
preds = model.predict(X_val)
probs = model.predict_proba(X_val)[:, 1]


probs = model.predict_proba(X_val)[:, 1]

# 2. Define your high-confidence threshold
CUSTOM_THRESHOLD = 0.60

# 3. Apply the threshold to create new predictions
# Returns 1 only if probability >= 0.7, else 0
custom_preds = (probs >= CUSTOM_THRESHOLD).astype(int)

print(f"\n--- Validation with {CUSTOM_THRESHOLD*100}% Threshold ---")
print(f"ROC AUC Score: {roc_auc_score(y_val, probs, sample_weight=w_val):.4f} (Unchanged by threshold)")

# Use the weights in the report to match your training logic
print(classification_report(y_val, custom_preds, sample_weight=w_val))

# 4. Compare distribution
print("\n--- Prediction Distribution Comparison ---")
print(f"Successes predicted at 0.5 threshold: {(probs >= 0.5).sum():,}")
print(f"Successes predicted at {CUSTOM_THRESHOLD} threshold: {custom_preds.sum():,}")
# 6. Feature Importance
plt.figure(figsize=(10, 12))
xgb.plot_importance(model, max_num_features=20, importance_type="gain")
plt.title("Top 20 Features (by Gain)")
plt.show()

model_path = "/Users/derek/UCLA_stats/statM148proj/derek_models/xgb_momentum_model.json"
model.save_model(model_path)
print(f"Model saved to {model_path}")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []


# 2. Manual CV Loop to handle weights
# We fold based on X and y, but keep weights aligned
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # Split data for this fold
    X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
    y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
    w_t, w_v = weights.iloc[train_idx], weights.iloc[val_idx]
    
    # Initialize a fresh model for this fold
    fold_model = xgb.XGBClassifier(
        n_estimators=100, # Lower for faster CV
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    # Fit with weights
    fold_model.fit(X_t, y_t, sample_weight=w_t)
    
    # Predict using your preferred threshold (e.g., 0.6)
    v_probs = fold_model.predict_proba(X_v)[:, 1]
    v_preds = (v_probs >= CUSTOM_THRESHOLD).astype(int)
    
    # Calculate WEIGHTED accuracy
    # This reflects the probability of correctly predicting a day of journey activity
    acc = accuracy_score(y_v, v_preds, sample_weight=w_v)
    cv_accuracies.append(acc)
    
    print(f"Fold {fold+1}: Weighted Accuracy = {acc:.4f}")

print(f"\n--- CV Results ---")
print(f"Mean Accuracy: {np.mean(cv_accuracies):.4f}")
print(f"Std Deviation: {np.std(cv_accuracies):.4f}")