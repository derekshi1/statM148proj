import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, log_loss, accuracy_score
import optuna

# ==========================================
# 2. XGBOOST MODEL (Optuna Tuning & Seed Blending)
# ==========================================
print("--- Tuning XGBoost ---")

def objective_xgb(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10), 
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), 
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'eval_metric': 'logloss',
        'random_state': 42 # Keep seed static during tuning for fair comparison
    }
    
    xgb_base = XGBClassifier(**param)
    # Using cv=3 inside the trial to save time; using sigmoid here is fine for speed
    xgb_calibrated = CalibratedClassifierCV(estimator=xgb_base, method='sigmoid', cv=3)
    xgb_calibrated.fit(X_train, y_train)
    
    preds = xgb_calibrated.predict_proba(X_val)[:, 1]
    return brier_score_loss(y_val, preds)

# Run Optimization
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=20) 

print(f"\nBest Optuna Params: {study_xgb.best_params}")

# ==========================================
# 3. SEED BLENDING & FINAL PREDICTIONS
# ==========================================
print("\n--- Training Best XGBoost with Seed Blending ---")

SEEDS = [42, 43, 44, 45, 46]
val_probs_list = []
test_probs_list = []

for seed in SEEDS:
    print(f"Fitting model with random seed {seed}...")
    
    # Inject the new seed into the best parameters
    best_params = study_xgb.best_params.copy()
    best_params['random_state'] = seed
    
    xgb_base = XGBClassifier(**best_params, eval_metric='logloss')
    
    # Using Isotonic for the final build as it handles the 1.1M rows better
    xgb_calibrated = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=5)
    xgb_calibrated.fit(X_train, y_train)
    
    # 1. Gather Validation Probs (to calculate your blended metrics)
    val_probs_list.append(xgb_calibrated.predict_proba(X_val)[:, 1])
    
    # 2. Gather Test Probs (for the actual submission)
    # NOTE: Ensure X_test_dense_scaled matches your test feature matrix!
    test_probs_list.append(xgb_calibrated.predict_proba(X_test_dense_scaled)[:, 1])

# --- BLEND (AVERAGE) THE PROBABILITIES ---
xgb_probs = np.mean(val_probs_list, axis=0)
final_test_probs = np.mean(test_probs_list, axis=0)

# --- CALCULATE BLENDED METRICS ---
xgb_preds = (xgb_probs >= 0.5).astype(int)

xgb_brier = brier_score_loss(y_val, xgb_probs)
xgb_auc = roc_auc_score(y_val, xgb_probs)
xgb_prauc = average_precision_score(y_val, xgb_probs)
xgb_ll = log_loss(y_val, xgb_probs)
xgb_acc = accuracy_score(y_val, xgb_preds)

print("\n--- Blended XGBoost Performance Stats ---")
print(f"Brier Score: {xgb_brier:.5f}")
print(f"ROC-AUC:     {xgb_auc:.5f}")
print(f"PR-AUC:      {xgb_prauc:.5f}")
print(f"Log Loss:    {xgb_ll:.5f}")
print(f"Accuracy:    {xgb_acc:.5f}")
#%%
# ==========================================
# TEST DATA PROCESSING (For Submission)
# ==========================================
print("Loading test snapshot data...")
# Make sure to change this to your actual test data filename
df_test = pd.read_parquet("test_data_codex.parquet") 

print("Generating manual features...")
df_test['velocity_events_per_day'] = df_test['n_events_observed'] / (df_test['observed_duration_days'] + 1e-5)
df_test['velocity_unique_events_per_day'] = df_test['n_unique_event_types_observed'] / (df_test['observed_duration_days'] + 1e-5)
df_test['ratio_unique_to_total_events'] = df_test['n_unique_event_types_observed'] / (df_test['n_events_observed'] + 1e-5)
df_test['ratio_inactive_to_active_days'] = df_test['days_since_last_event'] / (df_test['observed_duration_days'] + 1e-5)

# Keep track of IDs for the submission file
test_ids = df_test['id'].values

# Drop the exact same columns we dropped in training
X_test_dense_raw = df_test.drop(columns=cols_to_drop, errors='ignore')

# Drop any remaining object (string) columns safely
object_cols_test = X_test_dense_raw.select_dtypes(include=['object']).columns
if len(object_cols_test) > 0:
    X_test_dense_raw = X_test_dense_raw.drop(columns=object_cols_test)

# Make sure columns perfectly match the training columns before transformation
# (This adds missing columns full of 0s if the test set is missing an event type)
missing_cols = set(X_train_dense_raw.columns) - set(X_test_dense_raw.columns)
for c in missing_cols:
    X_test_dense_raw[c] = 0
X_test_dense_raw = X_test_dense_raw[X_train_dense_raw.columns]

print("Applying feature selection, imputation, and scaling...")
# CRITICAL: Only use .transform() here! 
# This forces the test data to have the exact same 13 features as the training data.
X_test_dense_sel = selector.transform(X_test_dense_raw)
X_test_dense_imp = imputer.transform(X_test_dense_sel)
X_test_dense_scaled = scaler.transform(X_test_dense_imp)

print(f"Test data ready! Shape: {X_test_dense_scaled.shape}")
# This should print (_, 13). If it does, your seed blending script will run perfectly!