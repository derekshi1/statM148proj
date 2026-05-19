
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================================
# PHASE 7: KAGGLE SUBMISSION (MULTI-MODEL SUPPORT)
# ==============================================================================
print("\n--- PREPARING KAGGLE SUBMISSION ---")

# --- CONTROL PANEL ---
# Change this variable to "xgboost" or "wide_deep" to easily switch models!
# --- CONFIGURATION ---
MODEL_TO_USE = "ensemble" 

# 2. Define exactly which models should be in the ensemble
ENSEMBLE_STACK = ["neural_network", "logistic", "lstm"]# --------------------

# 1. Load the Test Data
test_file_path = "truncated_data_codex1.parquet"
print(f"Loading test data from {test_file_path}...")
test_df = pd.read_parquet(test_file_path)

test_ids = test_df['id'].values

# 2. Engineer the exact same features we did for training
print("Engineering test features...")
test_df['velocity_events_per_day'] = test_df['n_events_observed'] / (test_df['observed_duration_days'] + 1e-5)
test_df['velocity_unique_events_per_day'] = test_df['n_unique_event_types_observed'] / (test_df['observed_duration_days'] + 1e-5)
test_df['ratio_unique_to_total_events'] = test_df['n_unique_event_types_observed'] / (test_df['n_events_observed'] + 1e-5)
test_df['ratio_inactive_to_active_days'] = test_df['days_since_last_event'] / (test_df['observed_duration_days'] + 1e-5)

# 3. Prepare Sequence & Time Data 
# (Only strictly needed for Wide & Deep, but we extract them anyway just in case)
print("Padding test sequences and time gaps...")
MAX_SEQ_LEN = 50 # Make sure this matches your training MAX_SEQ_LEN!

raw_test_sequences = test_df['observed_sequence'].tolist()
X_test_seq = pad_sequences(raw_test_sequences, maxlen=MAX_SEQ_LEN, padding='pre', truncating='pre')

raw_test_time = test_df['observed_time_deltas'].tolist()
X_test_time = pad_sequences(raw_test_time, maxlen=MAX_SEQ_LEN, padding='pre', truncating='pre', dtype='float32')

# 4. Prepare the Dense Features (USING THE SAVED LIST)
print("Formatting and scaling test features...")

# Convert the saved Index from Step 5 to a normal Python list
features_to_keep = selected_feature_names.tolist()

# Slice the test dataframe using ONLY the columns that survived feature selection
X_test_dense_sel = test_df[features_to_keep]

# Impute and Scale (Using transform!)
X_test_dense_imp = imputer.transform(X_test_dense_sel)
X_test_dense_scaled = scaler.transform(X_test_dense_imp)

# 5. Generate Predictions based on the chosen model
print(f"Generating probability predictions using {MODEL_TO_USE}...")
import numpy as np



# 3. Input your validation Brier scores here
# (Make sure these variables are defined earlier in your script, 
#  or just type the raw numbers like 0.051)
BRIER_SCORES = {
    "xgboost": xgb_brier,
    "logistic": log_brier,
    "neural_network": nn_brier,
    "lstm": lstm_brier,
    "wide_deep": wd_brier
}

# --- EXECUTION ---
if MODEL_TO_USE == "xgboost":
    y_test_probs = xgb_calibrated.predict_proba(X_test_dense_scaled)[:, 1]
    output_filename = "submission_xgb.csv"

elif MODEL_TO_USE == "wide_deep":
    y_test_probs = wd_model.predict([X_test_dense_scaled, X_test_seq, X_test_time]).ravel()
    output_filename = "submission_wd.csv"

elif MODEL_TO_USE == "logistic":
    y_test_probs = log_model.predict_proba(X_test_dense_scaled)[:, 1]
    output_filename = "submission_log.csv"

elif MODEL_TO_USE == "neural_network":
    y_test_probs = nn_model.predict(X_test_dense_scaled, verbose=1).ravel()
    output_filename = "submission_nn.csv"
elif MODEL_TO_USE == "lstm":
    y_test_probs = best_lstm_model.predict([X_test_seq, X_test_time], verbose=1).ravel()
    output_filename = "submission_lstm.csv"
elif MODEL_TO_USE == "ensemble":
    print(f"Generating Ensemble using stack: {ENSEMBLE_STACK}")
    
    # --- DYNAMIC INVERSE BRIER WEIGHTING ---
    inv_scores = {}
    total_inv = 0.0
    
    for model_name in ENSEMBLE_STACK:
        # Get the score, calculate inverse, and add to running total
        score = BRIER_SCORES[model_name]
        inv = 1.0 / score
        inv_scores[model_name] = inv
        total_inv += inv
        
    # Calculate the final proportional weights so they sum to exactly 1.0
    DYNAMIC_WEIGHTS = {name: (inv / total_inv) for name, inv in inv_scores.items()}
    
    print("\n--- Calculated Ensemble Weights ---")
    for name, weight in DYNAMIC_WEIGHTS.items():
        print(f"{name}: {weight:.4f} (Score: {BRIER_SCORES[name]:.4f})")
    print("-----------------------------------\n")

    # --- GENERATE PREDICTIONS ---
    stack_probs = []
    
    for model_name in ENSEMBLE_STACK:
        if model_name == "xgboost":
            p = xgb_calibrated.predict_proba(X_test_dense_scaled)[:, 1]
        elif model_name == "wide_deep":
            p = wd_model.predict([X_test_dense_scaled, X_test_seq, X_test_time]).ravel()
        elif model_name == "logistic":
            p = log_model.predict_proba(X_test_dense_scaled)[:, 1]
        elif model_name == "neural_network":
            p = nn_model.predict(X_test_dense_scaled, verbose=0).ravel()
        elif model_name == "lstm":
            p = best_lstm_model.predict([X_test_seq, X_test_time], verbose=1).ravel()
        else:
            continue
            
        weight = DYNAMIC_WEIGHTS[model_name]
        stack_probs.append(p * weight)
    
    # Compute the final weighted sum
    # (No need to divide by sum of weights here because DYNAMIC_WEIGHTS already perfectly sum to 1.0)
    y_test_probs = np.sum(stack_probs, axis=0)
    output_filename = f"submission_ensemble_{len(ENSEMBLE_STACK)}models.csv"

else:
    raise ValueError("Invalid MODEL_TO_USE.")


# final_test_probs = y_test_probs
# # --- CALCULATE BLENDED METRICS ---
# final_test_preds = (final_test_probs >= 0.5).astype(int)

# final_test_brier = brier_score_loss(y_val, final_test_probs)
# final_test_auc = roc_auc_score(y_val, final_test_probs)
# final_test_prauc = average_precision_score(y_val, final_test_probs)
# final_test_ll = log_loss(y_val, final_test_probs)
# final_test_acc = accuracy_score(y_val, final_test_preds)

# print("\n--- Blended Final Test Performance Stats ---")
# print(f"Brier Score: {final_test_brier:.5f}")
# print(f"ROC-AUC:     {final_test_auc:.5f}")
# print(f"PR-AUC:      {final_test_prauc:.5f}")
# print(f"Log Loss:    {final_test_ll:.5f}")
# print(f"Accuracy:    {final_test_acc:.5f}")
# print("Formatting Kaggle submission file...")

submission_df = pd.DataFrame({
    'id': test_ids,
    'order_shipped': y_test_probs
})

submission_df.to_csv(output_filename, index=False)
print(f"Done! Saved '{output_filename}' successfully.")


#%%from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas as pd

print("Generating Individual ICE Plots...")

# 1. Define the exact features you want to inspect
features_to_plot = [
    'sampling_weight_days',           
    'velocity_unique_events_per_day', 
    'count_ed_4', 'observed_duration_days','estimated_sessions', 'last_ed_id'                    
]

# 2. Extract the model that knows the column names
fitted_xgb = selector.estimator_

# 3. Create a manageable sample and cast to float64 to prevent Pandas interpolation errors
X_sample_df = pd.DataFrame(
    X_val_dense_raw, 
    columns=X_train_dense_raw.columns
).sample(1000, random_state=42).astype('float64')

# 4. Loop through each feature and create a separate plot
for feature in features_to_plot:
    fig, ax = plt.subplots(figsize=(10, 6)) # Size for a single plot
    
    display = PartialDependenceDisplay.from_estimator(
        estimator=fitted_xgb,
        X=X_sample_df,
        features=[feature], # Pass as a list with a single item
        kind='both',      
        subsample=100,    
        n_jobs=-1,        
        grid_resolution=30, 
        ax=ax,
        ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
        pd_line_kw={"color": "tab:orange", "linewidth": 3, "alpha": 1}
    )
    
    ax.set_title(f'ICE & PDP Plot: {feature}', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()
#%%
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

print("Extracting target users for Ceteris Paribus plot...")

# 1. Define the specific IDs
target_ids = ['775243448', '1145382956', '999750675', '-1735909364']

# 2. Find these users in the original dataframe
target_users_df = df[df['id'].astype(str).isin(target_ids)]

if len(target_users_df) == 0:
    print("WARNING: None of those IDs were found in the training dataframe. Are they in the test set?")
else:
    print(f"Found {len(target_users_df)} matching users. Generating plot...")
    
    # 3. Clean the features exactly like we did in the pipeline
    X_target_raw = target_users_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Drop strings safely
    object_cols_target = X_target_raw.select_dtypes(include=['object']).columns
    if len(object_cols_target) > 0:
        X_target_raw = X_target_raw.drop(columns=object_cols_target)
        
    # Ensure exact column match with the training data and cast to float64
    X_target_raw = X_target_raw[X_train_dense_raw.columns].astype('float64')
    
    # 4. Generate the CP Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    display = PartialDependenceDisplay.from_estimator(
        estimator=fitted_xgb,
        X=X_target_raw,
        features=['observed_duration_days'],
        kind='individual', # 'individual' plots ONLY the specific CP lines, no average line
        n_jobs=-1,
        grid_resolution=50, 
        ax=ax,
        # Making the lines thicker and darker since we only have a few to look at
        ice_lines_kw={"linewidth": 3, "alpha": 0.8} 
    )
    
    # Add a custom legend based on the IDs we actually found
    ax.legend(target_users_df['id'].astype(str).tolist(), title="User IDs")
    ax.set_title("Ceteris Paribus Plot: observed_duration_days", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()