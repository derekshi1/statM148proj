    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:43:07 2026

@author: jaydentani
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# ==========================================
# 1. LOAD DATA & ENGINEER FEATURES
# ==========================================
print("Loading baseline snapshot data...")
df = pd.read_parquet("truncated_data_codex.parquet")

print("Generating new features...")
df['velocity_events_per_day'] = df['n_events_observed'] / (df['observed_duration_days'] + 1e-5)
df['velocity_unique_events_per_day'] = df['n_unique_event_types_observed'] / (df['observed_duration_days'] + 1e-5)
df['ratio_unique_to_total_events'] = df['n_unique_event_types_observed'] / (df['n_events_observed'] + 1e-5)
df['ratio_inactive_to_active_days'] = df['days_since_last_event'] / (df['observed_duration_days'] + 1e-5)

# ==========================================
# 2. PREPARE DENSE FEATURES AND TARGET
# ==========================================
y = df['label_order_shipped'].values
weights = df['sampling_weight_days'].values

cols_to_drop = [
    'id', 'cutoff_time', 'journey_terminal_time', 'journey_terminal_reason', 
    'label_order_shipped', 'first_event_name', 'last_event_name', 
    'first_stage', 'last_stage', 'observed_sequence', 'journey',
    'observed_time_deltas' # Prevent lists from going to the Dense branch
]

X_dense_full = df.drop(columns=cols_to_drop, errors='ignore')

# Drop any remaining object (string) columns safely
object_cols = X_dense_full.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    X_dense_full = X_dense_full.drop(columns=object_cols)

# ==========================================
# 3A. PREPARE CATEGORICAL SEQUENCE DATA (For Embedding)
# ==========================================
print("Formatting categorical sequences...")
MAX_SEQ_LEN = 50 

raw_sequences = df['observed_sequence'].tolist()
VOCAB_SIZE = max([max(seq) if len(seq) > 0 else 0 for seq in raw_sequences]) + 1

X_seq_padded = pad_sequences(
    raw_sequences, 
    maxlen=MAX_SEQ_LEN, 
    padding='pre', 
    truncating='pre'
)

# ==========================================
# 3B. PREPARE CONTINUOUS STATE DATA (For LSTM)
# ==========================================
print("Formatting state sequences...")
safe_raw_states = []

for seq in df['observed_states']:
    # Catch any weird NaN or completely empty rows that Parquet might have generated
    if seq is None or len(seq) == 0:
        safe_raw_states.append(np.zeros((0, 3), dtype='float32'))
        continue
        
    try:
        # Force the PyArrow/Pandas object into a native Python list of lists
        clean_seq = [list(step) for step in seq]
        safe_raw_states.append(np.array(clean_seq, dtype='float32'))
    except Exception as e:
        # Failsafe: If a row is deeply corrupted, append a zero-matrix
        safe_raw_states.append(np.zeros((1, 3), dtype='float32'))

# Pad the safely formatted arrays
X_state_padded = pad_sequences(
    safe_raw_states, 
    maxlen=MAX_SEQ_LEN, 
    padding='pre',      
    truncating='pre',
    dtype='float32',
    value=0.0  # Pad with 0.0s for the missing time steps
)
print(f"Successfully padded {len(X_state_padded)} state sequences.")

# ==========================================
# 4. TRAIN / VAL SPLIT
# ==========================================
# CRITICAL: Split Dense, Categorical Seq, and Continuous State simultaneously
X_train_dense_raw, X_val_dense_raw, X_train_seq, X_val_seq, X_train_state, X_val_state, y_train, y_val, w_train, w_val = train_test_split(
    X_dense_full, X_seq_padded, X_state_padded, y, weights, test_size=0.2, random_state=42, stratify=y
)

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
print(f"Training Class Imbalance: {neg_count} Fails vs {pos_count} Successes")

# ==========================================
# 5. SCALE THE 3D STATE ARRAY (CRITICAL FOR LSTM)
# ==========================================
print("Scaling continuous state sequences...")
scaler_state = StandardScaler()

# Reshape to 2D: (Total_Timesteps, 3 Features)
X_train_state_2d = X_train_state.reshape(-1, 3)
X_val_state_2d = X_val_state.reshape(-1, 3)

# Fit on train, transform both
X_train_state_2d = scaler_state.fit_transform(X_train_state_2d)
X_val_state_2d = scaler_state.transform(X_val_state_2d)

# Reshape back to 3D: (Users, 50, 3)
X_train_state = X_train_state_2d.reshape(X_train_state.shape)
X_val_state = X_val_state_2d.reshape(X_val_state.shape)

print(f"Vocab Size for Embedding: {VOCAB_SIZE}")
print(f"Categorical Seq Shape: {X_train_seq.shape}")
print(f"Continuous State Shape: {X_train_state.shape}")

# ==========================================
# 6. AUTOMATED FEATURE SELECTION
# ==========================================
print(f"Starting feature selection on {X_train_dense_raw.shape[1]} features...")

selector_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')
selector = SelectFromModel(selector_model, threshold='1.25*mean')

# FIT ON TRAINING ONLY
X_train_dense_sel = selector.fit_transform(X_train_dense_raw, y_train)
# TRANSFORM VALIDATION
X_val_dense_sel = selector.transform(X_val_dense_raw) 

selected_feature_names = X_dense_full.columns[selector.get_support()]
print(f"Reduced feature set from {X_train_dense_raw.shape[1]} to {X_train_dense_sel.shape[1]} features.")

# --- PLOT FEATURE IMPORTANCE ---
print("Generating Feature Importance Plot...")

# Extract importances from the fitted XGBoost model inside the selector
importances = selector.estimator_.feature_importances_

# Create a DataFrame for easy sorting
importance_df = pd.DataFrame({
    'Feature': X_train_dense_raw.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the Top 20 features
plt.figure(figsize=(10, 8))
sns.barplot(
    data=importance_df.head(20), 
    x='Importance', 
    y='Feature', 
    hue='Feature', 
    legend=False, 
    palette='viridis'
)
plt.title("Top 20 Feature Importances (From XGBoost Selector)")
plt.xlabel("Importance (Gain)")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()

# --- SHAP VALUE PLOT ---
print("Generating SHAP Summary Plot...")

# Get the actual fitted XGBoost model from inside the selector
fitted_xgb = selector.estimator_

# Create the explainer
explainer = shap.TreeExplainer(fitted_xgb)

# CRITICAL FIX: Pass the RAW validation data (all features) so dimensions match
X_val_df = pd.DataFrame(X_val_dense_raw, columns=X_train_dense_raw.columns).sample(10000, random_state=42)

shap_values = explainer.shap_values(X_val_df)

# Generate the plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_val_df)

# ==========================================
# 7. IMPUTE AND SCALE DENSE FEATURES
# ==========================================
imputer = SimpleImputer(strategy='constant', fill_value=0.0)
X_train_dense_imp = imputer.fit_transform(X_train_dense_sel) # Fit on Train
X_val_dense_imp = imputer.transform(X_val_dense_sel)         # Transform Val

scaler = StandardScaler()
X_train_dense_scaled = scaler.fit_transform(X_train_dense_imp) # Fit on Train
X_val_dense_scaled = scaler.transform(X_val_dense_imp)         # Transform Val

# Rename for your downstream models
X_train_dense = X_train_dense_scaled
X_val_dense = X_val_dense_scaled
X_train = X_train_dense
X_val = X_val_dense
NUM_DENSE_FEATURES = X_train.shape[1]

initial_bias = np.log([pos_count / neg_count])
output_bias = tf.keras.initializers.Constant(initial_bias)

print("\nData pipeline complete. Ready for modeling.")
#%%

import polars as pl
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Embedding, LSTM, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV # <--- NEW
from xgboost import XGBClassifier
from numpy import shape
from tensorflow.keras.preprocessing.sequence import pad_sequences
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%
# ==========================================
# 0. DATA PREPARATION & SPLITTING
# ==========================================
# Load the data
training_parquet_path = "data_train1.parquet"
df = pl.read_parquet(training_parquet_path)

file_path = "truncated_data_codex.parquet"
df_truncated = pl.read_parquet(file_path)

dense_feature_cols = [
    'n_events_observed', 'n_unique_event_types_observed', 'observed_duration_days',
    'days_since_last_event', 'cutoff_hour_utc', 'cutoff_dayofweek_utc',
    'cutoff_month_utc', 'cutoff_dayofyear_utc', 'first_ed_id', 'last_ed_id',
    'max_milestone_seen', 
    'count_ed_2', 'seen_ed_2', 'count_ed_3', 'seen_ed_3', 'count_ed_4', 'seen_ed_4',
    'count_ed_5', 'seen_ed_5', 'count_ed_6', 'seen_ed_6', 'count_ed_7', 'seen_ed_7',
    'count_ed_8', 'seen_ed_8', 'count_ed_9', 'seen_ed_9', 'count_ed_10', 'seen_ed_10',
    'count_ed_11', 'seen_ed_11', 'count_ed_12', 'seen_ed_12', 'count_ed_13', 'seen_ed_13',
    'count_ed_14', 'seen_ed_14', 'count_ed_15', 'seen_ed_15', 'count_ed_16', 'seen_ed_16',
    'count_ed_17', 'seen_ed_17', 'count_ed_18', 'seen_ed_18', 'count_ed_19', 'seen_ed_19',
    'count_ed_20', 'seen_ed_20', 'count_ed_21', 'seen_ed_21', 'count_ed_22', 'seen_ed_22',
    'count_ed_23', 'seen_ed_23', 'count_ed_25', 'seen_ed_25', 'count_ed_26', 'seen_ed_26',
    'count_ed_27', 'seen_ed_27', 'count_ed_28', 'seen_ed_28', 'count_ed_29', 'seen_ed_29',
    'count_ed_37', 'seen_ed_37'
]

# Extract Dense Features and Target
y = df_truncated['label_order_shipped'].to_numpy()

X_dense = df_truncated.select(dense_feature_cols).to_numpy() 
X_dense[np.isinf(X_dense)] = np.nan 
imputer = SimpleImputer(strategy='constant', fill_value=0.0)
X_dense_clean = imputer.fit_transform(X_dense)

scaler = StandardScaler()
X_dense_scaled = scaler.fit_transform(X_dense_clean) 


# --- SPLIT BOTH DENSE AND SEQUENCE DATA TOGETHER ---
X_train_dense, X_val_dense, X_train_seq, X_val_seq, y_train, y_val = train_test_split(
    X_dense_scaled, X_seq_padded, y, test_size=0.2, random_state=42, stratify=y
)

# FIX: Map the old variable names so Models 1, 2, and 3 don't crash
X_train = X_train_dense
X_val = X_val_dense

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
print(f"Training Class Imbalance: {neg_count} Fails vs {pos_count} Successes")
print(f"Vocabulary Size for LSTM: {VOCAB_SIZE}\n")
#%%

# ==========================================
# 1. LOAD DATA & ENGINEER FEATURES
# ==========================================
print("Loading baseline snapshot data...")
df = pd.read_parquet("truncated_data_codex.parquet")

print("Generating new features...")
df['velocity_events_per_day'] = df['n_events_observed'] / (df['observed_duration_days'] + 1e-5)
df['velocity_unique_events_per_day'] = df['n_unique_event_types_observed'] / (df['observed_duration_days'] + 1e-5)
df['ratio_unique_to_total_events'] = df['n_unique_event_types_observed'] / (df['n_events_observed'] + 1e-5)
df['ratio_inactive_to_active_days'] = df['days_since_last_event'] / (df['observed_duration_days'] + 1e-5)

# ==========================================
# 2. PREPARE DENSE FEATURES AND TARGET
# ==========================================
y = df['label_order_shipped'].values
weights = df['sampling_weight_days'].values
cols_to_drop = [
    'id', 'cutoff_time', 'journey_terminal_time', 'journey_terminal_reason', 
    'label_order_shipped', 'first_event_name', 'last_event_name', 
    'first_stage', 'last_stage', 'observed_sequence', 'journey'
]

X_dense_full = df.drop(columns=cols_to_drop, errors='ignore')

# Drop any remaining object (string) columns safely
object_cols = X_dense_full.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    X_dense_full = X_dense_full.drop(columns=object_cols)

# ==========================================
# 3. PREPARE SEQUENCE DATA
# ==========================================
MAX_SEQ_LEN = 50 
raw_states = df['observed_states'].tolist()

# We no longer calculate VOCAB_SIZE because we aren't using categorical IDs.
# Instead, we pad the 3D array of continuous floats.
X_state_padded = pad_sequences(
    raw_states, 
    maxlen=MAX_SEQ_LEN, 
    padding='pre',      # 'pre' keeps the most recent events at the end of the memory
    truncating='pre',
    dtype='float32'     # Crucial: Must be float32, not int, for continuous time/counts
)

# ==========================================
# 4. TRAIN / VAL SPLIT
# ==========================================
# Crucial: Split BEFORE doing feature selection or scaling!
X_train_dense_raw, X_val_dense_raw, X_train_state, X_val_state, y_train, y_val, w_train, w_val = train_test_split(
    X_dense_full, X_state_padded, y, weights, test_size=0.2, random_state=42, stratify=y
)

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
print(f"Training Class Imbalance: {neg_count} Fails vs {pos_count} Successes")

# ==========================================
# 5. AUTOMATED FEATURE SELECTION
# ==========================================
print(f"Starting feature selection on {X_train_dense_raw.shape[1]} features...")

selector_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')
selector = SelectFromModel(selector_model, threshold='1.25*mean')
# FIT ON TRAINING ONLY
X_train_dense_sel = selector.fit_transform(X_train_dense_raw, y_train)
# TRANSFORM VALIDATION
X_val_dense_sel = selector.transform(X_val_dense_raw) 

selected_feature_names = X_dense_full.columns[selector.get_support()]
print(f"Reduced feature set from {X_train_dense_raw.shape[1]} to {X_train_dense_sel.shape[1]} features.")


# --- NEW: PLOT FEATURE IMPORTANCE ---
print("Generating Feature Importance Plot...")

# Extract importances from the fitted XGBoost model inside the selector
importances = selector.estimator_.feature_importances_

# Create a DataFrame for easy sorting
importance_df = pd.DataFrame({
    'Feature': X_train_dense_raw.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the Top 20 features
plt.figure(figsize=(10, 8))
sns.barplot(
    data=importance_df.head(20), 
    x='Importance', 
    y='Feature', 
    hue='Feature', 
    legend=False, 
    palette='viridis'
)
plt.title("Top 20 Feature Importances (From XGBoost Selector)")
plt.xlabel("Importance (Gain)")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()

# --- NEW: SHAP VALUE PLOT ---
print("Generating SHAP Summary Plot...")

# Get the actual fitted XGBoost model from inside the selector
fitted_xgb = selector.estimator_

# Create the explainer
explainer = shap.TreeExplainer(fitted_xgb)

# CRITICAL FIX: Pass the RAW validation data (all 72 features) so the dimensions match!
# We also take a random sample of 10,000 rows so SHAP calculates in seconds instead of minutes.
X_val_df = pd.DataFrame(X_val_dense_raw, columns=X_train_dense_raw.columns).sample(10000, random_state=42)

shap_values = explainer.shap_values(X_val_df)

# Generate the plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_val_df)
# ------------------------------------
# ------------------------------------
# ==========================================
# 6. IMPUTE AND SCALE
# ==========================================
imputer = SimpleImputer(strategy='constant', fill_value=0.0)
X_train_dense_imp = imputer.fit_transform(X_train_dense_sel) # Fit on Train
X_val_dense_imp = imputer.transform(X_val_dense_sel)         # Transform Val

scaler = StandardScaler()
X_train_dense_scaled = scaler.fit_transform(X_train_dense_imp) # Fit on Train
X_val_dense_scaled = scaler.transform(X_val_dense_imp)         # Transform Val

# Rename for your downstream models
X_train_dense = X_train_dense_scaled
X_val_dense = X_val_dense_scaled
X_train = X_train_dense
X_val = X_val_dense
NUM_DENSE_FEATURES = X_train.shape[1]
initial_bias = np.log([pos_count / neg_count])
output_bias = tf.keras.initializers.Constant(initial_bias)
print("\nData pipeline complete. Ready for modeling.")
#%%
# ==========================================
# 2. PREPARE DENSE FEATURES AND TARGET
# ==========================================
y = df['label_order_shipped'].values

cols_to_drop = [
    'id', 'cutoff_time', 'journey_terminal_time', 'journey_terminal_reason', 
    'label_order_shipped', 'first_event_name', 'last_event_name', 
    'first_stage', 'last_stage', 'observed_sequence', 'journey',
    'observed_time_deltas', # <--- ADDED: Prevent lists from going to the Dense branch
]

X_dense_full = df.drop(columns=cols_to_drop, errors='ignore')

# Drop any remaining object (string) columns safely
object_cols = X_dense_full.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    X_dense_full = X_dense_full.drop(columns=object_cols)

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# 3A. PREPARE CATEGORICAL SEQUENCE DATA (For Embedding)
# ==========================================
print("Formatting categorical sequences...")
MAX_SEQ_LEN = 50 

raw_sequences = df['observed_sequence'].tolist()
VOCAB_SIZE = max([max(seq) if len(seq) > 0 else 0 for seq in raw_sequences]) + 1

X_seq_padded = pad_sequences(
    raw_sequences, 
    maxlen=MAX_SEQ_LEN, 
    padding='pre', 
    truncating='pre'
)

# ==========================================
# 3B. PREPARE CONTINUOUS STATE DATA (For LSTM)
# ==========================================
print("Formatting state sequences...")
safe_raw_states = []

for seq in df['observed_states']:
    # Catch any weird NaN or completely empty rows that Parquet might have generated
    if seq is None or len(seq) == 0:
        safe_raw_states.append(np.zeros((0, 3), dtype='float32'))
        continue
        
    try:
        # Force the PyArrow/Pandas object into a native Python list of lists
        clean_seq = [list(step) for step in seq]
        safe_raw_states.append(np.array(clean_seq, dtype='float32'))
    except Exception as e:
        # Failsafe: If a row is deeply corrupted, append a zero-matrix
        safe_raw_states.append(np.zeros((1, 3), dtype='float32'))

# Pad the safely formatted arrays
X_state_padded = pad_sequences(
    safe_raw_states, 
    maxlen=MAX_SEQ_LEN, 
    padding='pre',      
    truncating='pre',
    dtype='float32',
    value=0.0  # Pad with 0.0s for the missing time steps
)
print(f"Successfully padded {len(X_state_padded)} state sequences.")

# ==========================================
# 4. TRAIN / VAL SPLIT
# ==========================================
# CRITICAL: Split Dense, Categorical Seq, and Continuous State simultaneously
X_train_dense, X_val_dense, X_train_seq, X_val_seq, X_train_state, X_val_state, y_train, y_val, w_train, w_val = train_test_split(
    X_dense_full, X_seq_padded, X_state_padded, y, weights, test_size=0.2, random_state=42, stratify=y
)

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
print(f"Training Class Imbalance: {neg_count} Fails vs {pos_count} Successes")

# ==========================================
# 5. SCALE THE 3D STATE ARRAY (CRITICAL FOR LSTM)
# ==========================================
print("Scaling continuous state sequences...")
scaler_state = StandardScaler()

# Reshape to 2D: (Total_Timesteps, 3 Features)
X_train_state_2d = X_train_state.reshape(-1, 3)
X_val_state_2d = X_val_state.reshape(-1, 3)

# Fit on train, transform both
X_train_state_2d = scaler_state.fit_transform(X_train_state_2d)
X_val_state_2d = scaler_state.transform(X_val_state_2d)

# Reshape back to 3D: (Users, 50, 3)
X_train_state = X_train_state_2d.reshape(X_train_state.shape)
X_val_state = X_val_state_2d.reshape(X_val_state.shape)

print(f"Vocab Size for Embedding: {VOCAB_SIZE}")
print(f"Categorical Seq Shape: {X_train_seq.shape}")
print(f"Continuous State Shape: {X_train_state.shape}")
# ==========================================
# 5. AUTOMATED FEATURE SELECTION
# ==========================================
print(f"Starting feature selection on {X_train_dense_raw.shape[1]} features...")

selector_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')
selector = SelectFromModel(selector_model, threshold='1.25*mean')
# FIT ON TRAINING ONLY
X_train_dense_sel = selector.fit_transform(X_train_dense_raw, y_train)
# TRANSFORM VALIDATION
X_val_dense_sel = selector.transform(X_val_dense_raw) 

selected_feature_names = X_dense_full.columns[selector.get_support()]
print(f"Reduced feature set from {X_train_dense_raw.shape[1]} to {X_train_dense_sel.shape[1]} features.")


# --- NEW: PLOT FEATURE IMPORTANCE ---
print("Generating Feature Importance Plot...")

# Extract importances from the fitted XGBoost model inside the selector
importances = selector.estimator_.feature_importances_

# Create a DataFrame for easy sorting
importance_df = pd.DataFrame({
    'Feature': X_train_dense_raw.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the Top 20 features
plt.figure(figsize=(10, 8))
sns.barplot(
    data=importance_df.head(20), 
    x='Importance', 
    y='Feature', 
    hue='Feature', 
    legend=False, 
    palette='viridis'
)
plt.title("Top 20 Feature Importances (From XGBoost Selector)")
plt.xlabel("Importance (Gain)")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()

# --- NEW: SHAP VALUE PLOT ---
print("Generating SHAP Summary Plot...")

# Get the actual fitted XGBoost model from inside the selector
fitted_xgb = selector.estimator_

# Create the explainer
explainer = shap.TreeExplainer(fitted_xgb)

# CRITICAL FIX: Pass the RAW validation data (all 72 features) so the dimensions match!
# We also take a random sample of 10,000 rows so SHAP calculates in seconds instead of minutes.
X_val_df = pd.DataFrame(X_val_dense_raw, columns=X_train_dense_raw.columns).sample(10000, random_state=42)

shap_values = explainer.shap_values(X_val_df)

# Generate the plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_val_df)
# ------------------------------------
# ------------------------------------
# ==========================================
# 6. IMPUTE AND SCALE
# ==========================================
imputer = SimpleImputer(strategy='constant', fill_value=0.0)
X_train_dense_imp = imputer.fit_transform(X_train_dense_sel) # Fit on Train
X_val_dense_imp = imputer.transform(X_val_dense_sel)         # Transform Val

scaler = StandardScaler()
X_train_dense_scaled = scaler.fit_transform(X_train_dense_imp) # Fit on Train
X_val_dense_scaled = scaler.transform(X_val_dense_imp)         # Transform Val

# Rename for your downstream models
X_train_dense = X_train_dense_scaled
X_val_dense = X_val_dense_scaled
X_train = X_train_dense
X_val = X_val_dense
NUM_DENSE_FEATURES = X_train.shape[1]
initial_bias = np.log([pos_count / neg_count])
output_bias = tf.keras.initializers.Constant(initial_bias)
print("\nData pipeline complete. Ready for modeling.")
#%% ==========================================
# 1. LOGISTIC REGRESSION MODEL (Unweighted)
# ==========================================
print("--- Training Logistic Regression ---")
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

log_probs = log_model.predict_proba(X_val)[:, 1]
log_preds = (log_probs >= 0.5).astype(int)

log_brier = brier_score_loss(y_val, log_probs)
log_auc = roc_auc_score(y_val, log_probs)
log_prauc = average_precision_score(y_val, log_probs)
log_ll = log_loss(y_val, log_probs)
log_acc = accuracy_score(y_val, log_preds)


#%% ==========================================
# 2. XGBOOST MODEL (Optuna Tuning)
# ==========================================
print("--- Tuning XGBoost ---")

def objective_xgb(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10), # Let trees get a bit deeper
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # Helps prevent overfitting deep trees
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    xgb_base = XGBClassifier(**param)
    # Using cv=3 inside the trial to save a bit of time
    xgb_calibrated = CalibratedClassifierCV(estimator=xgb_base, method='sigmoid', cv=3)
    xgb_calibrated.fit(X_train, y_train)
    
    preds = xgb_calibrated.predict_proba(X_val)[:, 1]
    return brier_score_loss(y_val, preds)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=20) # Increase n_trials if you have time

print("\n--- Training Best XGBoost ---")
best_xgb_base = XGBClassifier(**study_xgb.best_params, eval_metric='logloss', random_state=42)
xgb_calibrated = CalibratedClassifierCV(estimator=best_xgb_base, method='isotonic', cv=5)
xgb_calibrated.fit(X_train, y_train)

xgb_probs = xgb_calibrated.predict_proba(X_val)[:, 1]
xgb_preds = (xgb_probs >= 0.5).astype(int)

xgb_brier = brier_score_loss(y_val, xgb_probs)
xgb_auc = roc_auc_score(y_val, xgb_probs)
xgb_prauc = average_precision_score(y_val, xgb_probs)
xgb_ll = log_loss(y_val, xgb_probs)
xgb_acc = accuracy_score(y_val, xgb_preds)

#%% ==========================================
# 3. NEURAL NETWORK (Optuna Tuning)
# ==========================================
print("\n--- Tuning Neural Network (Dense Only) ---")

def build_nn(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    l1_reg = trial.suggest_float("l1_reg", 1e-5, 1e-2, log=True)
    dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.5)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.4)
    units_1 = trial.suggest_int("units_1", 64, 256)
    units_2 = trial.suggest_int("units_2", 32, 128)

    dense_input = Input(shape=(NUM_DENSE_FEATURES,), name="dense_input")
    x = Dense(units_1, activation='relu', kernel_regularizer=regularizers.l1(l1_reg))(dense_input) 
    x = BatchNormalization()(x)
    x = Dropout(dropout_1)(x)
    x = Dense(units_2, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_2)(x)
    output = Dense(1, activation='sigmoid', bias_initializer=output_bias, name="buy_probability")(x)

    model = Model(inputs=dense_input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.MeanSquaredError(name='brier_score')]
    )
    return model

def objective_nn(trial):
    model = build_nn(trial)
    early_stop = EarlyStopping(monitor='val_brier_score', mode='min', patience=2, restore_best_weights=True, verbose=1)
    
    model.fit(
        x=X_train, y=y_train, validation_data=(X_val, y_val),
        batch_size=256, epochs=10, callbacks=[early_stop], verbose=1
    )
    
    probs = model.predict(X_val, verbose=1).ravel()
    return brier_score_loss(y_val, probs)

study_nn = optuna.create_study(direction='minimize')
study_nn.optimize(objective_nn, n_trials=10) # Kept low for neural nets, increase if needed

print("\n--- Training Best Neural Network ---")
nn_model = build_nn(study_nn.best_trial)
early_stop = EarlyStopping(monitor='val_brier_score', mode='min', patience=3, restore_best_weights=True, verbose=1)
history = nn_model.fit(
    x=X_train, y=y_train, validation_data=(X_val, y_val),
    batch_size=256, epochs=20, callbacks=[early_stop], verbose=1 
)

nn_probs = nn_model.predict(X_val, verbose=1).ravel()
nn_preds = (nn_probs >= 0.5).astype(int)

nn_brier = brier_score_loss(y_val, nn_probs)
nn_auc = roc_auc_score(y_val, nn_probs)
nn_prauc = average_precision_score(y_val, nn_probs)
nn_ll = log_loss(y_val, nn_probs)
nn_acc = accuracy_score(y_val, nn_preds)
#%%
pd.DataFrame({"nn": nn_probs, "xgb": xgb_probs, "log": log_probs, "lstm": lstm_probs, "wd": wd_probs}).corr()
#%%
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, log_loss, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ==========================================
# 7. STANDALONE LSTM (Optuna Tuning)
# ==========================================
print("\n--- Tuning Standalone LSTM ---")

def build_lstm_only(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lstm_units = trial.suggest_int("lstm_units", 16, 64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

    # Inputs: Changed to a single 3D State array (MAX_SEQ_LEN, 3 features)
    state_input = tf.keras.layers.Input(shape=(MAX_SEQ_LEN, 3), name="state_input")

    # Masking (Replaces Embedding) - tells LSTM to ignore 0.0 padding
    x_masked = tf.keras.layers.Masking(mask_value=0.0)(state_input)

    # LSTM Branch
    x_lstm = tf.keras.layers.LSTM(lstm_units)(x_masked)
    x_lstm = tf.keras.layers.BatchNormalization()(x_lstm)
    x_lstm = tf.keras.layers.Dropout(dropout_rate)(x_lstm)

    # Classification Head
    x = tf.keras.layers.Dense(32, activation='relu')(x_lstm)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias, name="shipping_probability")(x)

    model = tf.keras.models.Model(inputs=state_input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error', 
        metrics=[tf.keras.metrics.MeanSquaredError(name='brier_score')]
    )
    return model

def build_hybrid_lstm(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lstm_units = trial.suggest_int("lstm_units", 32, 128) # Slightly larger to handle more info
    
    # 1. Input A: Categorical Event IDs (shape: 50)
    seq_input = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name="seq_input")
    # Embedding layer translates IDs into 16-dimension dense vectors
    x_seq = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=16, mask_zero=True)(seq_input)
    
    # 2. Input B: Continuous States (shape: 50, 3)
    state_input = tf.keras.layers.Input(shape=(MAX_SEQ_LEN, 3), name="state_input")
    # NOTE: You must apply a StandardScaler to your state_input before passing it to the model!
    
    # 3. Combine them at every time step
    # Now the LSTM sees: [Embedded Action + Hours + Carts + Checkouts] all at once
    combined_seq = tf.keras.layers.Concatenate(axis=-1)([x_seq, state_input])

    # 4. The LSTM Branch
    x_lstm = tf.keras.layers.LSTM(lstm_units)(combined_seq)
    x_lstm = tf.keras.layers.BatchNormalization()(x_lstm)
    x_lstm = tf.keras.layers.Dropout(0.3)(x_lstm)

    # Classification Head
    x = tf.keras.layers.Dense(32, activation='relu')(x_lstm)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name="shipping_probability")(x)

    model = tf.keras.models.Model(inputs=[seq_input, state_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    
    return model
# Then, in your Optuna study:
# Optuna Objective
def objective_lstm(trial):
    model = build_hybrid_lstm(trial)
    
    # CRITICAL FIX: Pass BOTH arrays as a list to x= and validation_data=
    model.fit(
        x=[X_train_seq, X_train_state], 
        y=y_train,
        validation_data=([X_val_seq, X_val_state], y_val),
        batch_size=1024, epochs=5, 
        verbose=1
    )
    
    # Also pass both arrays as a list to the predict function
    probs = model.predict([X_val_seq, X_val_state], verbose=0).ravel()
    
    return brier_score_loss(y_val, probs)
# Run Optimization
study_lstm = optuna.create_study(direction='minimize')
study_lstm.optimize(objective_lstm, n_trials=20) # Adjust trials based on time
#%%
print("\n--- Training Best LSTM Model ---")
best_lstm_model = build_hybrid_lstm(study_lstm.best_trial)
early_stop = EarlyStopping(monitor='val_brier_score', mode='min', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_lstm_model.keras", monitor='val_brier_score', save_best_only=True)

lstm_history = best_lstm_model.fit(
    # CRITICAL FIX: Pass BOTH arrays as a list here
    x=[X_train_seq, X_train_state], 
    y=y_train,
    # CRITICAL FIX: Pass BOTH arrays as a list here as well
    validation_data=([X_val_seq, X_val_state], y_val),
    batch_size=256, epochs=10, callbacks=[early_stop, checkpoint], verbose=1 
)

# --- EVALUATION ---
print("\n--- LSTM Performance Stats ---")
# CRITICAL FIX: Pass BOTH arrays as a list to the predict function
lstm_probs = best_lstm_model.predict([X_val_seq, X_val_state], verbose=1).ravel()
lstm_preds = (lstm_probs >= 0.5).astype(int)

# Calculate Stats
lstm_brier = brier_score_loss(y_val, lstm_probs)
lstm_auc = roc_auc_score(y_val, lstm_probs)
lstm_prauc = average_precision_score(y_val, lstm_probs)
lstm_ll = log_loss(y_val, lstm_probs)
lstm_acc = accuracy_score(y_val, lstm_preds)

print(f"Brier Score: {lstm_brier:.5f}")
print(f"ROC-AUC:     {lstm_auc:.5f}")
print(f"PR-AUC:      {lstm_prauc:.5f}")
print(f"Log Loss:    {lstm_ll:.5f}")
print(f"Accuracy:    {lstm_acc:.5f}")
#%%
# ==========================================
# 4. WIDE & DEEP NEURAL NETWORK (Optuna Tuning)
# ==========================================
print("\n--- Tuning Wide & Deep NN ---")

def build_wd(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    l1_reg = trial.suggest_float("l1_reg", 1e-5, 1e-2, log=True)
    lstm_units = trial.suggest_int("lstm_units", 16, 64)
    dense_units = trial.suggest_int("dense_units", 64, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

    # BRANCH A: Dense
    wd_dense_input = Input(shape=(NUM_DENSE_FEATURES,), name="wd_dense_input")
    x_dense = Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l1(l1_reg))(wd_dense_input) 
    x_dense = BatchNormalization()(x_dense)
    x_dense = Dropout(dropout_rate)(x_dense)

    # BRANCH B: LSTM (Now with Time Deltas!)
    seq_input = Input(shape=(MAX_SEQ_LEN,), name="seq_input")
    time_input = Input(shape=(MAX_SEQ_LEN, 1), name="time_input") # <--- 3rd Input Layer

    # Embed the actions
    x_seq = Embedding(input_dim=VOCAB_SIZE, output_dim=16, mask_zero=True)(seq_input)
    
    # Merge the action with the time gap at every single time step
    combined_seq = Concatenate(axis=-1)([x_seq, time_input])

    # Feed the combined "Action + Time" sequence into the LSTM
    x_lstm = LSTM(lstm_units)(combined_seq)
    x_lstm = Dropout(dropout_rate)(x_lstm)

    # COMBINE
    combined = Concatenate()([x_dense, x_lstm])
    x_comb = Dense(64, activation='relu')(combined)
    x_comb = Dense(32, activation='relu')(x_comb)
    x_comb = Dropout(0.2)(x_comb)
    
    wd_output = Dense(1, activation='sigmoid', bias_initializer=output_bias, name="wd_buy_probability")(x_comb)

    # ---> Tell the model to expect 3 inputs <---
    model = Model(inputs=[wd_dense_input, seq_input, time_input], outputs=wd_output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.MeanSquaredError(name='brier_score')]
    )
    return model

def objective_wd(trial):
    model = build_wd(trial)
    wd_early_stop = EarlyStopping(monitor='val_brier_score', mode='min', patience=3, restore_best_weights=True, verbose=1)
    
    model.fit(
        x=[X_train_dense, X_train_seq, X_train_time], # <--- Passing 3 inputs
        y=y_train,
        validation_data=([X_val_dense, X_val_seq, X_val_time], y_val), # <--- Passing 3 inputs
        batch_size=256, epochs=15, callbacks=[wd_early_stop], verbose=1
    )
    
    probs = model.predict([X_val_dense, X_val_seq, X_val_time], verbose=1).ravel()
    return brier_score_loss(y_val, probs)

study_wd = optuna.create_study(direction='minimize')
study_wd.optimize(objective_wd, n_trials=1)

print("\n--- Training Best Wide & Deep NN ---")
wd_model = build_wd(study_wd.best_trial)
wd_early_stop = EarlyStopping(monitor='val_brier_score', mode='min', patience=3, restore_best_weights=True, verbose=1)

wd_history = wd_model.fit(
    x=[X_train_dense, X_train_seq, X_train_time], # <--- Passing 3 inputs
    y=y_train,
    validation_data=([X_val_dense, X_val_seq, X_val_time], y_val), # <--- Passing 3 inputs
    batch_size=256, epochs=20, callbacks=[wd_early_stop], verbose=1 
)

wd_probs = wd_model.predict([X_val_dense, X_val_seq, X_val_time], verbose=1).ravel()
wd_preds = (wd_probs >= 0.5).astype(int)

wd_brier = brier_score_loss(y_val, wd_probs)
wd_auc = roc_auc_score(y_val, wd_probs)
wd_prauc = average_precision_score(y_val, wd_probs)
wd_ll = log_loss(y_val, wd_probs)
wd_acc = accuracy_score(y_val, wd_preds)
#%% ==========================================
# FINAL COMPARISON
# ==========================================
print("\n=====================================================================================")
print(f"{'FINAL MODEL COMPARISON (Validation Set)':^85}")
print("=====================================================================================")
print(f"{'Model':<22} | {'Brier (↓)':<10} | {'Log Loss (↓)':<12} | {'ROC-AUC (↑)':<11} | {'PR-AUC (↑)':<10} | {'Accuracy':<8}")
print("-" * 85)
# Make sure log_brier, log_ll, log_auc, log_prauc, log_acc are defined from your LR run earlier!
print(f"{'Logistic Regression':<22} | {log_brier:<10.4f} | {log_ll:<12.4f} | {log_auc:<11.4f} | {log_prauc:<10.4f} | {log_acc:<8.4f}")
print(f"{'XGBoost (Calibrated)':<22} | {xgb_brier:<10.4f} | {xgb_ll:<12.4f} | {xgb_auc:<11.4f} | {xgb_prauc:<10.4f} | {xgb_acc:<8.4f}")
print(f"{'Standard Neural Net':<22} | {nn_brier:<10.4f} | {nn_ll:<12.4f} | {nn_auc:<11.4f} | {nn_prauc:<10.4f} | {nn_acc:<8.4f}")
print(f"{'LSTM Neural Net':<22} | {lstm_brier:<10.4f} | {lstm_ll:<12.4f} | {lstm_auc:<11.4f} | {lstm_prauc:<10.4f} | {lstm_acc:<8.4f}")
print(f"{'Wide & Deep NN (LSTM)':<22} | {wd_brier:<10.4f} | {wd_ll:<12.4f} | {wd_auc:<11.4f} | {wd_prauc:<10.4f} | {wd_acc:<8.4f}")
print("=====================================================================================")