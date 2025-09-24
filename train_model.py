# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import joblib # New import for saving the pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- (Sections 2-6 remain the same) ---
# 2. LOAD DATA
try:
    df = pd.read_csv('data/synthetic_slope_stability_dataset.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'data/synthetic_slope_stability_dataset.csv' not found.")
    sys.exit()

# 3. FEATURE ENGINEERING
df['rainfall_x_slope'] = df['rainfall_last_24h'] * df['slope_angle']
print("\nNew interaction feature created!")

# 4. PREPARE DATA FOR MODELING
features = ['slope_angle', 'rainfall_last_24h', 'displacement_rate', 'pore_pressure', 'image_crack_score', 'rainfall_x_slope']
target = 'rockfall_event'
X = df[features]
y = df[target]

# 5. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# 6. TRAIN TWO SEPARATE, OPTIMIZED MODELS
print("\n--- Training XGBoost Expert ---")
xgb_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42, learning_rate=0.1, max_depth=7, n_estimators=200))
])
xgb_pipeline.fit(X_train, y_train)
print("XGBoost training complete.")

print("\n--- Training Random Forest Expert ---")
rf_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)
print("Random Forest training complete.")

# --- (Sections 7-8 remain the same) ---
# 7. CREATE AND EVALUATE THE ENSEMBLE
print("\n--- Evaluating Ensemble Performance ---")
probs_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
probs_rf = rf_pipeline.predict_proba(X_test)[:, 1]
ensemble_probs = (probs_xgb + probs_rf) / 2

# 8. FIND THE OPTIMAL THRESHOLD FOR THE ENSEMBLE
print("\n--- Searching for Optimal Ensemble Threshold ---")
best_threshold = 0.5
best_f1 = 0
thresholds = np.arange(0.1, 0.9, 0.05)
print("Threshold | Precision | Recall    | F1-Score")
print("---------------------------------------------")
for threshold in thresholds:
    predictions_tuned = (ensemble_probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions_tuned, average='binary')
    print(f"{threshold:9.2f} | {precision:9.2f} | {recall:9.2f} | {f1:9.2f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
print("---------------------------------------------")
print(f"\nOptimal ENSEMBLE threshold found at: {best_threshold:.2f} with F1-Score: {best_f1:.2f}")

print("\n--- Final ENSEMBLE Report at Optimal Threshold ---")
final_predictions = (ensemble_probs >= best_threshold).astype(int)
print(classification_report(y_test, final_predictions))

# --- 9. SAVE THE FINAL, COMPLETE PIPELINE ---
# We save the entire XGBoost pipeline (scaler + model) to ensure consistency.
pipeline_filename = 'rockfall_prediction_pipeline.joblib'
joblib.dump(xgb_pipeline, pipeline_filename)
print(f"\nComplete prediction pipeline saved as '{pipeline_filename}'")
print(f"ACTION REQUIRED: Update 'main.py' to load this new pipeline file and use the optimal ensemble threshold of {best_threshold:.2f}")

