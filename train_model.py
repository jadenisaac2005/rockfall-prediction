"""train_model.py
Train and export a rockfall prediction pipeline.

This script trains two classifiers (XGBoost and RandomForest) using a small
pipeline (SMOTE -> StandardScaler -> estimator), evaluates an average
ensemble of their probabilities, searches for an optimal classification
threshold, prints evaluation metrics, and saves the final pipeline.

Notes:
- The saved pipeline is the trained XGBoost pipeline (scaler + model).
- The ensemble threshold is printed for `main.py` to consume.
"""

from typing import Tuple

import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data from `path` and return a DataFrame.

    Exits the program with a clear message if the file cannot be found.
    """
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully")
        return df
    except FileNotFoundError:
        print(f"Error: '{path}' not found.")
        sys.exit(1)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create derived features and split DataFrame into X (features) and y (target)."""
    # Interaction feature
    df = df.copy()
    df['rainfall_x_slope'] = df['rainfall_last_24h'] * df['slope_angle']

    features = [
        'slope_angle', 'rainfall_last_24h', 'displacement_rate',
        'pore_pressure', 'image_crack_score', 'rainfall_x_slope'
    ]
    target = 'rockfall_event'
    X = df[features]
    y = df[target]
    return X, y


def build_pipelines(random_state: int = 42):
    """Construct two training pipelines: XGBoost and RandomForest.

    Each pipeline has the same preprocessing (SMOTE + StandardScaler) so we
    can compare model probabilities fairly.
    """
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state,
        learning_rate=0.1,
        max_depth=7,
        n_estimators=200,
    )

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)

    xgb_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('scaler', StandardScaler()),
        ('xgb', xgb_clf),
    ])

    rf_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('scaler', StandardScaler()),
        ('rf', rf_clf),
    ])

    return xgb_pipeline, rf_pipeline


def search_optimal_threshold(y_true, ensemble_probs, thresholds=None):
    """Search a list of thresholds and return the best threshold by F1 score.

    Returns (best_threshold, best_f1).
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    best_threshold = 0.5
    best_f1 = -1.0

    print("Threshold | Precision | Recall    | F1-Score")
    print("---------------------------------------------")
    for threshold in thresholds:
        preds = (ensemble_probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        print(f"{threshold:9.2f} | {precision:9.2f} | {recall:9.2f} | {f1:9.2f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    print("---------------------------------------------")
    return best_threshold, best_f1


def main():
    # Configuration
    data_path = 'data/synthetic_slope_stability_dataset.csv'
    pipeline_filename = 'rockfall_prediction_pipeline.joblib'

    # Load and prepare data
    df = load_data(data_path)
    X, y = prepare_features(df)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Build pipelines and train
    xgb_pipeline, rf_pipeline = build_pipelines(random_state=42)
    print("\nTraining XGBoost pipeline...")
    xgb_pipeline.fit(X_train, y_train)
    print("XGBoost training complete.")

    print("\nTraining Random Forest pipeline...")
    rf_pipeline.fit(X_train, y_train)
    print("Random Forest training complete.")

    # Ensemble evaluation
    print("\nEvaluating ensemble performance...")
    probs_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
    probs_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    ensemble_probs = (probs_xgb + probs_rf) / 2.0

    # Find optimal threshold
    print("\nSearching for optimal ensemble threshold...")
    best_threshold, best_f1 = search_optimal_threshold(y_test, ensemble_probs)
    print(f"\nOptimal ENSEMBLE threshold: {best_threshold:.2f} (F1 = {best_f1:.2f})")

    # Final report
    final_preds = (ensemble_probs >= best_threshold).astype(int)
    print("\nFinal classification report at optimal threshold:")
    print(classification_report(y_test, final_preds, zero_division=0))

    # Save trained XGBoost pipeline (scaler + model)
    joblib.dump(xgb_pipeline, pipeline_filename)
    print(f"\nSaved XGBoost pipeline to '{pipeline_filename}'")
    print(f"ACTION: Update 'main.py' to load this pipeline file and use ensemble threshold {best_threshold:.2f} if desired.")


if __name__ == '__main__':
    main()
