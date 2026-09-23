"""train_model.py
Train and export a rockfall prediction pipeline.

This script trains a single pipeline (SMOTE -> StandardScaler -> XGBoost),
searches for an optimal classification threshold on a validation split,
reports evaluation metrics on a held-out test split, and saves the pipeline.

Notes:
- The saved pipeline (rockfall_prediction_pipeline.joblib) is exactly the
  model that is evaluated here — every reported metric describes it.
- The chosen threshold is printed and saved for reference; main.py does not
  currently read it (see README's "Model & Data" section).
"""

from typing import Tuple

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


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


def build_pipeline(random_state: int = 42):
    """Construct the XGBoost training pipeline (SMOTE -> StandardScaler -> XGBoost).

    This is exactly the pipeline that gets saved and served by main.py.
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

    return ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('scaler', StandardScaler()),
        ('xgb', xgb_clf),
    ])


def search_optimal_threshold(y_true, probs, thresholds=None):
    """Search a list of thresholds and return the best threshold by F1 score.

    Returns (best_threshold, best_f1, per_threshold_scores) where
    per_threshold_scores is a list of {threshold, precision, recall, f1} dicts.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    best_threshold = 0.5
    best_f1 = -1.0
    per_threshold_scores = []

    print("Threshold | Precision | Recall    | F1-Score")
    print("---------------------------------------------")
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        print(f"{threshold:9.2f} | {precision:9.2f} | {recall:9.2f} | {f1:9.2f}")
        per_threshold_scores.append({
            'threshold': round(float(threshold), 2),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        })
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    print("---------------------------------------------")
    return best_threshold, best_f1, per_threshold_scores


def main():
    # Configuration
    data_path = 'data/synthetic_slope_stability_dataset.csv'
    pipeline_filename = 'rockfall_prediction_pipeline.joblib'
    metrics_path = 'results/metrics.json'

    # Load and prepare data
    df = load_data(data_path)
    X, y = prepare_features(df)

    # Train / validation / test split (60/20/20, stratified).
    # The threshold is chosen on the validation split; the test split is
    # reported on once, untouched by threshold selection, to avoid an
    # optimistic (leaked) estimate of performance.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Build and train the pipeline (on the training split only). This is
    # exactly the pipeline saved below and served by main.py — every metric
    # computed past this point describes this model, not a variant of it.
    pipeline = build_pipeline(random_state=42)
    print("\nTraining XGBoost pipeline...")
    pipeline.fit(X_train, y_train)
    print("XGBoost training complete.")

    # Threshold selection on the validation split
    print("\nEvaluating on validation split...")
    val_probs = pipeline.predict_proba(X_val)[:, 1]

    print("\nSearching for optimal threshold (on validation split)...")
    best_threshold, best_f1, per_threshold_scores = search_optimal_threshold(y_val, val_probs)
    print(f"\nOptimal threshold (chosen on validation): {best_threshold:.2f} (validation F1 = {best_f1:.2f})")

    # Final, one-time report on the untouched test split
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    final_preds = (test_probs >= best_threshold).astype(int)
    report_text = classification_report(y_test, final_preds, zero_division=0)
    report_dict = classification_report(y_test, final_preds, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, final_preds)
    tn, fp, fn, tp = cm.ravel()
    print("\nFinal classification report on TEST split at the validation-chosen threshold:")
    print(report_text)

    # Additional test-set metrics
    majority_class = y_test.value_counts().idxmax()
    majority_baseline_accuracy = (y_test == majority_class).mean()
    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\nMajority-class baseline accuracy (test): {majority_baseline_accuracy:.4f}")
    print(f"ROC-AUC (test): {roc_auc:.4f}")
    print(f"PR-AUC (test): {pr_auc:.4f}")
    print(f"False positive rate at threshold {best_threshold:.2f} (test): {false_positive_rate:.4f}")

    # Save trained pipeline (scaler + model) — this is the exact model evaluated above
    joblib.dump(pipeline, pipeline_filename)
    print(f"\nSaved pipeline to '{pipeline_filename}'")
    print(f"NOTE: main.py's risk-level cutoffs (GUARDED/ELEVATED/CRITICAL) are independent of this threshold — see README.")

    # Save metrics for reference (not consumed by main.py)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics = {
        'model': 'xgboost_pipeline (SMOTE->StandardScaler->XGBClassifier) — the exact pipeline saved to rockfall_prediction_pipeline.joblib and served by main.py',
        'threshold_selection': 'chosen on the validation split (max F1), evaluated once on a held-out test split',
        'optimal_threshold': round(float(best_threshold), 2),
        'optimal_threshold_validation_f1': float(best_f1),
        'per_threshold_scores_validation': per_threshold_scores,
        'classification_report_test': report_dict,
        'confusion_matrix_test': {
            'labels': ['no_rockfall', 'rockfall'],
            'matrix': cm.tolist(),
        },
        'majority_class_baseline_accuracy_test': float(majority_baseline_accuracy),
        'roc_auc_test': float(roc_auc),
        'pr_auc_test': float(pr_auc),
        'false_positive_rate_test': float(false_positive_rate),
        'train_shape': list(X_train.shape),
        'validation_shape': list(X_val.shape),
        'test_shape': list(X_test.shape),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to '{metrics_path}'")


if __name__ == '__main__':
    main()
