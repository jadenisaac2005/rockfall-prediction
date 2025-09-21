# 1. IMPORT LIBRARIES
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 2. LOAD DATA
import sys # Add this to your imports at the top of the file

try:
    # Make sure the path points to the data subfolder
    df = pd.read_csv('data/placeholder_data.csv')
    print("Data loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print("Error: 'data/placeholder_data.csv' not found. Make sure the file exists in the 'data' folder.")
    sys.exit() # Exit the script if the data can't be loaded

# 3. PREPARE DATA FOR MODELING
# For this first model, we will use only the numeric features.
# 'timestamp' and 'slope_id' will be ignored for now.
features = [
    'displacement_cm', 'strain', 'pore_pressure_kPa',
    'rainfall_mm', 'temperature_c', 'slope_angle', 'image_crack_score'
]
target = 'rockfall_event'

X = df[features]
y = df[target]

# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# test_size=0.3 means 30% of the data will be used for testing the model's performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 5. INITIALIZE AND TRAIN THE XGBOOST MODEL
# `scale_pos_weight` is a crucial parameter for imbalanced datasets.
# It tells the model to pay more attention to the minority class (rockfall events).
# A good starting value is (number of negatives / number of positives)
# In our tiny dataset, it's 7 / 1 = 7
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=7
)

print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training complete!")

# 6. EVALUATE THE MODEL
print("\nEvaluating model performance on the test set:")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# 7. SAVE THE TRAINED MODEL
# This model file is what your backend will use to make live predictions.
model_filename = 'rockfall_predictor_xgb.json'
model.save_model(model_filename)
print(f"\nModel saved as '{model_filename}'")
