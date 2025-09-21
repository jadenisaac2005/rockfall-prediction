from pydantic import BaseModel
from fastapi import FastAPI
import xgboost as xgb
import pandas as pd

# 1. Create an instance of the FastAPI class
app = FastAPI(
    title="Rockfall Prediction API",
    description="An API to predict rockfall incidents in open-pit mines.",
    version="0.1.0",
)

# Define the data model for the input features
class RockfallFeatures(BaseModel):
    displacement_cm: float
    strain: float
    pore_pressure_kPa: float
    rainfall_mm: float
    temperature_c: float
    slope_angle: float
    image_crack_score: float

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('rockfall_predictor_xgb.json')

# 2. Define a path operation decorator for the root URL
@app.get("/")
def read_root():
    """Returns a welcome message for the API root."""
    return {"message": "Welcome to the Rockfall Prediction API!"}


# 3. Define a health check endpoint
@app.get("/health")
def health_check():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok"}

# 4. Define the prediction endpoint
@app.post("/predict")
def predict_rockfall(features: RockfallFeatures):
    """
    Receives rockfall feature data and returns a prediction.
    """
    # 1. Convert the input features into a pandas DataFrame
    # The model expects the data in the same order as the training features
    feature_data = [
        features.displacement_cm,
        features.strain,
        features.pore_pressure_kPa,
        features.rainfall_mm,
        features.temperature_c,
        features.slope_angle,
        features.image_crack_score
    ]

    # Create a DataFrame with the correct feature names
    feature_names = [
        'displacement_cm', 'strain', 'pore_pressure_kPa',
        'rainfall_mm', 'temperature_c', 'slope_angle', 'image_crack_score'
    ]
    input_df = pd.DataFrame([feature_data], columns=feature_names)

    # 2. Make a prediction using the loaded model
    # predict_proba gives a probability score for each class (0 and 1)
    # We are interested in the probability of class 1 (rockfall)
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = int(prediction_proba > 0.5) # A simple threshold

    # 3. Return the result
    return {
        "prediction": prediction,
        "probability_of_rockfall": float(prediction_proba)
    }
