from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import sys
from fastapi.middleware.cors import CORSMiddleware
import random

# 1. Create an instance of the FastAPI class
app = FastAPI(
    title="Rockfall Prediction API",
    description="An API to predict rockfall incidents in open-pit mines.",
    version="0.1.0",
)

# Define the data model for the input features
# main.py

class RockfallFeatures(BaseModel):
    slope_angle: float
    rainfall_last_24h: float
    displacement_rate: float
    pore_pressure: float
    image_crack_score: float

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('rockfall_predictor_xgb.json')

# This allows your React app (running on localhost:5173) to make requests to your API
origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

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
# main.py

@app.post("/predict")
def predict_rockfall(features: RockfallFeatures):
    """
    Receives rockfall feature data and returns a prediction.
    """
    # 1. Define the feature names in the correct order for the model
    feature_names = [
        'slope_angle',
        'rainfall_last_24h',
        'displacement_rate',
        'pore_pressure',
        'image_crack_score'
    ]

    # 2. Create a DataFrame from the input features
    input_df = pd.DataFrame([features.dict()], columns=feature_names)

    # 3. Make a prediction
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = int(prediction_proba > 0.5)

    # 4. Return the result
    return {
        "prediction": prediction,
        "probability_of_rockfall": float(prediction_proba)
    }

# --- SIMULATED MINE ZONES ---
# In a real system, this would come from a database.
# We're defining the geographic coordinates and base sensor values for 4 zones.
mine_zones = {
    "zone_A": {"name": "North Quarry Face", "lat": 12.9716, "lon": 77.5946, "base_features": [0.5, 0.002, 15.0, 2.0, 28.0, 45.0, 0.1]},
    "zone_B": {"name": "East Haul Road", "lat": 12.9726, "lon": 77.6046, "base_features": [1.8, 0.009, 30.0, 5.0, 27.0, 35.0, 0.3]},
    "zone_C": {"name": "West Overburden Dump", "lat": 12.9706, "lon": 77.5906, "base_features": [2.5, 0.013, 42.0, 12.0, 24.0, 48.0, 0.6]},
    "zone_D": {"name": "South Stockpile", "lat": 12.9696, "lon": 77.5956, "base_features": [0.2, 0.001, 10.0, 0.0, 30.0, 30.0, 0.0]}
}

feature_names = [
    'displacement_cm', 'strain', 'pore_pressure_kPa',
    'rainfall_mm', 'temperature_c', 'slope_angle', 'image_crack_score'
]
# -----------------------------

@app.get("/risk-map")
def get_risk_map_data():
    """
    Generates simulated real-time data for all mine zones and returns their risks.
    """
    risk_data = []
    for zone_id, zone_info in mine_zones.items():
        # Generate slightly randomized data to simulate real-time fluctuations
        simulated_data = [val * random.uniform(0.95, 1.05) for val in zone_info["base_features"]]

        input_df = pd.DataFrame([simulated_data], columns=feature_names)

        # Make prediction
        probability = model.predict_proba(input_df)[0][1]
        prediction = int(probability > 0.5)

        risk_data.append({
            "zone_id": zone_id,
            "name": zone_info["name"],
            "coords": [zone_info["lat"], zone_info["lon"]],
            "probability": float(probability),
            "prediction": prediction
        })
    return risk_data
