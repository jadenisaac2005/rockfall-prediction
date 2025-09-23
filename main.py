import random
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. App Initialization ---
app = FastAPI(
    title="Rockfall Prediction API",
    description="An API to predict rockfall incidents in open-pit mines.",
    version="0.1.0",
)

# --- 2. CORS Configuration ---
# Allow all origins for flexible local development (e.g., opening dashboard.html)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Load The Trained AI Model ---
# This model is loaded once when the application starts.
try:
    model = xgb.XGBClassifier()
    model.load_model('rockfall_predictor_xgb.json')
    print("XGBoost model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 4. Define Data Input Model ---
class RockfallFeatures(BaseModel):
    slope_angle: float
    rainfall_last_24h: float
    displacement_rate: float
    pore_pressure: float
    image_crack_score: float

# --- 5. API Endpoints ---
@app.get("/")
def read_root():
    """Returns a welcome message for the API root."""
    return {"message": "Welcome to the Rockfall Prediction API!"}

@app.post("/predict")
def predict_rockfall(features: RockfallFeatures):
    """Receives rockfall feature data and returns a prediction."""
    if model is None:
        return {"error": "Model is not loaded."}

    feature_names = [
        'slope_angle', 'rainfall_last_24h', 'displacement_rate',
        'pore_pressure', 'image_crack_score'
    ]

    # Convert input data into a DataFrame the model can understand
    input_df = pd.DataFrame([features.dict()], columns=feature_names)

    # Make a prediction
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability > 0.5)

    return {
        "prediction": prediction,
        "probability_of_rockfall": float(probability)
    }

# --- 6. Real-Time Map Simulation Endpoint ---
# This section simulates data for the map on the dashboard.
# CORRECTED: Base features now have 5 values to match the new dataset schema.
mine_zones = {
    "zone_A": {"name": "North Quarry Face", "lat": 12.9716, "lon": 77.5946, "base_features": [45.0, 15.0, 0.5, 28.0, 0.1]},
    "zone_B": {"name": "East Haul Road", "lat": 12.9726, "lon": 77.6046, "base_features": [35.0, 80.0, 1.8, 90.0, 0.3]},
    "zone_C": {"name": "West Overburden Dump", "lat": 12.9706, "lon": 77.5906, "base_features": [48.0, 120.0, 2.5, 200.0, 0.8]}, # High risk zone
    "zone_D": {"name": "South Stockpile", "lat": 12.9696, "lon": 77.5956, "base_features": [30.0, 5.0, 0.2, 20.0, 0.0]}
}

@app.get("/risk-map")
def get_risk_map_data():
    """Generates simulated real-time data for all mine zones and returns their risks."""
    if model is None:
        return {"error": "Model is not loaded."}

    risk_data = []
    feature_names = [
        'slope_angle', 'rainfall_last_24h', 'displacement_rate',
        'pore_pressure', 'image_crack_score'
    ]

    for zone_id, zone_info in mine_zones.items():
        # Simulate real-time fluctuations
        simulated_data = [val * random.uniform(0.95, 1.05) for val in zone_info["base_features"]]
        input_df = pd.DataFrame([simulated_data], columns=feature_names)

        # Make prediction for the zone
        probability = model.predict_proba(input_df)[0][1]

        risk_data.append({
            "zone_id": zone_id,
            "name": zone_info["name"],
            "coords": [zone_info["lat"], zone_info["lon"]],
            "probability": float(probability),
            "prediction": int(probability > 0.5)
        })
    return risk_data
