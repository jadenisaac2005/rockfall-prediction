import random
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client

# --- 1. CONFIGURATION ---
# Replace these with your actual Twilio credentials.
# ⚠️ IMPORTANT: Do not commit this file to GitHub with your real credentials!
TWILIO_ACCOUNT_SID = "AC3*********************************"
TWILIO_AUTH_TOKEN = "03a**********************************"
TWILIO_PHONE_NUMBER = "+1**********"  #  Twilio phone number
YOUR_PHONE_NUMBER = "+91**********"   #  verified personal phone number
RISK_THRESHOLD = 0.80  # Send an alert if probability is above 80%

# --- 2. App Initialization & CORS ---
app = FastAPI(
    title="Rockfall Prediction API",
    description="An API to predict rockfall incidents in open-pit mines.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Load The AI Model ---
try:
    model = xgb.XGBClassifier()
    model.load_model('rockfall_predictor_xgb.json')
    print("✅ XGBoost model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- 4. Define Data Input Model ---
class RockfallFeatures(BaseModel):
    slope_angle: float
    rainfall_last_24h: float
    displacement_rate: float
    pore_pressure: float
    image_crack_score: float

# --- 5. Alerting Function ---
def send_user_alert(features: dict, probability: float):
    """Sends an SMS alert based on user input from the dashboard."""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        # Create a detailed message from the user's input
        details = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in features.items()])
        message_body = (
            f"CRITICAL RISK PREDICTION:\n"
            f"A manual simulation on the dashboard exceeded the risk threshold.\n"
            f"Risk Probability: {probability:.2%}\n\n"
            f"Input Parameters:\n{details}"
        )
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=YOUR_PHONE_NUMBER
        )
        print(f"✅ User-triggered alert sent successfully! SID: {message.sid}")
    except Exception as e:
        print(f"❌ Failed to send user-triggered alert. Error: {e}")


# --- 6. API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Rockfall Prediction API!"}

@app.post("/predict")
def predict_rockfall(features: RockfallFeatures):
    """Receives data, returns a prediction, and sends an alert if the threshold is crossed."""
    if model is None:
        return {"error": "Model is not loaded."}

    feature_names = [
        'slope_angle', 'rainfall_last_24h', 'displacement_rate',
        'pore_pressure', 'image_crack_score'
    ]

    input_df = pd.DataFrame([features.dict()], columns=feature_names)
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability > 0.5)

    # *** NEW: Check threshold and send alert if necessary ***
    if probability > RISK_THRESHOLD:
        print(f"🚨 High risk detected from user input! Probability: {probability:.2%}. Sending alert...")
        send_user_alert(features.dict(), probability)

    return {
        "prediction": prediction,
        "probability_of_rockfall": float(probability)
    }

# (The /risk-map endpoint remains the same)
mine_zones = {
    "zone_A": {"name": "North Quarry Face", "lat": 12.9716, "lon": 77.5946, "base_features": [45.0, 15.0, 0.5, 28.0, 0.1]},
    "zone_B": {"name": "East Haul Road", "lat": 12.9726, "lon": 77.6046, "base_features": [35.0, 80.0, 1.8, 90.0, 0.3]},
    "zone_C": {"name": "West Overburden Dump", "lat": 12.9706, "lon": 77.5906, "base_features": [48.0, 120.0, 2.5, 200.0, 0.8]},
    "zone_D": {"name": "South Stockpile", "lat": 12.9696, "lon": 77.5956, "base_features": [30.0, 5.0, 0.2, 20.0, 0.0]}
}

@app.get("/risk-map")
def get_risk_map_data():
    if model is None: return {"error": "Model is not loaded."}
    risk_data = []
    feature_names = ['slope_angle', 'rainfall_last_24h', 'displacement_rate', 'pore_pressure', 'image_crack_score']
    for zone_id, zone_info in mine_zones.items():
        simulated_data = [val * random.uniform(0.95, 1.05) for val in zone_info["base_features"]]
        input_df = pd.DataFrame([simulated_data], columns=feature_names)
        probability = model.predict_proba(input_df)[0][1]
        risk_data.append({
            "zone_id": zone_id, "name": zone_info["name"], "coords": [zone_info["lat"], zone_info["lon"]],
            "probability": float(probability), "prediction": int(probability > 0.5)
        })
    return risk_data
