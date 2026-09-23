
# main.py -- Rockfall Prediction API
# ----------------------------------
# FastAPI backend for rockfall risk prediction, dynamic thresholds, and alerting.

import logging
from typing import Dict, Tuple

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from twilio.rest import Client

# =========================================================
# 1. SECURE CONFIGURATION MANAGEMENT
# =========================================================

class Settings(BaseSettings):
    """Application settings read from `.env` using pydantic-settings.

    Default thresholds are provided but can be overridden with environment
    variables or at runtime via the `/set-thresholds` endpoint.
    """
    OPTIMAL_THRESHOLD: float = 0.25
    THRESHOLD_GUARDED: float = 0.45
    THRESHOLD_ELEVATED: float = 0.70
    THRESHOLD_CRITICAL: float = 0.95

    # Twilio credentials and phone numbers. Optional so the backend can boot
    # for a local demo without real credentials; see .env.example. When left
    # unset (or set to a placeholder), SMS alerts are skipped — not sent.
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""
    YOUR_PHONE_NUMBER: str = ""

    class Config:
        env_file = ".env"


settings = Settings()

# =========================================================
# 2. STRUCTURED LOGGING & APP INITIALIZATION
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rockfall Prediction API",
    description="An API to predict rockfall incidents with a multi-level risk system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 3. LOAD THE AI PIPELINE
# =========================================================

try:
    pipeline = joblib.load('rockfall_prediction_pipeline.joblib')
    logger.info("Prediction pipeline loaded (scaler + model).")
except Exception as e:
    logger.error(f"Error loading pipeline: {e}")
    pipeline = None

# =========================================================
# 4. DATA MODELS
# =========================================================

class RockfallFeatures(BaseModel):
    """Request model containing all numeric features required by the model."""
    slope_angle: float
    rainfall_last_24h: float
    displacement_rate: float
    pore_pressure: float
    image_crack_score: float

class ThresholdUpdate(BaseModel):
    """Payload model for updating in-memory risk thresholds at runtime."""
    guarded: float
    elevated: float
    critical: float

# =========================================================
# 5. CORE BUSINESS LOGIC
# =========================================================

def get_risk_level(probability: float) -> Tuple[str, str]:
    """Return a (risk_level, color) tuple given a probability and current thresholds.

    Colors are simple labels used by the frontend (not strict hex values).
    """
    if probability >= settings.THRESHOLD_CRITICAL:
        return "CRITICAL", "red"
    if probability >= settings.THRESHOLD_ELEVATED:
        return "ELEVATED", "orange"
    if probability >= settings.THRESHOLD_GUARDED:
        return "GUARDED", "yellow"
    return "LOW", "green"

def is_twilio_configured() -> bool:
    """Return True only if all Twilio settings look like real values."""
    values = (
        settings.TWILIO_ACCOUNT_SID,
        settings.TWILIO_AUTH_TOKEN,
        settings.TWILIO_PHONE_NUMBER,
        settings.YOUR_PHONE_NUMBER,
    )
    return all(values) and not settings.TWILIO_ACCOUNT_SID.lower().startswith("acplaceholder")

def send_sms_alert(features: Dict[str, float], probability: float) -> None:
    """Send a CRITICAL alert SMS using Twilio.

    The function logs success/failure. It is intended to run as a background task
    (so errors should not interrupt the API response). Skipped entirely if
    Twilio credentials are unset or placeholders (see .env.example).
    """
    if not is_twilio_configured():
        logger.info("Twilio not configured — skipping SMS alert.")
        return
    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        details = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in features.items()])
        message_body = (
            "CRITICAL RISK ALERT:\n"
            "A simulation exceeded the CRITICAL threshold.\n"
            f"Risk Probability: {probability:.2%}\n\n"
            f"Input Parameters:\n{details}"
        )
        message = client.messages.create(
            body=message_body,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=settings.YOUR_PHONE_NUMBER,
        )
        logger.info(f"CRITICAL alert sent, SID={message.sid}")
    except Exception as e:
        logger.exception("Failed to send CRITICAL SMS alert")

# =========================================================
# 6. API ENDPOINTS
# =========================================================

@app.get("/")
def read_root() -> Dict[str, str]:
    """Simple health-check/root endpoint."""
    return {"message": "Rockfall Prediction API is running"}

@app.post("/predict")
def predict_rockfall(features: RockfallFeatures, background_tasks: BackgroundTasks):
    """Predict the probability of rockfall and return a risk summary.

    If the result is CRITICAL (and the input isn't all zeros) an SMS alert is
    queued as a background task so the API response is not blocked.
    """
    if pipeline is None:
        return {"error": "Prediction pipeline is not loaded."}

    # Prepare input with the derived interaction feature used during training
    feature_dict = features.dict()
    feature_dict['rainfall_x_slope'] = feature_dict['rainfall_last_24h'] * feature_dict['slope_angle']
    model_feature_names = [
        'slope_angle', 'rainfall_last_24h', 'displacement_rate',
        'pore_pressure', 'image_crack_score', 'rainfall_x_slope'
    ]
    input_df = pd.DataFrame([feature_dict], columns=model_feature_names)

    probability = float(pipeline.predict_proba(input_df)[0][1])
    risk_level, color = get_risk_level(probability)

    # Avoid spurious alerts when the user submits an all-zero feature vector
    is_zero_input = all(value == 0 for value in features.dict().values())
    if risk_level == "CRITICAL" and not is_zero_input:
        logger.info("CRITICAL risk detected — queueing SMS alert")
        background_tasks.add_task(send_sms_alert, features.dict(), probability)

    return {
        "risk_level": risk_level,
        "color": color,
        "probability_of_rockfall": probability,
    }

@app.post("/set-thresholds")
def set_thresholds(new_thresholds: ThresholdUpdate):
    """Update the in-memory risk thresholds used for classification.

    These changes are not persisted to disk; they only affect the running
    instance and will be reset when the process restarts.
    """
    settings.THRESHOLD_GUARDED = new_thresholds.guarded
    settings.THRESHOLD_ELEVATED = new_thresholds.elevated
    settings.THRESHOLD_CRITICAL = new_thresholds.critical
    logger.info(
        "Risk thresholds updated: GUARDED=%s, ELEVATED=%s, CRITICAL=%s",
        settings.THRESHOLD_GUARDED,
        settings.THRESHOLD_ELEVATED,
        settings.THRESHOLD_CRITICAL,
    )
    return {"message": "Thresholds updated successfully", "new_thresholds": new_thresholds.dict()}
