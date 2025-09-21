import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Rockfall Prediction Dashboard",
    page_icon="⛏️",
    layout="wide"
)

# --- API Endpoint ---
API_URL = "http://127.0.0.1:8000/predict"

# --- Page Title ---
st.title("⛏️ Rockfall Prediction Dashboard")
st.markdown("Enter the site parameters below to get a real-time rockfall risk assessment.")

st.divider()

# --- Input Form ---
with st.form("prediction_form"):
    st.header("Input Features")

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        displacement_cm = st.slider("Displacement (cm)", 0.0, 5.0, 2.0, 0.1)
        strain = st.slider("Strain", 0.0, 0.02, 0.01, 0.001, format="%.4f")
        pore_pressure_kPa = st.slider("Pore Pressure (kPa)", 0.0, 50.0, 40.0, 1.0)
        rainfall_mm = st.slider("Rainfall (mm)", 0.0, 50.0, 10.0, 1.0)

    with col2:
        temperature_c = st.slider("Temperature (°C)", -10.0, 40.0, 25.0, 1.0)
        slope_angle = st.slider("Slope Angle (°)", 20.0, 60.0, 45.0, 1.0)
        image_crack_score = st.slider("Image Crack Score", 0.0, 1.0, 0.5, 0.05)

    # Submit button
    submitted = st.form_submit_button("Predict Risk")

# --- Prediction Logic ---
if submitted:
    # 1. Create the JSON payload from the form inputs
    payload = {
        "displacement_cm": displacement_cm,
        "strain": strain,
        "pore_pressure_kPa": pore_pressure_kPa,
        "rainfall_mm": rainfall_mm,
        "temperature_c": temperature_c,
        "slope_angle": slope_angle,
        "image_crack_score": image_crack_score
    }

    # 2. Send the POST request to the FastAPI backend
    try:
        response = requests.post(API_URL, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for bad status codes

        result = response.json()

        # 3. Display the result
        st.subheader("Prediction Result")
        probability = result['probability_of_rockfall']

        if result['prediction'] == 1:
            st.error(f"High Risk of Rockfall! (Probability: {probability:.2%})")
        else:
            st.success(f"Low Risk of Rockfall. (Probability: {probability:.2%})")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API. Make sure the backend is running. Error: {e}")
