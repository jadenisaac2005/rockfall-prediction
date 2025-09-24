git clone <your-repository-url>

# ⛏️ AI-Powered Rockfall Prediction System

An intelligent, end-to-end system designed to predict and prevent rockfall incidents in open-pit mines. This project was developed for the **Smart India Hackathon** and combines a powerful machine learning backend with a real-time, interactive web dashboard to create a practical and life-saving tool.

## ✨ Live Dashboard Preview

Here is a preview of the interactive dashboard that provides mine planners with at-a-glance risk assessments.

![Dashboard Screenshot](https://github.com/jadenisaac2005/rockfall-prediction-sih25/blob/5a2733339007cc4714cfc6c17c8b00c2867f647e/media/Dashboard%20Screenshot.jpg)

## 🎯 Problem Statement

Al-Based Rockfall Prediction and Alert System for Open-Pit Mines

## 🚀 Key Features

- **⚡ Real-Time Prediction API:** High-performance backend built with **FastAPI** serving AI model predictions in milliseconds.
- **🧠 Machine Learning Core:** Trained **XGBoost** model (with scaler) to analyze multi-source data and calculate rockfall probabilities.
- **🗺️ Interactive Dashboard:** Single-page web app (React via CDN) for "what-if" analysis, live predictions, and risk visualization.
- **🎚️ Dynamic Risk Thresholds:** Users can adjust risk level thresholds (Guarded, Elevated, Critical) in the dashboard, which update the backend in real time.
- **📲 Automated Alert System:** Background worker sends SMS alerts via **Twilio** when a critical threshold is breached.
- **🎨 Color-Coded Risk Display:** Risk levels are shown in their respective colors (green/yellow/orange/red) for instant clarity.

## 🛠️ Technology Stack

| Area          | Technologies                                       |
| ------------- | -------------------------------------------------- |
| **Backend**   | Python, FastAPI, Uvicorn, joblib, pydantic-settings|
| **AI Model**  | XGBoost, Pandas, Scikit-learn                      |
| **Frontend**  | HTML, CSS, React (CDN), Babel                      |
| **Alerting**  | Twilio                                             |
| **Environment** | `venv` for Python, `.env` for secrets             |

## 🏃‍♀️ How to Run the Project

This project consists of two parts: the **Backend API** and the **Frontend Dashboard**

### Prerequisites

- Python 3.11 (managed via `pyenv` is recommended)
- Git

### 1. Backend Setup & Launch

First, get the AI server running.

```bash
# Clone the repository
git clone <your-repository-url>
cd <your-repo-name>

# Set up and activate the Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install all required Python packages
pip install -r requirements.txt

# Create a .env file in the project root with your Twilio credentials and phone numbers:
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=...
YOUR_PHONE_NUMBER=...

# Launch the FastAPI server
uvicorn main:app --reload
```

### 2. Frontend Dashboard

- Open `frontend/dashboard.html` directly in your browser (no build step required).
- The dashboard will connect to the backend at `http://127.0.0.1:8000` by default.

### 3. Dynamic Thresholds

- Use the **Risk Threshold Settings** panel in the dashboard to adjust Guarded, Elevated, and Critical thresholds.
- Click the reload icon to reset thresholds to their default values.
- Changes are sent to the backend and take effect immediately for all predictions.

## 📏 Project-Specific Conventions

- No JS frontend framework build step (React via CDN, single HTML file)
- Model pipeline file: `rockfall_prediction_pipeline.joblib` in project root
- All sensitive info in `.env` (never commit credentials)
- Thresholds are managed in backend memory and can be updated via API

## 🔗 Integration Points

- **Twilio**: SMS alerts, credentials set in `.env`
- **XGBoost**: Model pipeline must exist before backend serves predictions

## 🧩 Examples & Patterns

- See `frontend/dashboard.html` for `/predict` and `/set-thresholds` API usage
- `main.py` loads model pipeline and thresholds at startup; thresholds can be updated at runtime

## 🗝️ Key Files

- `main.py`
- `frontend/dashboard.html`
- `requirements.txt`
- `rockfall_prediction_pipeline.joblib`
- `data/`
- `.env`

---

**Update this file with new features, workflows, or integration points as the project evolves.**
Leave this terminal running. The API is now live at `http://127.0.0.1:8000`.

### 2. Frontend Launch

The frontend is a self-contained HTML file.

1. Navigate to the project folder in your file explorer.

2. Open the `dashboard.html` file directly in your web browser (like Chrome or Firefox).

The dashboard should load and immediately connect to your running backend, allowing you to make predictions.

## 🔮 Future Work

* Probability Forecasts: Integrate a trained LSTM model to provide probability-based forecasts over time.

* Database Integration: Replace the simulated data with a real database (like PostgreSQL) for storing and retrieving historical sensor readings.
