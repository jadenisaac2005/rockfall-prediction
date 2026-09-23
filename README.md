# ⛏️ AI-Powered Rockfall Prediction System

An intelligent, end-to-end system designed to predict and prevent rockfall incidents in open-pit mines. Built for the **Smart India Hackathon**, combining a machine learning backend with a real-time, interactive web dashboard.

## 🎯 Problem Statement

AI-Based Rockfall Prediction and Alert System for Open-Pit Mines

## 📊 Model & Data

The model is trained on `data/synthetic_slope_stability_dataset.csv` — **synthetically generated data**, not real sensor readings from a mine. It's used to demonstrate the pipeline (SMOTE → StandardScaler → XGBoost/RandomForest ensemble → threshold search) end-to-end. Evaluation metrics (classification report, per-threshold precision/recall/F1, confusion matrix) are computed by `train_model.py` on every run and written to `results/metrics.json` — see that file for actual numbers rather than assuming any.

## 🚀 Key Features

- **⚡ Real-Time Prediction API** — FastAPI backend serving model predictions in milliseconds.
- **🧠 Machine Learning Core** — trained XGBoost model (with scaler) analyzing multi-source data to calculate rockfall probabilities.
- **🗺️ Interactive Dashboard** — single-page React (CDN) app for "what-if" analysis, live predictions, and risk visualization.
- **🎚️ Dynamic Risk Thresholds** — adjustable Guarded/Elevated/Critical thresholds, updated on the backend in real time.
- **📲 Automated Alert System** — background worker sends SMS alerts via Twilio when a critical threshold is breached.
- **🎨 Color-Coded Risk Display** — green/yellow/orange/red risk levels for instant clarity.

## 🛠️ Technology Stack

| Area | Technologies |
| --- | --- |
| **Backend** | Python, FastAPI, Uvicorn, joblib, pydantic-settings |
| **AI Model** | XGBoost, Pandas, Scikit-learn |
| **Frontend** | HTML, CSS, React (CDN), Babel |
| **Alerting** | Twilio |
| **Environment** | `venv` for Python, `.env` for secrets |

## 🏃 How to Run

### Prerequisites

- Python 3.11 (managed via `pyenv` is recommended)
- Git

### 1. Backend Setup & Launch

```bash
git clone https://github.com/jadenisaac2005/rockfall-prediction.git
cd rockfall-prediction

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

The placeholder values in `.env.example` are enough to run the app locally — the backend boots and `/predict` works with the demo model. SMS alerting is simply skipped until you fill in real Twilio credentials and phone numbers.

Launch the FastAPI server:

```bash
uvicorn main:app --reload
```

(Optional) Retrain the model and regenerate evaluation metrics:

```bash
python train_model.py
```

This overwrites `rockfall_prediction_pipeline.joblib` and writes evaluation metrics (classification report, per-threshold precision/recall/F1, confusion matrix) to `results/metrics.json`.

### 2. Frontend Dashboard

Open `frontend/dashboard.html` directly in your browser (no build step required). The dashboard connects to the backend at `http://127.0.0.1:8000` by default.

### 3. Dynamic Thresholds

Use the **Risk Threshold Settings** panel in the dashboard to adjust Guarded, Elevated, and Critical thresholds. Click the reload icon to reset thresholds to their default values. Changes are sent to the backend and take effect immediately for all predictions.

## 📏 Project-Specific Conventions

- No JS frontend framework build step — React via CDN, single HTML file
- Model pipeline file: `rockfall_prediction_pipeline.joblib` in project root
- All sensitive info in `.env` (never committed)
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
- `data/synthetic_slope_stability_dataset.csv` — synthetic training data
- `results/metrics.json` — evaluation metrics from the last `train_model.py` run
- `.env` (create from `.env.example`, not committed)

## 🔮 Future Work

- Probability Forecasts: integrate a trained LSTM model to provide probability-based forecasts over time.
- Database Integration: replace the simulated data with a real database (like PostgreSQL) for storing and retrieving historical sensor readings.
