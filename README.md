# ⛏️ Rockfall Risk Classifier (Smart India Hackathon 2025)

An end-to-end ML pipeline — SMOTE → StandardScaler → XGBoost — that scores rockfall risk on a **synthetic** slope-stability dataset, served via FastAPI with a real-time React dashboard and Twilio SMS alerting on the highest risk tier. Built as the sole developer on a 3-person team, for **Smart India Hackathon 2025**.

## 🎯 Problem Statement

Demonstrate an AI-based rockfall risk-alerting pipeline end to end — data → model → API → dashboard → SMS — using synthetic slope-stability data, since no real-world sensor dataset was available to train on.

## 📊 Model & Data

The model is trained on `data/synthetic_slope_stability_dataset.csv` — **synthetically generated data**, not real sensor readings from a mine. It's used to demonstrate the pipeline (SMOTE → StandardScaler → XGBoost) end-to-end. `train_model.py` trains and evaluates the exact pipeline that gets saved to `rockfall_prediction_pipeline.joblib` and served by `main.py` — nothing else is trained or averaged in. It splits the data 60/20/20 (train/validation/test, stratified), picks a classification threshold on the validation split by maximizing F1, then reports metrics once on the held-out test split. Metrics (classification report, per-threshold precision/recall/F1, confusion matrix, ROC-AUC, PR-AUC, majority-class baseline, false-positive rate) are written to `results/metrics.json` on every run — see that file for actual numbers rather than assuming any. On the held-out test split, the served model scores **ROC-AUC 0.75**, against a majority-class (always-predict-rockfall) baseline of **64.75%**.

**The validation-chosen threshold is not wired into the API.** `main.py`'s risk levels use three independent, hand-set cutoffs on the raw probability (`THRESHOLD_GUARDED=0.45`, `THRESHOLD_ELEVATED=0.70`, `THRESHOLD_CRITICAL=0.95`, defined in `main.py`'s `Settings`), adjustable at runtime via `/set-thresholds`. `train_model.py`'s F1-optimal threshold is printed and saved to `results/metrics.json` for reference only — it is not read by `main.py` and has no code-level relationship to GUARDED/ELEVATED/CRITICAL.

### Evaluated at the API's actual risk cutoffs

`train_model.py` also evaluates the deployed model directly at `main.py`'s hand-set cutoffs (test split, 20,000 rows, 12,950 real rockfall events):

| Risk level | Cutoff | Rows ≥ cutoff | Share of test | Precision | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| GUARDED  | 0.45 | 11,849 | 59.2% | 0.783 | 0.717 | 0.364 |
| ELEVATED | 0.70 |  6,209 | 31.0% | 0.882 | 0.423 | 0.104 |
| CRITICAL | 0.95 |  2,286 | 11.4% | 0.983 | 0.174 | 0.006 |

**CRITICAL is the SMS trigger.** Of 20,000 test rows, 2,286 would send an SMS. 98.3% of those are real rockfall events (very few false alarms) — but that's only 17.4% of all 12,950 real rockfall events in the test set. In other words: when this model does send a CRITICAL SMS, it's almost always right, but it stays silent for roughly 5 out of 6 actual rockfall events. Full numbers, including raw TP/FP/TN/FN counts, are in `results/metrics.json` under `api_cutoffs`.

## 🚀 Key Features

- **⚡ Real-Time Prediction API** — FastAPI backend serving model predictions in milliseconds.
- **🧠 Machine Learning Core** — trained XGBoost model (with scaler) scoring rockfall probability on the synthetic slope-stability dataset.
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
