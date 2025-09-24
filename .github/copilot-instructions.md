
# Copilot Instructions for AI Coding Agents

## 🗺️ Project Overview

**Purpose:** Predict and prevent rockfall incidents in open-pit mines using:
- XGBoost ML model
- FastAPI backend
- Simple HTML dashboard

**Major Components:**
- `main.py`: FastAPI server (prediction API, Twilio alerting)
- `train_model.py`: Trains/exports XGBoost model (`rockfall_predictor_xgb.json`)
- `frontend/dashboard.html`: User dashboard (inputs, displays predictions)
- `data/`: Datasets for training/testing (e.g., `synthetic_slope_stability_dataset.csv`)
- `requirements.txt`: Python dependencies

## 🔄 Architecture & Data Flow

**End-to-end flow:**
1. User enters slope parameters in `dashboard.html`
2. Dashboard sends request to FastAPI backend (`main.py`)
3. Backend loads XGBoost model, predicts risk, returns result
4. If risk > threshold, Twilio SMS alert may be sent

**Model lifecycle:**
- Train offline with `train_model.py` → saves `rockfall_predictor_xgb.json`
- Backend loads model at runtime

## 🛠️ Developer Workflows

**Backend:**
- Start server: `uvicorn main:app --reload`
- Install deps: `pip install -r requirements.txt`
- Retrain model: Run `train_model.py` (updates `rockfall_predictor_xgb.json`)
- Twilio: Set credentials in `main.py` (not env vars)

**Frontend:**
- Open `frontend/dashboard.html` in browser (no build step)
- Expects backend at FastAPI default port (8000)

**Data:**
- Add new datasets to `data/`
- Update `train_model.py` if schema changes

## 📏 Project-Specific Conventions

- No JS frontend framework (pure HTML/JS, no React/Vue)
- Model file always `rockfall_predictor_xgb.json` in project root
- Twilio credentials set directly in code
- No Docker/cloud deployment scripts (local dev only)

## 🔗 Integration Points

- **Twilio**: SMS alerts, manual credential entry in `main.py`
- **XGBoost**: Model file must exist before backend serves predictions

## 🧩 Examples & Patterns

- See `dashboard.html` for `/predict` API usage
- `main.py` loads model at startup; retrain with `train_model.py` as needed

## 🗝️ Key Files

- `main.py`
- `train_model.py`
- `frontend/dashboard.html`
- `requirements.txt`
- `data/`
- `rockfall_predictor_xgb.json`

---

**Update this file with new conventions, workflows, or integration points when adding features.**
