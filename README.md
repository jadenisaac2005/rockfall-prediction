# ⛏️ AI-Powered Rockfall Prediction System

An intelligent, end-to-end system designed to predict and prevent rockfall incidents in open-pit mines. This project was developed for the **Smart India Hackathon** and combines a powerful machine learning backend with a real-time, interactive web dashboard to create a practical and life-saving tool.

## ✨ Live Dashboard Preview

Here is a preview of the interactive dashboard that provides mine planners with at-a-glance risk assessments.

*(**Action required:** Please replace the line above with a screenshot of your `dashboard.html` in action!)*

## 🎯 Problem Statement

Rockfalls in open-pit mines pose a significant threat to personnel safety and can cause costly operational disruptions. This project addresses the need for a proactive monitoring system that can identify patterns preceding rockfall events and provide timely warnings for preventive action.

## 🚀 Key Features

* **⚡ Real-Time Prediction API:** A high-performance backend built with **FastAPI** that serves AI model predictions in milliseconds.
* **🧠 Machine Learning Core:** Utilizes a trained **XGBoost** model to analyze multi-source data and calculate rockfall probabilities.
* **🗺️ Interactive Dashboard:** A single-page web application built with **React** that allows users to perform "what-if" analysis using input sliders and see live predictions.
* **📍 Live Risk Map:** An integrated **Leaflet.js** map that displays real-time, color-coded risk levels for different zones across the mine site.
* **📲 Automated Alert System:** A background worker script that continuously monitors high-risk zones and can send SMS alerts via **Twilio** when a critical threshold is breached.

## 🛠️ Technology Stack

| Area          | Technologies                                       |
| ------------- | -------------------------------------------------- |
| **Backend** | Python, FastAPI, Uvicorn                           |
| **AI Model** | XGBoost, Pandas, Scikit-learn                      |
| **Frontend** | HTML, CSS, JavaScript, React (via CDN), Babel      |
| **Mapping** | Leaflet.js                                         |
| **Alerting** | Twilio                                             |
| **Environment** | `venv` for Python, `pyenv` for version management  |

## 🏃‍♀️ How to Run the Project

This project consists of three parts: the **Backend API**, the **Frontend Dashboard**, and the optional **Alert Worker**.

### Prerequisites

* Python 3.11 (managed via `pyenv` is recommended)
* Git

### 1. Backend Setup & Launch

First, get the AI server running.

```bash
# Clone the repository
git clone <your-repository-url>
cd <your-repo-name>

# Set up and activate the Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate

# Install all required Python packages
pip install -r requirements.txt

# Launch the FastAPI server
uvicorn main:app --reload
```
Leave this terminal running. The API is now live at `http://127.0.0.1:8000`.

### 2. Frontend Launch
The frontend is a self-contained HTML file.

1. Navigate to the project folder in your file explorer.

2. Open the `dashboard.html` file directly in your web browser (like Chrome or Firefox).

The dashboard should load and immediately connect to your running backend, populating the map and allowing you to make predictions.

### 3. (Optional) Run the Alert Worker
To run the automated SMS alert system:

1. Open a new terminal and navigate to the project directory.

2. Activate the virtual environment: source `venv/bin/activate`.

3. Edit the `alert_worker.py` file and add your Twilio credentials.

4. Run the script:
```bash
python alert_worker.py
```

## 🔮 Future Work

* Probability Forecasts: Integrate a trained LSTM model to provide probability-based forecasts over time.

* Database Integration: Replace the simulated data with a real database (like PostgreSQL) for storing and retrieving historical sensor readings.


