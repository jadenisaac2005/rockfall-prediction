# ⛏️ AI-Powered Rockfall Prediction System

An intelligent system designed to predict and prevent rockfall incidents in open-pit mines, developed for the **Smart India Hackathon 2025**. Our solution leverages machine learning to analyze multi-source data and provide real-time risk assessments to ensure personnel safety and operational continuity.

---

## ## Problem Statement

Rockfalls in open-pit mines pose a significant threat to both personnel and equipment. Traditional detection methods are often reactive, labor-intensive, or lack real-time predictive capabilities. This project aims to create a smart, cost-effective, and scalable AI-based system capable of predicting potential rockfall incidents before they occur.

---

## ## Our Solution

We have developed a complete end-to-end system that ingests multi-source data, processes it, and serves predictions via a high-speed API. The system is designed to provide mine planners with a user-friendly dashboard for proactive decision-making.

The core of our solution is a machine learning model (XGBoost) trained to identify complex patterns that precede rockfall events. This model is served by a robust FastAPI backend, ready to be integrated with any modern frontend or hardware system.

---

## ## Key Features

* **Real-Time Prediction API:** A high-performance FastAPI endpoint that delivers rockfall probability scores in milliseconds.
* **User-Friendly Dashboard:** An interactive web interface (built with React) for visualizing risks and forecasts.
* **Interactive Risk Maps:** A color-coded map of the mine site, updated in real-time to highlight high-risk zones.
* **Automated Alert System:** Proactive SMS and email notifications are sent to key personnel when risk levels exceed a critical threshold.
* **Hardware Integration Ready:** The system is built with ingestion endpoints capable of receiving data from low-cost IoT monitoring hardware.

---

## ## Technology Stack

* **Backend:** 🐍 Python, FastAPI
* **Machine Learning:** 🤖 Scikit-learn, XGBoost, Pandas
* **Frontend:** ⚛️ React, Leaflet (for maps), Chart.js (for charts)
* **Alerts:** twilio (for SMS), SendGrid (for Email)
* **Deployment:** Docker, Uvicorn

---

## ## Getting Started

### ### Prerequisites

* Python 3.10+
* Git
* A virtual environment manager (`venv`)

### ### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Backend Server:**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

5.  **Run the Frontend (Instructions to be added):**
    ```bash
    # cd frontend
    # npm install
    # npm start
    ```

---

## ## Team Members

* [Your Name]
* [Teammate's Name]
* [Teammate's Name]
* [Teammate's Name]
* [Teammate's Name]
