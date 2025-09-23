// src/PredictionForm.jsx

import { useState } from 'react';

function PredictionForm({ onPrediction }) {
  // 1. Update the state to use the new feature names
  const [features, setFeatures] = useState({
    slope_angle: 40.0,
    rainfall_last_24h: 100.0,
    displacement_rate: 2.0,
    pore_pressure: 150.0,
    image_crack_score: 0.5,
  });

  const handleSliderChange = (e) => {
    const { name, value } = e.target;
    setFeatures(prevFeatures => ({
      ...prevFeatures,
      [name]: parseFloat(value),
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    fetch("http://127.0.0.1:8000/predict", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(features),
    })
    .then(response => response.json())
    .then(data => {
      onPrediction(data);
    })
    .catch(error => {
      console.error("Error making prediction:", error);
      onPrediction({ error: "Failed to get prediction." });
    });
  };

  return (
    <form onSubmit={handleSubmit} className="prediction-form">
      <h3>Input Parameters</h3>
      <div className="form-grid">
        {/* 2. The form now automatically creates sliders based on the new state */}
        {Object.keys(features).map(key => (
          <div key={key} className="form-group">
            <label>{key.replace(/_/g, ' ')}: {features[key]}</label>
            <input
              type="range"
              name={key}
              value={features[key]}
              onChange={handleSliderChange}
              // You might want to adjust min/max/step for the new data ranges
              min={0}
              max={key === 'pore_pressure' ? 300 : (key === 'rainfall_last_24h' ? 200 : 60)}
              step={0.1}
            />
          </div>
        ))}
      </div>
      <button type="submit" className="predict-button">Predict Risk</button>
    </form>
  );
}

export default PredictionForm;
