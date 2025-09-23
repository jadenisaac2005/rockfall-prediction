// src/App.jsx

import { useState } from 'react';
import PredictionForm from './PredictionForm';
import RiskMap from './RiskMap'; // 1. IMPORT THE MAP
import './App.css';

function App() {
  const [predictionResult, setPredictionResult] = useState(null);

  const handlePrediction = (result) => {
    setPredictionResult(result);
  };

  return (
    <div className="container">
      <h1>⛏️ Rockfall Prediction Dashboard</h1>

      {/* 2. RENDER THE MAP <RiskMap />*/}


      <div className="form-and-result">
          {/* Your PredictionForm and result display can go here */}
          <PredictionForm onPrediction={handlePrediction} />
          {predictionResult && (
            <div className="result-container">
              {/* ... result display JSX ... */}
            </div>
          )}
      </div>
    </div>
  );
}

export default App;
