// src/App.jsx

// 1. Import useState and useEffect from React
import { useState, useEffect } from 'react';
import './App.css'; // You can keep your default CSS

function App() {
  // 2. Create a state variable to hold the message from the API
  const [message, setMessage] = useState("Loading...");

  // 3. useEffect runs once when the component loads
  useEffect(() => {
    // 4. Fetch data from your FastAPI backend's root endpoint
    fetch("http://127.0.0.1:8000/")
      .then(response => response.json())
      .then(data => {
        // 5. Update the message state with the data from the API
        setMessage(data.message);
      })
      .catch(error => {
        console.error("Error fetching data:", error);
        setMessage("Failed to connect to the API.");
      });
  }, []); // The empty array [] means this effect runs only once

  return (
    <div className="container">
      <h1>⛏️ Rockfall Prediction Dashboard</h1>

      {/* 6. Display the message from the API */}
      <h2>API Connection Status:</h2>
      <p>{message}</p>
    </div>
  );
}

export default App;
