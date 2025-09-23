// src/RiskMap.jsx

import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css'; // Import leaflet's CSS
import L from 'leaflet'; // Import leaflet library

// A function to get the right color for the map marker based on risk
const getRiskColor = (probability) => {
  if (probability > 0.75) return 'red';
  if (probability > 0.5) return 'orange';
  return 'green';
};

// Create custom icons
const createCustomIcon = (color) => {
  return new L.Icon({
    iconUrl: `https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-${color}.png`,
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
  });
};

function RiskMap() {
  const [mapData, setMapData] = useState([]);

  useEffect(() => {
    const fetchData = () => {
      fetch("http://127.0.0.1:8000/risk-map")
        .then(res => res.json())
        .then(data => setMapData(data))
        .catch(err => console.error("Error fetching map data:", err));
    };

    fetchData(); // Fetch immediately on load
    const interval = setInterval(fetchData, 5000); // Then fetch every 5 seconds

    return () => clearInterval(interval); // Cleanup on component unmount
  }, []);

  return (
    <div className="map-container">
      <h3>Real-Time Risk Map</h3>
      <MapContainer center={[12.9716, 77.5946]} zoom={14} style={{ height: '500px', width: '100%' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {mapData.map(zone => (
          <Marker
            key={zone.zone_id}
            position={zone.coords}
            icon={createCustomIcon(getRiskColor(zone.probability))}
          >
            <Popup>
              <b>{zone.name}</b><br />
              Risk Probability: {(zone.probability * 100).toFixed(2)}%
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}

export default RiskMap;
