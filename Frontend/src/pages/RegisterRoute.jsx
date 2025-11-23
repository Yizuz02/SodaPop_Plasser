import { useState, useEffect } from "react";
import Navbar from "../components/Navbar";
import "./AdminPanel.css";

export default function RegisterRoute() {
  const [routeCode, setRouteCode] = useState("");
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [stations, setStations] = useState([]);

  const token = sessionStorage.getItem("authToken");

  // === Fetch stations from backend ===
  useEffect(() => {
    const fetchStations = async () => {
      if (!token) return;

      try {
        const response = await fetch("http://127.0.0.1:8000/api/stations/", {
          headers: {
            "Authorization": `Token ${token}`,
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) throw new Error("Failed to fetch stations");

        const data = await response.json();
        setStations(data);
      } catch (err) {
        console.error(err.message);
      }
    };

    fetchStations();
  }, [token]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!routeCode || !origin || !destination) {
      alert("All fields are required");
      return;
    }

    if (!token) {
      alert("No auth token found. Please login.");
      return;
    }

    const newRoute = {
      route_code: routeCode.trim(),
      origin: parseInt(origin),
      destination: parseInt(destination),
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/routes/", {
        method: "POST",
        headers: {
          "Authorization": `Token ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newRoute),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(JSON.stringify(errorData));
      }

      // Limpiar formulario
      setRouteCode("");
      setOrigin("");
      setDestination("");

      alert("Route registered successfully!");
    } catch (err) {
      console.error("Error registering route:", err.message);
      alert("Error: " + err.message);
    }
  };

  return (
    <div className="admin-container">
      {/* Header */}
      <div className="admin-header">
        <h1 className="admin-title">Admin Panel</h1>
      </div>

      {/* Navbar */}
      <Navbar />

      {/* Form */}
      <div className="form-box">
        <h3>Register Route</h3>
        <form onSubmit={handleSubmit}>
          <input
            placeholder="Route Code"
            value={routeCode}
            onChange={(e) => setRouteCode(e.target.value)}
          />

          {/* Origin */}
          <select value={origin} onChange={(e) => setOrigin(e.target.value)}>
            <option value="">Select Origin Station</option>
            {stations.map((station) => (
              <option key={station.id} value={station.id}>
                {station.name} — {station.city}
              </option>
            ))}
          </select>

          {/* Destination */}
          <select value={destination} onChange={(e) => setDestination(e.target.value)}>
            <option value="">Select Destination Station</option>
            {stations.map((station) => (
              <option key={station.id} value={station.id}>
                {station.name} — {station.city}
              </option>
            ))}
          </select>

          <button type="submit" className="btn-add">
            Save Route
          </button>
        </form>
      </div>
    </div>
  );
}
