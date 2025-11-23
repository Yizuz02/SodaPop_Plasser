import { useState, useEffect } from "react";
import Navbar from "../components/Navbar";
import "./AdminPanel.css";

export default function RegisterTrainTrip() {
  const [trainId, setTrainId] = useState("");
  const [routeId, setRouteId] = useState("");
  const [cargoWeight, setCargoWeight] = useState("");
  const [averageSpeed, setAverageSpeed] = useState("");
  const [departureTime, setDepartureTime] = useState("");
  const [arrivalTime, setArrivalTime] = useState("");

  const [trains, setTrains] = useState([]);
  const [routes, setRoutes] = useState([]);

  const token = sessionStorage.getItem("authToken");

  // === Fetch trains ===
  useEffect(() => {
    const fetchTrains = async () => {
      if (!token) return;

      try {
        const response = await fetch("http://127.0.0.1:8000/api/trains/", {
          headers: {
            "Authorization": `Token ${token}`,
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) throw new Error("Failed to fetch trains");

        const data = await response.json();
        setTrains(data); // setTrains con los datos del backend
      } catch (err) {
        console.error("Error fetching trains:", err.message);
      }
    };

    fetchTrains();
  }, [token]); // Se ejecuta cuando token cambia o se monta

  // === Fetch routes ===
  useEffect(() => {
    const fetchRoutes = async () => {
      if (!token) return;

      try {
        const response = await fetch("http://127.0.0.1:8000/api/routes/", {
          headers: {
            "Authorization": `Token ${token}`,
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) throw new Error("Failed to fetch routes");

        const data = await response.json();
        setRoutes(data); // setRoutes con los datos del backend
      } catch (err) {
        console.error("Error fetching routes:", err.message);
      }
    };

    fetchRoutes();
  }, [token]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!trainId || !routeId || !cargoWeight || !averageSpeed || !departureTime || !arrivalTime) {
      alert("All fields are required");
      return;
    }

    const newTrip = {
      train: parseInt(trainId),
      route: parseInt(routeId),
      cargo_weight: parseFloat(cargoWeight),
      average_speed: parseFloat(averageSpeed),
      departure_time: new Date(departureTime).toISOString(),
      arrival_time: new Date(arrivalTime).toISOString(),
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/train-trips/", {
        method: "POST",
        headers: {
          "Authorization": `Token ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newTrip),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(JSON.stringify(errorData));
      }

      alert("Train trip registered successfully!");
      setTrainId("");
      setRouteId("");
      setCargoWeight("");
      setAverageSpeed("");
      setDepartureTime("");
      setArrivalTime("");
    } catch (err) {
      console.error("Error registering train trip:", err.message);
      alert("Error: " + err.message);
    }
  };

  return (
    <div className="admin-container">
      <div className="admin-header">
        <h1 className="admin-title">Admin Panel</h1>
      </div>

      <Navbar />

      <div className="form-box">
        <h3>Register Train Trip</h3>
        <form onSubmit={handleSubmit}>
          {/* Select Trains */}
          <select value={trainId} onChange={(e) => setTrainId(e.target.value)}>
            <option value="">Select Train</option>
            {trains.map((train) => (
              <option key={train.id} value={train.id}>
                {train.model} ({train.train_type})
              </option>
            ))}
          </select>

          {/* Select Routes */}
          <select value={routeId} onChange={(e) => setRouteId(e.target.value)}>
            <option value="">Select Route</option>
            {routes.map((route) => (
              <option key={route.id} value={route.id}>
                {route.route_code}: {route.origin.name} → {route.destination.name}
              </option>
            ))}
          </select>

          <input
            placeholder="Cargo Weight"
            type="number"
            value={cargoWeight}
            onChange={(e) => setCargoWeight(e.target.value)}
          />
          <input
            placeholder="Average Speed"
            type="number"
            value={averageSpeed}
            onChange={(e) => setAverageSpeed(e.target.value)}
          />
          <input
            placeholder="Departure Time"
            type="datetime-local"
            value={departureTime}
            onChange={(e) => setDepartureTime(e.target.value)}
          />
          <input
            placeholder="Arrival Time"
            type="datetime-local"
            value={arrivalTime}
            onChange={(e) => setArrivalTime(e.target.value)}
          />

          <button type="submit" className="btn-add">
            Save Train Trip
          </button>
        </form>
      </div>
    </div>
  );
}
