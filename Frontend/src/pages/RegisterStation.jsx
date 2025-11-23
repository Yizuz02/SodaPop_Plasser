import { useState } from "react";
import Navbar from "../components/Navbar";
import "./AdminPanel.css";

export default function RegisterStation() {
  const [name, setName] = useState("");
  const [city, setCity] = useState("");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");

  const token = sessionStorage.getItem("authToken");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!name || !city || !latitude || !longitude) {
      alert("All fields are required");
      return;
    }

    if (!token) {
      alert("No auth token found. Please login.");
      return;
    }

    const newStation = {
      name: name.trim(),
      city: city.trim(),
      latitude: parseFloat(latitude),
      longitude: parseFloat(longitude),
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/stations/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Token ${token}`,
        },
        body: JSON.stringify(newStation),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(JSON.stringify(errorData));
      }

      // Limpiar formulario
      setName("");
      setCity("");
      setLatitude("");
      setLongitude("");

      alert("Station registered successfully!");

    } catch (err) {
      console.error("Error registering station:", err.message);
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
        <h3>Register Station</h3>
        <form onSubmit={handleSubmit}>
          <input
            placeholder="Station Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <input
            placeholder="City"
            value={city}
            onChange={(e) => setCity(e.target.value)}
          />
          <input
            placeholder="Latitude"
            type="number"
            step="0.0001"
            value={latitude}
            onChange={(e) => setLatitude(e.target.value)}
          />
          <input
            placeholder="Longitude"
            type="number"
            step="0.0001"
            value={longitude}
            onChange={(e) => setLongitude(e.target.value)}
          />
          <button type="submit" className="btn-add">
            Save Station
          </button>
        </form>
      </div>
    </div>
  );
}
