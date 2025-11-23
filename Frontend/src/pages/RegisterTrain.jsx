import { useState } from "react";
import Navbar from "../components/Navbar";
import "./AdminPanel.css";

export default function RegisterTrain() {
  const [model, setModel] = useState("");
  const [weight, setWeight] = useState("");
  const [maxSpeed, setMaxSpeed] = useState("");
  const [trainType, setTrainType] = useState("passenger");
  const [manufacturer, setManufacturer] = useState("");
  const [yearBuilt, setYearBuilt] = useState("");

  const token = sessionStorage.getItem("authToken");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!model || !weight || !maxSpeed || !trainType || !manufacturer || !yearBuilt) {
      alert("All fields are required");
      return;
    }

    if (!token) {
      alert("No auth token found. Please login.");
      return;
    }

    const newTrain = {
      model: model.trim(),
      weight: parseInt(weight),
      max_speed: parseInt(maxSpeed),
      train_type: trainType,
      manufacturer: manufacturer.trim(),
      year_built: parseInt(yearBuilt),
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/trains/", {
        method: "POST",
        headers: {
          "Authorization": `Token ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newTrain),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(JSON.stringify(errorData));
      }

      // Limpiar formulario
      setModel("");
      setWeight("");
      setMaxSpeed("");
      setTrainType("passenger");
      setManufacturer("");
      setYearBuilt("");

      alert("Train registered successfully!");
    } catch (err) {
      console.error("Error registering train:", err.message);
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
        <h3>Register Train</h3>
        <form onSubmit={handleSubmit}>
          <input
            placeholder="Model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
          <input
            placeholder="Weight"
            type="number"
            value={weight}
            onChange={(e) => setWeight(e.target.value)}
          />
          <input
            placeholder="Max Speed"
            type="number"
            value={maxSpeed}
            onChange={(e) => setMaxSpeed(e.target.value)}
          />
          <select value={trainType} onChange={(e) => setTrainType(e.target.value)}>
            <option value="passenger">Passenger</option>
            <option value="cargo">Cargo</option>
            <option value="highspeed">High-Speed</option>
          </select>
          <input
            placeholder="Manufacturer"
            value={manufacturer}
            onChange={(e) => setManufacturer(e.target.value)}
          />
          <input
            placeholder="Year Built"
            type="number"
            value={yearBuilt}
            onChange={(e) => setYearBuilt(e.target.value)}
          />
          <button type="submit" className="btn-add">
            Save Train
          </button>
        </form>
      </div>
    </div>
  );
}
