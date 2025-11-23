import { useState } from "react";
import Navbar from "../components/Navbar";
import "./AdminPanel.css";

export default function RegisterMachine() {
  const [model, setModel] = useState("");
  const [yearBuilt, setYearBuilt] = useState("");
  const [weight, setWeight] = useState("");
  const [maxSpeed, setMaxSpeed] = useState("");
  const [workingSpeed, setWorkingSpeed] = useState("");
  const [tamperType, setTamperType] = useState("");

  const token = sessionStorage.getItem("authToken");

  // Opciones válidas de tamper_type
  const tamperOptions = [
    { value: "unimat_4s", label: "Unimat 4S (Universal)" },
    { value: "unimat_08_475", label: "Unimat 08-475/4S (Desvíos)" },
    { value: "09_3x_dyn", label: "09-3X Dynamic (Alto Rendimiento)" },
    { value: "09_32_csm", label: "09-32 CSM (Línea Continua)" },
    { value: "dynamic_9000", label: "Dynamic 9000 (Estabilización/Tampeo)" },
    { value: "pms_2030", label: "PMS 2030 (Plasser Measuring System)" },
    { value: "duomatic_09", label: "Duomatic 09 (Doble Bateadora)" },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!model || !yearBuilt || !weight || !maxSpeed || !workingSpeed || !tamperType) {
      alert("All fields are required");
      return;
    }

    if (!token) {
      alert("No auth token found. Please login.");
      return;
    }

    const newMachine = {
      model,
      manufacturer: "Plasser & Theurer",
      year_built: parseInt(yearBuilt),
      weight: parseFloat(weight),
      max_speed: parseFloat(maxSpeed),
      working_speed: parseFloat(workingSpeed),
      tamper_type: tamperType, // Opción seleccionada
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/tamper-machines/", {
        method: "POST",
        headers: {
          "Authorization": `Token ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newMachine),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(JSON.stringify(errorData));
      }

      // Limpiar formulario
      setModel("");
      setYearBuilt("");
      setWeight("");
      setMaxSpeed("");
      setWorkingSpeed("");
      setTamperType("");

      alert("Machine registered successfully!");
    } catch (err) {
      console.error("Error registering machine:", err.message);
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
        <h3>Register Machine</h3>
        <form onSubmit={handleSubmit}>
          {/* Modelo libre */}
          <input
            placeholder="Model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />

          {/* Tamper Type como select */}
          <select value={tamperType} onChange={(e) => setTamperType(e.target.value)}>
            <option value="">Select Tamper Type</option>
            {tamperOptions.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>

          <input
            placeholder="Year Built"
            type="number"
            value={yearBuilt}
            onChange={(e) => setYearBuilt(e.target.value)}
          />
          <input
            placeholder="Weight"
            type="number"
            step="0.1"
            value={weight}
            onChange={(e) => setWeight(e.target.value)}
          />
          <input
            placeholder="Max Speed"
            type="number"
            step="0.1"
            value={maxSpeed}
            onChange={(e) => setMaxSpeed(e.target.value)}
          />
          <input
            placeholder="Working Speed"
            type="number"
            step="0.1"
            value={workingSpeed}
            onChange={(e) => setWorkingSpeed(e.target.value)}
          />

          <button type="submit" className="btn-add">
            Save Machine
          </button>
        </form>
      </div>
    </div>
  );
}
