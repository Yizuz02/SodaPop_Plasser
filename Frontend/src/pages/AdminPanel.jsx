import { useState, useEffect } from "react";
import "./AdminPanel.css";
import Navbar from "../components/Navbar";

export default function AdminPanel({ onLogout }) {
  const [name, setName] = useState("");
  const [lastname, setLastname] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const [operators, setOperators] = useState([]);

  const token = sessionStorage.getItem("authToken");

  // === FETCH EXISTING OPERATORS FROM BACKEND ===
  useEffect(() => {
    const fetchOperators = async () => {
      if (!token) return;

      try {
        const response = await fetch("http://127.0.0.1:8000/api/users/", {
          headers: {
            "Authorization": `Token ${token}`,
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) throw new Error("Failed to fetch operators");

        const data = await response.json();
        setOperators(data);
      } catch (err) {
        console.error(err.message);
      }
    };

    fetchOperators();
  }, [token]);

  // === ADD NEW OPERATOR ===
  const handleAdd = async (e) => {
    e.preventDefault();

    if (!name.trim() || !lastname.trim() || !username.trim() || !password.trim()) {
      alert("All fields are required");
      return;
    }

    if (!token) {
      alert("No auth token found. Please login again.");
      return;
    }

    const newOperator = {
      username: username.trim(),
      password: password.trim(),
      first_name: name.trim(),
      last_name: lastname.trim(),
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/users/", {
        method: "POST",
        headers: {
          "Authorization": `Token ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newOperator),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(JSON.stringify(errorData));
      }

      const createdUser = await response.json();

      // Actualizar la lista de operadores
      setOperators([...operators, createdUser]);

      // Limpiar formulario
      setName("");
      setLastname("");
      setUsername("");
      setPassword("");

    } catch (err) {
      console.error("Failed to add operator:", err.message);
      alert("Error adding operator: " + err.message);
    }
  };

  return (
    <div className="admin-container">
      <div className="admin-header">
        <h1 className="admin-title">Admin Panel</h1>
        <button className="logout-btn" onClick={onLogout}>
          Log Out
        </button>
      </div>
      <Navbar />

      {/* Form to add operator */}
      <form className="admin-form" onSubmit={handleAdd}>
        <input
          type="text"
          placeholder="First Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <input
          type="text"
          placeholder="Last Name"
          value={lastname}
          onChange={(e) => setLastname(e.target.value)}
        />
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit" className="add-btn">
          Add Operator
        </button>
      </form>

      {/* Operators list */}
      <h2 className="list-title">Registered Operators</h2>
      <ul className="operator-list">
        {operators.length === 0 && <p className="no-ops">No operators registered yet.</p>}

        {operators.map((op, index) => (
          <li key={index} className="operator-item">
            <strong>{op.first_name} {op.last_name}</strong> — {op.username}
          </li>
        ))}
      </ul>
    </div>
  );
}
