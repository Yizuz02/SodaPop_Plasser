import { useState } from "react";
import { Link } from "react-router-dom";

export default function AdminPanel({ onLogout }) {
  const [name, setName] = useState("");
  const [lastname, setLastname] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const [operators, setOperators] = useState(() => {
    // Evitar error si localStorage no está disponible (SSR o primera carga)
    if (typeof window !== "undefined") {
        return JSON.parse(localStorage.getItem("operators")) || [];
    }
    return [];
  });

  // === SAVE TO LOCALSTORAGE ===
  const saveOperators = (list) => {
    localStorage.setItem("operators", JSON.stringify(list));
    setOperators(list);
  };

  // === ADD NEW OPERATOR ===
  const handleAdd = (e) => {
    e.preventDefault();

    if (
      !name.trim() ||
      !lastname.trim() ||
      !username.trim() ||
      !password.trim()
    ) {
      alert("All fields are required");
      return;
    }

    const newOperator = { name, lastname, username, password };

    saveOperators([...operators, newOperator]);

    setName("");
    setLastname("");
    setUsername("");
    setPassword("");
  };

  return (
    <div className="admin-container">
      {/* Estilos embebidos para evitar error de importación */}
      <style>{`
        /* Contenedor principal */
        .admin-container {
          min-height: 100vh;
          background-color: #1a1a1a; /* Fondo oscuro */
          color: #fff;
          padding: 20px;
          font-family: Arial, sans-serif;
        }

        /* --- NUEVA BARRA DE NAVEGACIÓN --- */
        .admin-nav {
          background-color: #333;
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 30px;
          display: flex;
          align-items: center;
          border-bottom: 2px solid #ffcc00; /* Acento amarillo */
        }

        .nav-label {
          font-weight: bold;
          color: #ffcc00;
          margin-right: 15px;
          text-transform: uppercase;
          font-size: 0.9rem;
        }

        .nav-links {
          display: flex;
          gap: 15px;
          flex-wrap: wrap;
        }

        .nav-item {
          text-decoration: none;
          color: #fff;
          background-color: #444;
          padding: 8px 16px;
          border-radius: 4px;
          transition: all 0.3s ease;
          font-size: 0.9rem;
        }

        .nav-item:hover {
          background-color: #ffcc00;
          color: #000;
          transform: translateY(-2px);
        }

        /* Header existente */
        .admin-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
          padding-bottom: 10px;
          border-bottom: 1px solid #444;
        }

        .admin-title {
          color: #ffcc00;
          margin: 0;
        }

        .logout-btn {
          background-color: #d9534f;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 5px;
          cursor: pointer;
          font-weight: bold;
        }

        .logout-btn:hover {
          background-color: #c9302c;
        }

        /* Formulario */
        .admin-form {
          background-color: #2a2a2a;
          padding: 25px;
          border-radius: 8px;
          max-width: 500px;
          margin: 0 auto 40px auto;
          box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }

        .form-subtitle {
          margin-top: 0;
          color: #ddd;
          text-align: center;
          margin-bottom: 20px;
        }

        .admin-form input {
          width: 100%;
          padding: 12px;
          margin-bottom: 15px;
          background-color: #333;
          border: 1px solid #444;
          border-radius: 4px;
          color: white;
          box-sizing: border-box; 
        }

        .add-btn {
          width: 100%;
          padding: 12px;
          background-color: #ffcc00;
          color: #000;
          border: none;
          border-radius: 4px;
          font-weight: bold;
          cursor: pointer;
          transition: background 0.3s;
        }

        .add-btn:hover {
          background-color: #e6b800;
        }

        /* Lista */
        .list-title {
          text-align: center;
          color: #ffcc00;
          margin-bottom: 20px;
        }

        .operator-list {
          list-style: none;
          padding: 0;
          max-width: 600px;
          margin: 0 auto;
        }

        .operator-item {
          background-color: #333;
          margin-bottom: 10px;
          padding: 15px;
          border-radius: 4px;
          border-left: 4px solid #ffcc00;
          display: flex;
          justify-content: space-between;
        }

        .op-username {
          color: #aaa;
          font-style: italic;
        }

        .no-ops {
          text-align: center;
          color: #777;
        }
      `}</style>
      
      {/* --- NUEVA BARRA DE NAVEGACIÓN --- */}
      <nav className="admin-nav">
        <span className="nav-label">Registros:</span>
        <div className="nav-links">
        <div className="nav-links">
        {/* CORREGIDO: Debe apuntar a la URL configurada en App.jsx */}
        <Link to="/admin" className="nav-item">Panel Principal</Link> 
        {/* Estas rutas ya estaban bien si las definiste en App.jsx: */}
        <Link to="/admin/trains" className="nav-item">Train</Link>
        <Link to="/admin/machines" className="nav-item">Machine</Link>
        <Link to="/admin/traintrips" className="nav-item">Train Trip</Link>
</div>
        </div>
      </nav>
      {/* ------------------------------- */}

      {/* Header with logout */}
      <div className="admin-header">
        <h1 className="admin-title">Admin Panel</h1>
        <button className="logout-btn" onClick={onLogout}>
          Log Out
        </button>
      </div>

      {/* Form to add operator */}
      <form className="admin-form" onSubmit={handleAdd}>
        <h3 className="form-subtitle">Add New Operator</h3>
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
        {operators.length === 0 && (
          <p className="no-ops">No operators registered yet.</p>
        )}

        {operators.map((op, index) => (
          <li key={index} className="operator-item">
            <strong>
              {op.name} {op.lastname}
            </strong>{" "}
            <span className="op-username">({op.username})</span>
          </li>
        ))}
      </ul>
    </div>
  );
}