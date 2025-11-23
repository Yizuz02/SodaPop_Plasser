import { useState } from "react";
import "./Login.css";

export default function Login({ onLogin }) {
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    const username = user.trim();
    const password = pass.trim();

    if (!username || !password) {
      setError("Please enter username and password");
      return;
    }

    // **Login local para admin**
    const adminUser = "admin";
    const adminPass = "Plasser2025!";
    if (username === adminUser && password === adminPass) {
      // Guardar token ficticio en session para admin local
      sessionStorage.setItem("authToken", "local-admin-token");
      onLogin("admin", "Admin");
      return;
    }

    try {
      // **Login para operadores usando backend**
      const response = await fetch("http://127.0.0.1:8000/api/token/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        throw new Error("Invalid username or password");
      }

      const data = await response.json();
      const token = data.token;

      // Guardar token en sessionStorage
      sessionStorage.setItem("authToken", token);

      // Llamar callback con tipo y nombre de usuario
      onLogin("operator", username);

    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <img
          src="/logoaustria.png"
          className="logo-img"
          alt="Fast Tamping AI Logo"
        />
        <h2 className="title">Fast Tamping AI</h2>
        <p className="subtitle">Railway Maintenance</p>

        {error && <p className="error-msg">{error}</p>}

        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Username"
            value={user}
            onChange={(e) => setUser(e.target.value)}
          />
          <input
            type="password"
            placeholder="Password"
            value={pass}
            onChange={(e) => setPass(e.target.value)}
          />
          <button type="submit">Login</button>
        </form>
      </div>
    </div>
  );
}
