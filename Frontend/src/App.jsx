import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";

import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import AdminPanel from "./pages/AdminPanel";
// Importa los nuevos componentes de página
import Train from "./pages/Train"; // Asegúrate de que este archivo exista
import Machine from "./pages/machine"; // Asegúrate de que este archivo exista
import TrainTrip from "./pages/traintrip"; // Asegúrate de que este archivo exista
import DiagnosticsHistory from "./pages/DiagnosticsHistory";
import TrainCameras from "./pages/MachineCameras";

export default function App() {
  const [role, setRole] = useState(null);
  const [currentUser, setCurrentUser] = useState(null);

  const handleLogin = (role, userData) => {
    setRole(role);
    setCurrentUser(userData);
  };

  const handleLogout = () => {
    setRole(null);
    setCurrentUser(null);
  };

  if (!role) return <Login onLogin={handleLogin} />;

  return (
    <Routes>
      {/* ADMIN */}
      {role === "admin" && (
        <>
          <Route
            path="/admin"
            element={<AdminPanel onLogout={handleLogout} />}
          />
          {/* AÑADE ESTAS NUEVAS RUTAS para los enlaces de la NavBar: */}
          <Route
            path="/admin/trains" // NUEVA RUTA 1
            element={<Train />}
          />
          <Route
            path="/admin/machines" // NUEVA RUTA 2
            element={<Machine />}
          />
          <Route
            path="/admin/traintrips" // NUEVA RUTA 3
            element={<TrainTrip />}
          />
          {/* La redirección debe ser al final, después de todas las rutas válidas de Admin */}
          <Route path="*" element={<Navigate to="/admin" />} />
        </>
      )}

      {/* OPERATOR */}
      {role === "operator" && (
        <>
          <Route
            path="/dashboard"
            element={<Dashboard user={currentUser} />}
          />
          <Route
            path="/train-cameras"
            element={<TrainCameras user={currentUser} />}
          />

          <Route
            path="/diagnostics"
            element={<DiagnosticsHistory user={currentUser} />}
          />
          <Route path="*" element={<Navigate to="/dashboard" />} />
        </>
      )}
    </Routes>
  );
}