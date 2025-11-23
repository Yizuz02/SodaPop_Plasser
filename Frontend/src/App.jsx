import { useState } from "react";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import AdminPanel from "./pages/AdminPanel";

export default function App() {
  const [role, setRole] = useState(null);

  const handleLogout = () => setRole(null);

  if (!role) return <Login onLogin={(r) => setRole(r)} />;

  if (role === "admin")
    return <AdminPanel onLogout={handleLogout} />;

  if (role === "operator")
    return <Dashboard onLogout={handleLogout} />;

  return null;
}