import { useEffect, useState } from "react";
import styles from "./MachinesStatus.module.css";
import MachineCard from "./MachineCard";

export default function MachinesStatus() {
  const [machines, setMachines] = useState([]);
  const [loading, setLoading] = useState(true);

  // Supongamos que tienes el token guardado en algún lugar
  const token = sessionStorage.getItem("authToken");

  useEffect(() => {
    fetch("http://localhost:8000/api/machines-status/", {
      method: "GET",
      headers: {
        "Authorization": `Token ${token}`,
        "Content-Type": "application/json",
      },
    })
      .then((res) => {
        if (!res.ok) throw new Error("Error al obtener las máquinas");
        return res.json();
      })
      .then((data) => {
        setMachines(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching machines:", err);
        setLoading(false);
      });
  }, [token]);

  const getStatusColor = (status) => {
    if (status === "Free") return "green";
    if (status === "Busy") return "red";
    return "yellow";
  };

  if (loading) return <p>Cargando máquinas...</p>;

  return (
    <div className={styles.container}>
      {machines.map((machine) => (
        <MachineCard
          key={machine.id}
          title={machine.model}
          status={
            machine.status === "Busy" && machine.current_route
              ? `Ocupado en ruta ${machine.current_route}`
              : machine.status
          }
          color={getStatusColor(machine.status)}
        />
      ))}
    </div>
  );
}
