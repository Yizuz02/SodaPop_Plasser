import { useEffect } from "react";
import Sidebar from "../components/Sidebar/Sidebar.jsx";
import Header from "../components/Header/Header.jsx";
import TrainMap from "../components/TrainMap/TrainMap.jsx";
import AlertsPanel from "../components/AlertsPanel/XAlertsPanel.jsx";
import MachinesStatus from "../components/MachinesStatus/MachinesStatus.jsx";
import ArrivalTimer from "../components/ArrivalTimer/ArrivalTimer.jsx";
import KPIsPanel from "../components/KPIsPanel/KPIsPanel.jsx";

import styles from "./Dashboard.module.css";

export default function Dashboard({ user }) {
  
  useEffect(() => {
    const token = sessionStorage.getItem("authToken");

    async function callTamperReportAPI() {
      try {
        const response = await fetch(
          "http://127.0.0.1:8000/api/tamper-report/",
          {
            method: "GET",
            headers: {
              Authorization: `Token ${token}`,
              "Content-Type": "application/json",
            },
          }
        );

        if (!response.ok) {
          const err = await response.text();
          console.error("API error:", err);
          return;
        }

        const data = await response.json();
        console.log("Tamper report response:", data);

      } catch (err) {
        console.error("Error calling tamper-report:", err);
      }
    }

    // Ejecutar de inmediato al cargar
    callTamperReportAPI();

    // Ejecutar cada 5 minutos
    const interval = setInterval(callTamperReportAPI, 300000);

    // Limpiar al desmontar
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={styles.layout}>
      <Sidebar />

      <main className={styles.content}>
        <Header user={user} />

        <div className={styles.grid}>
          <TrainMap />
          <AlertsPanel />
          <ArrivalTimer />
          <MachinesStatus />
          <KPIsPanel />
        </div>
      </main>
    </div>
  );
}
