import useSimClock from "../hook/useSimClock";
import styles from "./ArrivalTimer.module.css";

export default function ArrivalTimer() {
  const simTime = useSimClock({ speed: 10 }); // 10x tiempo rápido

  const hh = String(Math.floor(simTime / 3600)).padStart(2, "0");
  const mm = String(Math.floor((simTime % 3600) / 60)).padStart(2, "0");
  const ss = String(simTime % 60).padStart(2, "0");

  return (
    <div className={styles.box}>
      <h3>Simulation Time</h3>
      <p className={styles.time}>{hh}:{mm}:{ss}</p>
      <p className={styles.label}>Sim Speed: 10x</p>
    </div>
  );
}
