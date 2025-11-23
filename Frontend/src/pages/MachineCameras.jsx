import Sidebar from "../components/Sidebar/Sidebar";
import Header from "../components/Header/Header";
import styles from "./MachineCamara.module.css";
export default function TrainCameras({ user }) {
  return (
    <div className={styles.container}>
      {/* Sidebar izquierda */}
      <Sidebar />
      {/* Contenido principal */}
      <main className={styles.main}>
        <Header user={user} />
        <h2 className={styles.title}>Train Cameras Monitoring</h2>
        <div className={styles.grid}>
          
          {/* Cámara 1 */}
          <div className={styles.cameraCard}>
            <h3 className={styles.camTitle}>Camara Maquina 1</h3>

            <img
              src="http://localhost:8000/api/stream-video/1/" // puedes cambiar el 1 por el ID
              alt="Video en vivo"
              style={{
                width: "100%",
                height: "240px",
                objectFit: "cover",
                borderRadius: "8px",
                border: "1px solid #444",
                display: "block"
              }}
            />
          </div>
          {/* Cámara 2 */}
          <div className={styles.cameraCard}>
            <h3 className={styles.camTitle}>Camara Maquina 2</h3>

            <img
              src="http://localhost:8000/api/stream-video/2/" // puedes cambiar el 1 por el ID
              alt="Video en vivo"
              style={{
                width: "100%",
                height: "240px",
                objectFit: "cover",
                borderRadius: "8px",
                border: "1px solid #444",
                display: "block"
              }}
            />
          </div>
          {/* Cámara 3 */}
          <div className={styles.cameraCard}>
            <h3 className={styles.camTitle}>Camara Maquina 3</h3>

            <img
              src="http://localhost:8000/api/stream-video/3/" // puedes cambiar el 1 por el ID
              alt="Video en vivo"
              style={{
                width: "100%",
                height: "240px",
                objectFit: "cover",
                borderRadius: "8px",
                border: "1px solid #444",
                display: "block"
              }}
            />
          </div>
        </div>
      </main>
    </div>
  );
} 