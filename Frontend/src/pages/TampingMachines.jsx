import { useState, useEffect, useRef } from "react";
import Sidebar from "../components/Sidebar/Sidebar";
import Header from "../components/Header/Header";
import styles from "./MachineCamara.module.css";

// Configuración de la API
const API_BASE_URL = "http://localhost:5000/api";

// Videos de fallback
const FALLBACK_VIDEOS = [
  "/videos/Tamping/video1.mp4",
  "/videos/Tamping/video2.mp4",
];

// Configuración de las máquinas bateadoras
const TAMPING_CONFIG = [
  {
    id: 1,
    title: "Tamping Machine #1",
    description: "Zona industrial - Parada a los 5s",
    apiEndpoint: "/videos/tamping/1",
  },
  {
    id: 2,
    title: "Tamping Machine #2",
    description: "Zona rural - Parada a los 8s",
    apiEndpoint: "/videos/tamping/2",
  },
];

// Componente de tarjeta de video
function VideoCard({ config, fallbackSrc }) {
  const videoRef = useRef(null);
  const [videoSrc, setVideoSrc] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isUsingFallback, setIsUsingFallback] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [debugInfo, setDebugInfo] = useState({});

  const MAX_RETRIES = 3;

  const addDebug = (key, value) => {
    console.log(`[Tamping ${config.id}] ${key}:`, value);
    setDebugInfo(prev => ({ ...prev, [key]: value, lastUpdate: new Date().toISOString() }));
  };

  const loadVideoFromAPI = async () => {
    setIsLoading(true);
    setError(null);
    addDebug("status", "Iniciando carga...");

    const url = `${API_BASE_URL}${config.apiEndpoint}`;
    addDebug("url", url);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout para videos largos

      addDebug("fetch", "Iniciando fetch...");
      const response = await fetch(url, { signal: controller.signal });

      clearTimeout(timeoutId);

      addDebug("response.ok", response.ok);
      addDebug("response.status", response.status);
      addDebug("content-type", response.headers.get("content-type"));

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      addDebug("blob", "Obteniendo blob...");
      const blob = await response.blob();
      addDebug("blob.size", blob.size);
      addDebug("blob.type", blob.type);

      if (blob.size === 0) {
        throw new Error("El servidor devolvió un video vacío");
      }

      let finalBlob = blob;
      if (!blob.type || blob.type === "" || blob.type === "application/octet-stream") {
        finalBlob = new Blob([blob], { type: "video/mp4" });
      }

      const blobUrl = URL.createObjectURL(finalBlob);
      addDebug("blobUrl", blobUrl);
      
      setVideoSrc(blobUrl);
      setIsUsingFallback(false);
      setIsLoading(false);
      setError(null);
      addDebug("status", "✅ Video cargado correctamente");

    } catch (err) {
      console.error(`Error loading tamping ${config.id}:`, err);
      addDebug("error", err.message);
      
      if (retryCount < MAX_RETRIES && err.name !== 'AbortError') {
        addDebug("retry", `Reintentando en ${2 * (retryCount + 1)}s...`);
        setTimeout(() => setRetryCount(prev => prev + 1), 2000 * (retryCount + 1));
      } else {
        addDebug("fallback", "Usando video de fallback");
        setVideoSrc(fallbackSrc);
        setIsUsingFallback(true);
        setIsLoading(false);
        setError(err.message);
      }
    }
  };

  useEffect(() => {
    loadVideoFromAPI();
    return () => {
      if (videoSrc && videoSrc.startsWith("blob:")) {
        URL.revokeObjectURL(videoSrc);
      }
    };
  }, [retryCount]);

  const handleRetry = () => {
    setRetryCount(0);
    setDebugInfo({});
    if (videoSrc && videoSrc.startsWith("blob:")) {
      URL.revokeObjectURL(videoSrc);
    }
    setVideoSrc(null);
    loadVideoFromAPI();
  };

  const handleVideoError = (e) => {
    const video = e.target;
    const error = video.error;
    
    addDebug("videoError", {
      code: error?.code,
      message: error?.message,
    });

    if (!isUsingFallback) {
      setVideoSrc(fallbackSrc);
      setIsUsingFallback(true);
      setError(`Error de reproducción: ${error?.message || 'Formato no soportado'}`);
    }
  };

  return (
    <div className={styles.cameraCard}>
      <div className={styles.cardHeader}>
        <h3 className={styles.camTitle}>{config.title}</h3>
        {isUsingFallback && (
          <span className={styles.fallbackBadge}>⚠️ Offline</span>
        )}
        {!isUsingFallback && !isLoading && !error && (
          <span className={styles.liveBadge}>
            <span className={styles.liveIndicator}></span>
            API
          </span>
        )}
      </div>

      <p className={styles.camDescription}>{config.description}</p>

      <div className={styles.videoContainer}>
        {isLoading ? (
          <div className={styles.loadingOverlay}>
            <div className={styles.spinner}></div>
            <p>Generando simulación...</p>
            <p className={styles.retryText}>
              {retryCount > 0 ? `Reintento ${retryCount}/${MAX_RETRIES}` : "Esto puede tomar unos segundos"}
            </p>
          </div>
        ) : (
          <video
            ref={videoRef}
            className={styles.video}
            controls
            autoPlay
            loop
            muted
            playsInline
            src={videoSrc}
            onError={handleVideoError}
            onCanPlay={() => addDebug("videoEvent", "canplay ✅")}
            onPlaying={() => addDebug("videoEvent", "playing ✅")}
          />
        )}
      </div>

      <div className={styles.debugPanel}>
        <details>
          <summary className={styles.debugSummary}>
            🔧 Debug Info (Tamping {config.id})
          </summary>
          <div className={styles.debugContent}>
            <pre>{JSON.stringify(debugInfo, null, 2)}</pre>
          </div>
        </details>
      </div>

      {error && (
        <div className={styles.errorContainer}>
          <p className={styles.errorText}>
            <span className={styles.errorIcon}>⚠️</span>
            {error}
          </p>
          <button className={styles.retryButton} onClick={handleRetry}>
            🔄 Reintentar
          </button>
        </div>
      )}
    </div>
  );
}

// Componente principal
export default function TampingMachines({ user }) {
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    const checkAPIHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`, {
          signal: AbortSignal.timeout(5000),
        });
        setApiStatus(response.ok ? "online" : "offline");
      } catch {
        setApiStatus("offline");
      }
    };

    checkAPIHealth();
    const interval = setInterval(checkAPIHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={styles.container}>
      <Sidebar />
      
      <main className={styles.main}>
        <Header user={user} />

        <div className={styles.titleContainer}>
          <h2 className={styles.title}>Tamping Machines Monitoring</h2>
          <div className={styles.apiStatusContainer}>
            <span
              className={`${styles.apiStatus} ${
                apiStatus === "online"
                  ? styles.apiOnline
                  : apiStatus === "offline"
                  ? styles.apiOffline
                  : styles.apiChecking
              }`}
            >
              <span className={styles.statusDot}></span>
              {apiStatus === "online" && "API Conectada"}
              {apiStatus === "offline" && "API Desconectada"}
              {apiStatus === "checking" && "Verificando..."}
            </span>
          </div>
        </div>

        {apiStatus === "offline" && (
          <div className={styles.warningBanner}>
            <span className={styles.warningIcon}>⚠️</span>
            <p>
              No se puede conectar con la API del simulador.
            </p>
            <span className={styles.warningHint}>
              Servidor: {API_BASE_URL}
            </span>
          </div>
        )}

        <div className={styles.grid}>
          {TAMPING_CONFIG.map((config, index) => (
            <VideoCard
              key={config.id}
              config={config}
              fallbackSrc={FALLBACK_VIDEOS[index]}
            />
          ))}
        </div>
      </main>
    </div>
  );
}