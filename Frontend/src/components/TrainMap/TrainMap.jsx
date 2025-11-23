// src/components/TrainMap.jsx
import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import styles from "./TrainMap.module.css";

import { drawRailLines } from "./drawRailLines";
import { scheduleSimulation, clearSimulation } from "./scheduleSimulation";
import useSimClock from "../hook/useSimClock";
import { updateSimTime } from "./scheduleSimulation";

export default function TrainMap() {
  const simTime = useSimClock({ speed: 10 });

  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const routeLayerRef = useRef(null);
  const simulationRegistryRef = useRef([]); // <-- ahora es array desde el inicio

  const token = sessionStorage.getItem("authToken");

  // Actualizar tiempo global
  useEffect(() => {
    updateSimTime(simTime);
  }, [simTime]);

  useEffect(() => {
    // ---------- INICIALIZAR MAPA ----------
    if (!mapInstanceRef.current && mapRef.current) {
      mapInstanceRef.current = L.map(mapRef.current, {
        center: [45.0, 10.0],
        zoom: 5,
        zoomControl: true,
      });

      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        maxZoom: 19,
      }).addTo(mapInstanceRef.current);
    }

    // Limpiar capa anterior si existe
    if (routeLayerRef.current) {
      mapInstanceRef.current.removeLayer(routeLayerRef.current);
    }
    routeLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);

    // ---------- CARGAR CSV DE VÍAS ----------
    fetch("/resources/datos_inspeccion_vias.csv")
      .then((res) => res.text())
      .then((text) => {
        const rows = text.trim().split("\n");
        const header = rows[0].split(",");
        const latIdx = header.indexOf("Lat_caro");
        const lonIdx = header.indexOf("Lon_caro");

        if (latIdx === -1 || lonIdx === -1) {
          console.error("No se encontraron columnas Lat_caro / Lon_caro");
          return;
        }

        const latlngs = [];
        for (let i = 1; i < rows.length; i++) {
          const cols = rows[i].split(",");
          const lat = parseFloat(cols[latIdx]);
          const lon = parseFloat(cols[lonIdx]);
          if (!isNaN(lat) && !isNaN(lon)) latlngs.push([lat, lon]);
        }

        if (latlngs.length === 0) return;

        const stationIcon = L.divIcon({
          html: `<div style="background:white;width:10px;height:10px;border-radius:50%;border:3px solid #ffcc00;box-shadow:0 0 8px rgba(255,204,0,0.8);"></div>`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        });

        const { lineSegments, bounds } = drawRailLines({
          map: mapInstanceRef.current,
          layerGroup: routeLayerRef.current,
          latlngs,
          stationIcon,
        });

        if (bounds.isValid()) {
          mapInstanceRef.current.fitBounds(bounds, { padding: [40, 40] });
        }

        // ---------- CARGAR HORARIOS Y LANZAR SIMULACIÓN ----------
        fetch("/resources/trains_schedule.json")
          .then((r) => r.json())
          .then((schedule) => {
            // Iniciar simulación (sin onFinish)
            scheduleSimulation({
              map: mapInstanceRef.current,
              layerGroup: routeLayerRef.current,
              lineSegments,
              schedule,
              registryRef: simulationRegistryRef,
            });

            // ---- LLAMAR A LA API DESPUÉS DE 15 SEGUNDOS ----
            const timeoutId = setTimeout(() => {
              console.log("15 segundos → Generando reporte de bateo...");

              fetch("/api/tamper-report/", {
                method: "GET",
                headers: {
                  Authorization: `Token ${token}`,
                  "Content-Type": "application/json",
                },
              })
                .then((res) => {
                  if (!res.ok) throw new Error("Error en el servidor");
                  return res.json();
                })
                .then((data) => {
                  console.log("Reporte recibido:", data);
                  alert("Reporte de mantenimiento generado correctamente");
                })
                .catch((err) => {
                  console.error("Error al obtener reporte:", err);
                  alert("No se pudo generar el reporte");
                });
            }, 15_000); // 15 segundos

            // Guardar timeout para poder cancelarlo
            simulationRegistryRef.current.push({ timeoutId });
          })
          .catch((err) => console.error("Error cargando trains_schedule.json", err));
      })
      .catch((err) => console.error("Error cargando CSV", err));

    // ---------- LIMPIEZA ----------
    return () => {
      if (simulationRegistryRef.current) {
        simulationRegistryRef.current.forEach((item) => {
          if (item.timeoutId) clearTimeout(item.timeoutId);
          if (item.intervalId) clearInterval(item.intervalId);
        });
      }
      clearSimulation(simulationRegistryRef, routeLayerRef.current);
    };
  }, []);

  return <div ref={mapRef} className={styles.map}></div>;
}