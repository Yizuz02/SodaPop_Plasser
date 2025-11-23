import L from "leaflet";

// Tiempo global de simulación
export let globalSimTime = 0;
export function updateSimTime(t) {
  globalSimTime = t;
}

/* -------------------------------------------------------
UTILS
--------------------------------------------------------*/
/** Encuentra el punto más cercano de una ruta */
function findClosestIndex(path, targetLat, targetLon) {
  let best = 0;
  let bestDist = Infinity;

  for (let i = 0; i < path.length; i++) {
    const [lat, lon] = path[i];
    const d = (lat - targetLat) ** 2 + (lon - targetLon) ** 2;
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

/** Animación sin duplicar markers */
function animateMarkerAlongPath({
  path,
  durationMs,
  marker,
  registry,
  isTrain = false,
  lineIdx,
  blockState,
  onStep,
  onComplete,
  checkIfFinished,
}) {
  if (!marker || !path || path.length < 2) return;

  const steps = path.length - 1;
  const stepTime = durationMs / steps;
  let idx = 0;
  const safeDistance = 10; // distancia de seguridad antes de un bloqueo

  const intervalId = setInterval(() => {
    // Si es tren y hay bloqueo en la línea
    if (isTrain && blockState.blocks[lineIdx]) {
      const blockedIdx = blockState.blocks[lineIdx].idx;
      if (idx + safeDistance >= blockedIdx) {
        return; // se detiene antes del bloqueo
      }
    }

    idx++;

    if (idx >= path.length) {
      clearInterval(intervalId);
      if (onComplete) onComplete(marker);

      // Marcar esta animación como terminada
      const entry = registry.find((x) => x.intervalId === intervalId);
      if (entry) entry.intervalId = null;

      if (checkIfFinished) checkIfFinished();
      return;
    }

    marker.setLatLng(path[idx]);
    if (onStep) onStep(idx, marker);
  }, stepTime);

  registry.push({ marker, intervalId });
}

/* -------------------------------------------------------
SIMULADOR PRINCIPAL
--------------------------------------------------------*/
export function scheduleSimulation({
  map,
  layerGroup,
  lineSegments,
  schedule,
  registryRef,
  onFinish, // callback cuando toda la simulación termina
}) {
  const registry = [];
  registryRef.current = registry;

  const blueMachines = {};        // Máquinas de bateo (una por ID)
  const blockState = { blocks: {} }; // Bloqueos por línea
  const tampingLaunchedForLine = {}; // Evita lanzar bateo múltiples veces por línea

  /* -------------------------------------------------------
  FUNCIÓN PARA SABER SI TODO TERMINÓ
  --------------------------------------------------------*/
  function checkIfFinished() {
    const stillRunning = registry.some((i) => i.intervalId !== null);
    if (!stillRunning && onFinish) {
      onFinish();
    }
  }

  /* -------------------------------------------------------
  ICONOS
  --------------------------------------------------------*/
  const trainIcon = L.divIcon({
    html: '<div style="background:#ffcc00;width:24px;height:16px;border-radius:4px;box-shadow:0 0 15px #fc0;display:flex;align-items:center;justify-content:center;font-size:18px;">Train</div>',
    iconSize: [24, 16],
    iconAnchor: [12, 8],
  });

  const tampingIcon = L.divIcon({
    html: '<div style="background:#00e0ff;width:22px;height:18px;border-radius:4px;box-shadow:0 0 12px #0ef;display:flex;align-items:center;justify-content:center;font-size:16px;">Tool</div>',
    iconSize: [22, 18],
    iconAnchor: [11, 9],
  });

  /* -------------------------------------------------------
  ACTIVAR REPARACIÓN (MÁQUINA DE BATEO)
  --------------------------------------------------------*/
  function triggerTampingForLine(lineIdx) {
    const fullPath = lineSegments[lineIdx];
    if (!fullPath) return;

    const machines = schedule.tampingMachines.filter((m) => m.line - 1 === lineIdx);

    machines.forEach((tm) => {
      const targetIndex =
        tm.targetIndex ?? findClosestIndex(fullPath, tm.targetLat, tm.targetLon);

      const [zLat, zLon] = fullPath[targetIndex];

      // Marcar bloqueo desde el inicio
      blockState.blocks[lineIdx] = { idx: 0 };

      // Zona de hundimiento
      const zoneMarker = L.circleMarker([zLat, zLon], {
        radius: 8,
        color: "#00e0ff",
        fillColor: "#00e0ff",
        fillOpacity: 0.4,
        weight: 2,
      }).addTo(layerGroup);

      registry.push({ marker: zoneMarker });

      // Popup informativo
      L.popup()
        .setLatLng([zLat, zLon])
        .setContent(
          `<b>Warning Hundimiento detectado</b><br>Línea ${tm.line}<br>${tm.description || ''}`
        )
        .openOn(map);

      const pathToTarget = fullPath.slice(0, targetIndex + 1);
      const restPath = fullPath.slice(targetIndex);

      const dwellMs = 4000;   // Tiempo en la zona de reparación
      const travelMs = 3000;  // Tiempo de viaje (ida y vuelta)

      // Crear máquina si no existe
      if (!blueMachines[tm.id]) {
        blueMachines[tm.id] = L.marker(pathToTarget[0], { icon: tampingIcon }).addTo(layerGroup);
      }
      const blue = blueMachines[tm.id];

      const onStepTo = (i) => {
        blockState.blocks[lineIdx] = { idx: i };
      };

      const onStepRest = (i) => {
        blockState.blocks[lineIdx] = { idx: targetIndex + i };
      };

      /* --- IR a reparar --- */
      animateMarkerAlongPath({
        path: pathToTarget,
        durationMs: travelMs,
        marker: blue,
        registry,
        isTrain: false,
        lineIdx,
        blockState,
        onStep: onStepTo,
        onComplete: () => {
          setTimeout(() => {
            layerGroup.removeLayer(zoneMarker);
            blockState.blocks[lineIdx] = null;

            /* --- VOLVER por el resto del camino --- */
            animateMarkerAlongPath({
              path: restPath,
              durationMs: travelMs,
              marker: blue,
              registry,
              isTrain: false,
              lineIdx,
              blockState,
              onStep: onStepRest,
              onComplete: () => {
                if (blue) layerGroup.removeLayer(blue);
                delete blueMachines[tm.id];
                delete blockState.blocks[lineIdx];
                checkIfFinished();
              },
              checkIfFinished,
            });
          }, dwellMs);
        },
        checkIfFinished,
      });
    });
  }

  /* -------------------------------------------------------
  TRENES CONTROLADOS POR simTime
  --------------------------------------------------------*/
  function processTrainSchedule() {
    schedule.trains.forEach((train) => {
      const lineIdx = train.line - 1;
      const path = lineSegments[lineIdx];
      if (!path) return;

      // Lanzar tren solo una vez cuando llegue su hora
      if (!train._launched && globalSimTime >= train.departureSec) {
        train._launched = true;

        // Si hay bloqueo activo en la línea, no lanzar el tren
        if (blockState.blocks[lineIdx]) {
          return;
        }

        const marker = L.marker(path[0], { icon: trainIcon }).addTo(layerGroup);

        animateMarkerAlongPath({
          path,
          durationMs: train.travelTimeSec * 1000,
          marker,
          registry,
          isTrain: true,
          lineIdx,
          blockState,
          onComplete: () => {
            layerGroup.removeLayer(marker);

            // Lanzar máquina de bateo solo la primera vez que pase un tren por esta línea
            if (!tampingLaunchedForLine[lineIdx]) {
              tampingLaunchedForLine[lineIdx] = true;
              triggerTampingForLine(lineIdx);
            }
          },
          checkIfFinished,
        });
      }
    });
  }

  /* -------------------------------------------------------
  LOOP PRINCIPAL
  --------------------------------------------------------*/
  const mainLoop = setInterval(() => {
    updateSimTime(globalSimTime + 0.2); // avanza 0.2 segundos por tick
    processTrainSchedule();
  }, 200);

  // Guardar referencia para poder pararlo si es necesario
  registryRef.current.push({ intervalId: mainLoop });
}

/* -------------------------------------------------------
LIMPIEZA
--------------------------------------------------------*/
export function clearSimulation(registryRef, layerGroup) {
  if (!registryRef.current) return;

  registryRef.current.forEach((item) => {
    if (item.intervalId) clearInterval(item.intervalId);
    if (item.marker && layerGroup.hasLayer(item.marker)) {
      layerGroup.removeLayer(item.marker);
    }
  });

  registryRef.current = [];
}