import { useState, useEffect, useRef } from "react";

export default function useSimClock({ speed = 1 }) {
  const [simTime, setSimTime] = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    intervalRef.current = setInterval(() => {
      setSimTime((prev) => prev + speed);
    }, 1000);

    return () => clearInterval(intervalRef.current);
  }, [speed]);

  return simTime;
}
