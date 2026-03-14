"use client";

import { useEffect, useRef, useState } from "react";
import { useStore } from "@/store/useStore";

// Mock BBox Data
interface BBox {
  id: number;
  class: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

const mockBBoxes: BBox[] = [
  { id: 1, class: "person", confidence: 0.94, x1: 200, y1: 150, x2: 350, y2: 450 }
];

export function VideoCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const { setTelemetry } = useStore();

  useEffect(() => {
    // Handle container resize
    let timeoutId: NodeJS.Timeout;
    const observer = new ResizeObserver((entries) => {
      if (entries[0]) {
        const { width, height } = entries[0].contentRect;
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            setDimensions({ width, height });
        }, 100);
      }
    });

    if (containerRef.current) observer.observe(containerRef.current);
    return () => {
        observer.disconnect();
        clearTimeout(timeoutId);
    };
  }, []);

  // Connect to actual WebSocket
  useEffect(() => {
     let wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/live-stream";
     const socket = new WebSocket(wsUrl);

     socket.onopen = () => {
       setTelemetry({ isConnected: true, modelLoaded: true });
     };

     socket.onmessage = (event) => {
       try {
         const data = JSON.parse(event.data);
         setTelemetry(data);
       } catch (err) {
         console.error("Telemetry parse error", err);
       }
     };

     socket.onclose = () => {
       setTelemetry({ isConnected: false });
     };

     return () => {
       socket.close();
     };
  }, [setTelemetry]);

  return (
    <div className="flex-1 bg-zinc-900 rounded-lg overflow-hidden relative border border-zinc-800" ref={containerRef}>
      {/* Live Stream MJPEG Feed */}
      <img
          src={process.env.NEXT_PUBLIC_API_URL ? `${process.env.NEXT_PUBLIC_API_URL}/video_feed` : "http://localhost:8000/video_feed"}
          alt="Live Video Stream"
          className="absolute inset-0 w-full h-full object-contain bg-zinc-950"
      />
      
      {/* Connection Offline Indicator */}
      <div className="absolute inset-x-0 bottom-0 p-2 bg-gradient-to-t from-black/80 to-transparent pointer-events-none">
         <span className="text-xs text-zinc-400">Live Inference Feed</span>
      </div>
    </div>
  );
}
