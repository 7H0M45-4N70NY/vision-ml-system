"use client";

import { useEffect, useRef, useState } from "react";
import { useStore } from "@/store/useStore";

export function VideoCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { setTelemetry, streamSource } = useStore();
  const [imgKey, setImgKey] = useState(Date.now());

  // Force img reload when stream source changes
  useEffect(() => {
    setImgKey(Date.now());
  }, [streamSource]);

  // Connect to WebSocket with reconnection logic
  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/live-stream";
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        setTelemetry({ isConnected: true, modelLoaded: true });
        console.log("WebSocket Connected");
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
        console.log("WebSocket Disconnected. Retrying in 3s...");
        reconnectTimeout = setTimeout(connect, 3000);
      };

      socket.onerror = (err) => {
        console.error("WebSocket Error", err);
        socket?.close();
      };
    };

    connect();

    return () => {
      if (socket) {
        socket.onclose = null; // Prevent reconnection on unmount
        socket.close();
      }
      clearTimeout(reconnectTimeout);
    };
  }, [setTelemetry]);

  return (
    <div className="flex-1 bg-zinc-900 rounded-lg overflow-hidden relative border border-zinc-800" ref={containerRef}>
      {/* Live Stream MJPEG Feed */}
      <img
        key={imgKey}
        src={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/video_feed?t=${imgKey}`}
        alt="Live inference video stream"
        className="absolute inset-0 w-full h-full object-contain bg-zinc-950"
      />

      {/* Stream label */}
      <div className="absolute inset-x-0 bottom-0 p-2 bg-gradient-to-t from-black/80 to-transparent pointer-events-none">
        <span className="text-xs text-zinc-400">Live Inference Feed</span>
      </div>
    </div>
  );
}
