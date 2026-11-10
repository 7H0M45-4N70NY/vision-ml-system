"use client";

import { useEffect, useRef, useState } from "react";
import { useStore } from "@/store/useStore";
import { VideoOff, Play, Power, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

function getSourceLabel(src: string): string {
  if (src === "0") return "Webcam 0 (Default)";
  if (src === "1") return "Webcam 1 (Secondary)";
  if (src.startsWith("rtsp://") || src.startsWith("http://")) return `RTSP (${src.length > 30 ? src.slice(0, 30) + "…" : src})`;
  const base = src.split(/[/\\]/).pop();
  return `Video: ${base || src}`;
}

export function VideoCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { setTelemetry, streamSource, streamActive, turnOnStream, turnOffStream, fetchSources } = useStore();
  const [loadError, setLoadError] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  // Fetch initial sources from backend on mount
  useEffect(() => {
    fetchSources();
  }, [fetchSources]);

  // Connect to WebSocket with reconnection logic
  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/live-stream";
      socket = new WebSocket(wsUrl);

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
        reconnectTimeout = setTimeout(connect, 3000);
      };

      socket.onerror = () => {
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
    <div className="flex-1 bg-zinc-900 rounded-lg overflow-hidden relative border border-zinc-800 flex items-center justify-center min-h-[350px]" ref={containerRef}>
      {streamActive ? (
        <>
          {/* Live Stream MJPEG Feed */}
          <img
            key={`${streamSource}-${retryKey}`}
            src={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/video_feed?source=${encodeURIComponent(streamSource)}`}
            alt="Live inference video stream"
            className="absolute inset-0 w-full h-full object-contain bg-zinc-950"
            onError={() => setLoadError(true)}
            onLoad={() => setLoadError(false)}
          />

          {loadError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/90 text-zinc-300 p-6 z-10 space-y-3">
              <RefreshCw className="h-8 w-8 text-amber-500 animate-spin" />
              <div className="text-sm font-medium">Connecting to stream feed...</div>
              <p className="text-xs text-zinc-500 text-center max-w-sm">
                If the backend is not running yet, start it with <code className="text-zinc-300">uvicorn src.vision_ml.api.main:app</code> on port 8000.
              </p>
              <Button
                size="sm"
                variant="outline"
                onClick={() => { setLoadError(false); setRetryKey(k => k + 1); }}
                className="text-xs gap-1.5"
              >
                <RefreshCw className="h-3.5 w-3.5" /> Reconnect
              </Button>
            </div>
          )}

          {/* Stream HUD overlay */}
          <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/90 via-black/40 to-transparent flex items-center justify-between pointer-events-auto z-10">
            <div className="flex items-center gap-2">
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-xs font-mono text-zinc-200">{getSourceLabel(streamSource)}</span>
            </div>
            <Button
              size="sm"
              variant="destructive"
              onClick={turnOffStream}
              className="h-7 text-[11px] px-2.5 gap-1.5 bg-rose-600/80 hover:bg-rose-600"
            >
              <Power className="h-3 w-3" /> Turn Off
            </Button>
          </div>
        </>
      ) : (
        /* Standby / Off Overlay */
        <div className="flex flex-col items-center justify-center text-center p-8 space-y-4 max-w-md">
          <div className="h-14 w-14 rounded-full bg-zinc-800/80 border border-zinc-700/60 flex items-center justify-center text-zinc-400 shadow-inner">
            <VideoOff className="h-7 w-7 text-zinc-400" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-zinc-200 tracking-tight">Stream Turned Off</h3>
            <p className="text-xs text-zinc-400">
              Hardware capture and processing are idle. Selected source: <span className="font-mono text-zinc-300">{getSourceLabel(streamSource)}</span>
            </p>
          </div>
          <div className="pt-2 flex gap-3">
            <Button
              onClick={() => turnOnStream()}
              className="text-xs bg-emerald-600 hover:bg-emerald-500 text-white gap-2 font-medium px-4 shadow-lg shadow-emerald-950/50"
            >
              <Play className="h-3.5 w-3.5 fill-current" /> Turn On Stream
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
