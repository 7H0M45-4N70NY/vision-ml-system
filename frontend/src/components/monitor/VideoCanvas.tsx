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
        }, 100); // 100ms debounce
      }
    });

    if (containerRef.current) observer.observe(containerRef.current);
    return () => {
        observer.disconnect();
        clearTimeout(timeoutId);
    };
  }, []);

  // Set mock telemetry
  useEffect(() => {
     setTelemetry({ objectCount: mockBBoxes.length, fps: 30, latency: 120, gpuUtilization: 45, modelLoaded: true, isConnected: true });
  }, [setTelemetry]);

  return (
    <div className="flex-1 bg-zinc-900 rounded-lg overflow-hidden relative border border-zinc-800" ref={containerRef}>
      {/* Stream Placeholder (will be replaced by an <img> tag connected to MJPEG stream or a Canvas) */}
      <div className="absolute inset-0 flex items-center justify-center text-zinc-600 bg-zinc-950">
        <div className="text-center">
            <div className="w-16 h-16 rounded-full border-t-2 border-r-2 border-zinc-600 animate-spin mx-auto mb-4" />
            <p className="text-sm font-mono tracking-widest uppercase">Waiting for Stream</p>
        </div>
      </div>

      {/* SVG Overlay for Bounding Boxes */}
      {dimensions.width > 0 && (
        <svg 
          className="absolute inset-0 pointer-events-none" 
          width={dimensions.width} 
          height={dimensions.height}
          viewBox="0 0 1280 720" // Assuming 720p stream
          preserveAspectRatio="xMidYMid slice"
        >
          {mockBBoxes.map((box) => (
            <g key={box.id}>
              {/* Box */}
              <rect
                x={box.x1}
                y={box.y1}
                width={box.x2 - box.x1}
                height={box.y2 - box.y1}
                fill="none"
                stroke="oklch(0.708 0 0)" // Primary color approximation
                strokeWidth="2"
                className="transition-all duration-75"
              />
              {/* Label Background */}
              <rect
                x={box.x1}
                y={box.y1 - 24}
                width="140"
                height="24"
                fill="oklch(0.708 0 0)"
                opacity={0.9}
              />
              {/* Label Text */}
              <text
                x={box.x1 + 6}
                y={box.y1 - 8}
                fill="black"
                className="text-xs font-mono font-bold"
              >
                {box.class} {box.id} ({box.confidence.toFixed(2)})
              </text>
            </g>
          ))}
        </svg>
      )}
    </div>
  );
}
