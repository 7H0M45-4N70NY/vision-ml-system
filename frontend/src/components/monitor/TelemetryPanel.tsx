"use client";

import { useStore } from "@/store/useStore";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { ScrollArea } from "@/components/ui/scroll-area";
import React, { useEffect, useState } from "react";

// Mock data for initial rendering until WebSocket is hooked up
const mockChartData = Array.from({ length: 20 }, (_, i) => ({
  time: i,
  confidence: 0.7 + Math.random() * 0.2,
}));

export function TelemetryPanel() {
  const { fps, latency, objectCount, isConnected } = useStore();
  const [logs, setLogs] = useState<string[]>([]);
  const previousObjectCount = React.useRef(0);

  // Generate traceability events when active objects change
  useEffect(() => {
    if (objectCount !== previousObjectCount.current) {
        const diff = objectCount - previousObjectCount.current;
        const timestamp = new Date().toISOString().split('T')[1].slice(0, 11); // HH:MM:SS.mmm
        
        let message = '';
        if (diff > 0) {
            message = `[EVENT] +${diff} Object(s) detected. Total active: ${objectCount}`;
        } else if (diff < 0) {
            message = `[EVENT] ${diff} Object(s) lost. Total active: ${objectCount}`;
        }
        
        if (message) {
            setLogs(prev => {
                const newLogs = [ `${timestamp} ${message}`, ...prev];
                return newLogs.slice(0, 50); // Keep last 50
            });
        }
        
        previousObjectCount.current = objectCount;
    }
  }, [objectCount]);

  return (
    <div className="flex flex-col gap-4 w-full xl:w-80 shrink-0 overflow-y-visible xl:overflow-y-auto xl:pl-2">
      <div className="grid grid-cols-2 gap-2">
        <Card className="bg-zinc-950/50 border-zinc-800">
          <CardContent className="p-4">
            <div className="text-xs text-zinc-500 font-medium mb-1">FPS</div>
            <div className="text-2xl font-bold text-zinc-100">{fps.toFixed(0)}</div>
          </CardContent>
        </Card>
        <Card className="bg-zinc-950/50 border-zinc-800">
          <CardContent className="p-4">
            <div className="text-xs text-zinc-500 font-medium mb-1">Latency</div>
            <div className="text-2xl font-bold text-zinc-100">{latency} <span className="text-sm font-normal text-zinc-500">ms</span></div>
          </CardContent>
        </Card>
        <Card className="bg-zinc-950/50 border-zinc-800 col-span-2">
          <CardContent className="p-4 flex justify-between items-center">
            <div className="text-xs text-zinc-500 font-medium">Objects Detected</div>
            <div className="text-xl font-bold text-primary">{objectCount}</div>
          </CardContent>
        </Card>
      </div>

      <Card className="bg-zinc-950/50 border-zinc-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium font-mono">Confidence Drift (60s)</CardTitle>
        </CardHeader>
        <CardContent className="h-40 p-0 pl-[-10px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mockChartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <YAxis domain={[0, 1]} hide />
              <Tooltip
                contentStyle={{ backgroundColor: "#18181b", borderColor: "#27272a", fontSize: "12px" }}
                itemStyle={{ color: "#f4f4f5" }}
                labelStyle={{ display: "none" }}
              />
              <Line type="monotone" dataKey="confidence" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="bg-zinc-950/50 border-zinc-800 flex-1 flex flex-col min-h-[200px]">
        <CardHeader className="py-3 px-4 border-b border-zinc-800/50 flex flex-row items-center justify-between">
          <CardTitle className="text-xs font-medium font-mono text-zinc-400">Live Event Traceability</CardTitle>
          <div className="flex items-center gap-1.5">
              <div className={`h-1.5 w-1.5 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
              <span className="text-[10px] text-zinc-500 uppercase">{isConnected ? 'Recording' : 'Offline'}</span>
          </div>
        </CardHeader>
        <CardContent className="p-0 flex-1 relative">
          <ScrollArea className="absolute inset-0 p-4">
            <div className="space-y-2 font-mono text-[10px] text-zinc-300 leading-tight">
              {logs.length === 0 && <span className="text-zinc-600 italic">Waiting for events...</span>}
              {logs.map((log, i) => (
                <div key={i} className={
                    log.includes("lost") ? "text-amber-500" : 
                    log.includes("+") ? "text-emerald-400" : 
                    "text-zinc-400"
                }>
                  {log}
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
