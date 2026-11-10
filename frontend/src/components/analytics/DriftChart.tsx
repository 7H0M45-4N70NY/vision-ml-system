"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useEffect, useState } from "react";
import { useMounted } from "@/hooks/useMounted";

interface DriftPoint {
  date: string;
  score: number;
}

export function DriftChart() {
  const [data, setData] = useState<DriftPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const mounted = useMounted();

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/analytics/timeseries`)
      .then(res => res.json())
      .then(d => setData(d.drift ?? []))
      .catch(() => setError("Failed to load drift data"));
  }, []);

  return (
    <Card className="bg-zinc-950/50 border-zinc-800">
      <CardHeader>
        <CardTitle className="text-base font-medium">Model Drift Trends</CardTitle>
        <CardDescription>Confidence domain shift score over recent runs</CardDescription>
      </CardHeader>
      <CardContent className="h-64">
        {error ? (
          <div className="h-full flex items-center justify-center text-sm text-zinc-500">{error}</div>
        ) : data.length === 0 ? (
          <div className="h-full flex items-center justify-center text-sm text-zinc-500">
            No drift points recorded yet. Run inference to evaluate confidence drift.
          </div>
        ) : mounted ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="date" stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: "var(--background)", borderColor: "var(--border)", borderRadius: "6px" }}
                itemStyle={{ color: "#ef4444", fontWeight: "bold" }}
              />
              <Area type="monotone" dataKey="score" stroke="#ef4444" fillOpacity={1} fill="url(#colorScore)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        ) : null}
      </CardContent>
    </Card>
  );
}
