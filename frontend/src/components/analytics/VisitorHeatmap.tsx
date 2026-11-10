"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useMounted } from "@/hooks/useMounted";

interface FlowPoint {
  hour: string;
  visitors: number;
}

export function VisitorHeatmap() {
  const [data, setData] = useState<FlowPoint[]>([]);
  const [totalVisitors, setTotalVisitors] = useState<number>(0);
  const mounted = useMounted();

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/analytics/visitor-flow`)
      .then(res => res.json())
      .then(d => {
        const flow = d.flow || [];
        setData(flow);
        const sum = flow.reduce((acc: number, curr: FlowPoint) => acc + (curr.visitors || 0), 0);
        setTotalVisitors(sum);
      })
      .catch(() => {
        setData([]);
      });
  }, []);

  return (
    <Card className="bg-zinc-950/50 border-zinc-800">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-base font-medium">Visitor Footfall Flow</CardTitle>
            <CardDescription>Foot-traffic distribution by hour of day</CardDescription>
          </div>
          <span className="text-xs font-mono text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
            Total: {totalVisitors}
          </span>
        </div>
      </CardHeader>
      <CardContent className="h-64">
        {mounted && (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 30, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="hour" stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: "var(--background)", borderColor: "var(--border)", borderRadius: "6px" }}
                cursor={{ fill: 'var(--muted)', opacity: 0.4 }}
              />
              <Bar dataKey="visitors" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
