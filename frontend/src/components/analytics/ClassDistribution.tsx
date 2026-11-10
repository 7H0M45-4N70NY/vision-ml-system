"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { useMounted } from "@/hooks/useMounted";

interface TierItem {
  name: string;
  value: number;
}

const COLORS = ["#60a5fa", "#34d399", "#fbbf24", "#f87171"];

export function ClassDistribution() {
  const [data, setData] = useState<TierItem[]>([]);
  const [total, setTotal] = useState<number>(0);
  const mounted = useMounted();

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/analytics/dwell-distribution`)
      .then(res => res.json())
      .then(res => {
        setData(res.distribution || []);
        setTotal(res.total || 0);
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
            <CardTitle className="text-base font-medium">Visitor Engagement by Dwell Time</CardTitle>
            <CardDescription>Segmented from verified tracking duration</CardDescription>
          </div>
          <span className="text-xs font-mono text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
            {total} tracked
          </span>
        </div>
      </CardHeader>
      <CardContent className="h-64 flex items-center justify-center">
        {mounted && (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={75}
                paddingAngle={4}
                dataKey="value"
                stroke="none"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ backgroundColor: "var(--background)", borderColor: "var(--border)", borderRadius: "6px" }} 
                itemStyle={{ color: "var(--foreground)" }}
              />
              <Legend verticalAlign="bottom" height={36} iconType="circle" />
            </PieChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
