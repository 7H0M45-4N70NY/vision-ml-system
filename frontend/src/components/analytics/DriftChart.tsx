"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const data = [
  { date: "Oct 1", score: 0.12 },
  { date: "Oct 2", score: 0.14 },
  { date: "Oct 3", score: 0.18 },
  { date: "Oct 4", score: 0.15 },
  { date: "Oct 5", score: 0.22 },
  { date: "Oct 6", score: 0.28 }, // Drift starts increasing
  { date: "Oct 7", score: 0.35 },
  { date: "Oct 8", score: 0.45 },
  { date: "Oct 9", score: 0.42 },
];

export function DriftChart() {
  return (
    <Card className="bg-zinc-950/50 border-zinc-800">
      <CardHeader>
        <CardTitle className="text-base font-medium">Model Drift Trends</CardTitle>
        <CardDescription>Average domain shift score over the last 10 days</CardDescription>
      </CardHeader>
      <CardContent className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--destructive)" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="var(--destructive)" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey="date" stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
            <YAxis stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ backgroundColor: "var(--background)", borderColor: "var(--border)", borderRadius: "6px" }}
              itemStyle={{ color: "var(--destructive)", fontWeight: "bold" }}
            />
            <Area type="monotone" dataKey="score" stroke="var(--destructive)" fillOpacity={1} fill="url(#colorScore)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
