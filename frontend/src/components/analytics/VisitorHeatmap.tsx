"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const data = [
  { hour: "8am", visitors: 45 },
  { hour: "10am", visitors: 85 },
  { hour: "12pm", visitors: 142 },
  { hour: "2pm", visitors: 110 },
  { hour: "4pm", visitors: 90 },
  { hour: "6pm", visitors: 165 },
  { hour: "8pm", visitors: 30 },
];

export function VisitorHeatmap() {
  return (
    <Card className="bg-zinc-950/50 border-zinc-800">
      <CardHeader>
        <CardTitle className="text-base font-medium">Visitor Flow (Today)</CardTitle>
        <CardDescription>Peak hours by detection count</CardDescription>
      </CardHeader>
      <CardContent className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 30, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey="hour" stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
            <YAxis stroke="var(--muted-foreground)" fontSize={12} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ backgroundColor: "var(--background)", borderColor: "var(--border)", borderRadius: "6px" }}
              cursor={{ fill: 'var(--muted)', opacity: 0.4 }}
            />
            <Bar dataKey="visitors" fill="var(--primary)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
