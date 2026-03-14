"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";

const data = [
  { name: "Person", value: 8500 },
  { name: "Vehicle", value: 3400 },
  { name: "Backpack", value: 1250 },
  { name: "Bicycle", value: 640 },
];

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];

export function ClassDistribution() {
  return (
    <Card className="bg-zinc-950/50 border-zinc-800 lg:col-span-2">
      <CardHeader>
        <CardTitle className="text-base font-medium">Class Distribution</CardTitle>
        <CardDescription>All-time detected objects</CardDescription>
      </CardHeader>
      <CardContent className="h-64 flex items-center justify-center">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
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
      </CardContent>
    </Card>
  );
}
