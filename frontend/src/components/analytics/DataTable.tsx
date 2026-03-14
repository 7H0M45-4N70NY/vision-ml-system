"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const runs = [
  { id: "RUN-8492", time: "10 min ago", source: "cam_01", objects: 145, duration: "3m 12s", status: "Completed" },
  { id: "RUN-8491", time: "45 min ago", source: "file_upload", objects: 89, duration: "1m 45s", status: "Completed" },
  { id: "RUN-8490", time: "2 hrs ago", source: "cam_01", objects: 312, duration: "8m 00s", status: "Drift Alert" },
  { id: "RUN-8489", time: "5 hrs ago", source: "rtsp_front", objects: 540, duration: "15m 30s", status: "Completed" },
];

export function DataTable() {
  return (
    <Card className="bg-zinc-950/50 border-zinc-800 lg:col-span-2">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-base font-medium">Recent Inference Runs</CardTitle>
        <Button variant="outline" size="sm" className="h-8 border-zinc-800 bg-zinc-900 text-zinc-300">
          <Download className="mr-2 h-4 w-4" />
          Export CSV
        </Button>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow className="border-zinc-800 hover:bg-transparent">
              <TableHead className="text-zinc-400">Run ID</TableHead>
              <TableHead className="text-zinc-400">Source</TableHead>
              <TableHead className="text-zinc-400 text-right">Objects</TableHead>
              <TableHead className="text-zinc-400 text-right">Duration</TableHead>
              <TableHead className="text-zinc-400 text-right">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {runs.map((run) => (
              <TableRow key={run.id} className="border-zinc-800/50 hover:bg-zinc-900/50">
                <TableCell className="font-mono text-xs">{run.id}</TableCell>
                <TableCell>{run.source}</TableCell>
                <TableCell className="text-right font-mono">{run.objects}</TableCell>
                <TableCell className="text-right text-zinc-400">{run.duration}</TableCell>
                <TableCell className="text-right">
                  <Badge variant="outline" className={run.status === "Completed" ? "border-emerald-500/30 text-emerald-400" : "border-rose-500/30 text-rose-400"}>
                    {run.status}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
