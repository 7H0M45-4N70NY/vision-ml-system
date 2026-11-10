"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Download, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface RunItem {
  run_id: string;
  timestamp: string;
  source_type: string;
  unique_visitors: number;
  duration_seconds: number;
  total_frames: number;
  avg_dwell_time_seconds: number;
  avg_confidence: number;
  status: string;
}

export function DataTable() {
  const [runs, setRuns] = useState<RunItem[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchRuns = () => {
    setLoading(true);
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/analytics/runs`)
      .then(res => res.json())
      .then(data => {
        setRuns(data.runs || []);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  const exportCSV = () => {
    if (!runs.length) return;
    const headers = ["Run ID", "Timestamp", "Source", "Unique Visitors", "Avg Dwell (s)", "Total Frames", "Duration (s)", "Status"];
    const rows = runs.map(r => [
      r.run_id,
      r.timestamp,
      r.source_type,
      r.unique_visitors,
      r.avg_dwell_time_seconds,
      r.total_frames,
      r.duration_seconds,
      r.status,
    ]);
    const csvContent = [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `vision_ml_runs_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Card className="bg-zinc-950/50 border-zinc-800 lg:col-span-2">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-medium">Recent Inference Runs</CardTitle>
          <p className="text-xs text-zinc-500 mt-0.5">Persisted in SQLite database (`data/analytics.db`)</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={fetchRuns} className="h-8 w-8 p-0 text-zinc-400 hover:text-zinc-100">
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
          <Button variant="outline" size="sm" onClick={exportCSV} disabled={!runs.length} className="h-8 border-zinc-800 bg-zinc-900 text-zinc-300">
            <Download className="mr-2 h-4 w-4" />
            Export CSV
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {runs.length === 0 ? (
          <div className="py-8 text-center text-sm text-zinc-500">
            {loading ? "Loading inference history..." : "No inference runs recorded yet. Run `python scripts/inference.py` or start a live stream."}
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="border-zinc-800 hover:bg-transparent">
                <TableHead className="text-zinc-400">Run ID</TableHead>
                <TableHead className="text-zinc-400">Timestamp</TableHead>
                <TableHead className="text-zinc-400">Source</TableHead>
                <TableHead className="text-zinc-400 text-right">Visitors</TableHead>
                <TableHead className="text-zinc-400 text-right">Avg Dwell</TableHead>
                <TableHead className="text-zinc-400 text-right">Frames</TableHead>
                <TableHead className="text-zinc-400 text-right">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {runs.map((run) => (
                <TableRow key={run.run_id} className="border-zinc-800/50 hover:bg-zinc-900/50">
                  <TableCell className="font-mono text-xs">{run.run_id}</TableCell>
                  <TableCell className="text-xs text-zinc-400">{run.timestamp}</TableCell>
                  <TableCell className="text-xs">{run.source_type}</TableCell>
                  <TableCell className="text-right font-mono text-emerald-400 font-medium">{run.unique_visitors}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{run.avg_dwell_time_seconds ? `${run.avg_dwell_time_seconds.toFixed(2)}s` : '0.0s'}</TableCell>
                  <TableCell className="text-right text-zinc-400 font-mono text-xs">{run.total_frames}</TableCell>
                  <TableCell className="text-right">
                    <Badge variant="outline" className={run.status === "completed" ? "border-emerald-500/30 text-emerald-400" : "border-rose-500/30 text-rose-400"}>
                      {run.status}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
