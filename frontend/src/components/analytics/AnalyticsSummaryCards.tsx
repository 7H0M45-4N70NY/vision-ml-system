"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Users, Clock, Video, Activity } from "lucide-react";

interface AnalyticsSummary {
  total_runs: number;
  total_visitors: number;
  avg_dwell_time_seconds: number;
  total_frames: number;
  total_labeling_events: number;
  total_training_events: number;
}

export function AnalyticsSummaryCards() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/analytics/stats`)
      .then(res => res.json())
      .then(data => setSummary(data))
      .catch(() => {});
  }, []);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <Card className="bg-zinc-950/50 border-zinc-800">
        <CardContent className="p-4 flex items-center justify-between">
          <div>
            <div className="text-xs text-zinc-500 font-medium mb-1">Total Unique Visitors</div>
            <div className="text-2xl font-bold text-emerald-400 font-mono">
              {summary ? summary.total_visitors : "—"}
            </div>
            <div className="text-[10px] text-zinc-500 mt-0.5">Persisted track records</div>
          </div>
          <div className="h-10 w-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Users className="h-5 w-5" />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-950/50 border-zinc-800">
        <CardContent className="p-4 flex items-center justify-between">
          <div>
            <div className="text-xs text-zinc-500 font-medium mb-1">Average Dwell Time</div>
            <div className="text-2xl font-bold text-zinc-100 font-mono">
              {summary ? `${summary.avg_dwell_time_seconds.toFixed(2)}s` : "—"}
            </div>
            <div className="text-[10px] text-zinc-500 mt-0.5">Calculated across all tracks</div>
          </div>
          <div className="h-10 w-10 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
            <Clock className="h-5 w-5" />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-950/50 border-zinc-800">
        <CardContent className="p-4 flex items-center justify-between">
          <div>
            <div className="text-xs text-zinc-500 font-medium mb-1">Total Frames Processed</div>
            <div className="text-2xl font-bold text-zinc-100 font-mono">
              {summary ? summary.total_frames.toLocaleString() : "—"}
            </div>
            <div className="text-[10px] text-zinc-500 mt-0.5">Inference pipeline throughput</div>
          </div>
          <div className="h-10 w-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
            <Video className="h-5 w-5" />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-950/50 border-zinc-800">
        <CardContent className="p-4 flex items-center justify-between">
          <div>
            <div className="text-xs text-zinc-500 font-medium mb-1">Completed Runs</div>
            <div className="text-2xl font-bold text-zinc-100 font-mono">
              {summary ? summary.total_runs : "—"}
            </div>
            <div className="text-[10px] text-zinc-500 mt-0.5">Offline runs + live sessions</div>
          </div>
          <div className="h-10 w-10 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
            <Activity className="h-5 w-5" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
