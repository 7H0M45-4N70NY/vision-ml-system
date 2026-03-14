"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export function ControlPanel() {
  return (
    <div className="flex flex-col gap-4 w-full xl:w-80 shrink-0 overflow-y-visible xl:overflow-y-auto pr-2">
      <Card className="bg-zinc-950/50 border-zinc-800">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Source Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label className="text-xs text-zinc-400">Input Source</Label>
            <Select defaultValue="webcam">
              <SelectTrigger className="w-full bg-zinc-900 border-zinc-800 text-sm">
                <SelectValue placeholder="Select Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="webcam">Webcam (0)</SelectItem>
                <SelectItem value="rtsp">RTSP Stream</SelectItem>
                <SelectItem value="file">Local File</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-950/50 border-zinc-800">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Model Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Dual-Detector Mode</Label>
            <Switch id="dual-mode" defaultChecked />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-400">YOLO Confidence</Label>
              <span className="text-xs font-medium text-zinc-300">0.50</span>
            </div>
            <Slider defaultValue={[0.5]} max={1} step={0.05} />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-400">IoU Threshold</Label>
              <span className="text-xs font-medium text-zinc-300">0.45</span>
            </div>
            <Slider defaultValue={[0.45]} max={1} step={0.05} />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-950/50 border-zinc-800 mt-auto">
         <CardContent className="p-4">
            <div className="text-xs text-zinc-500 mb-2 font-mono uppercase tracking-wider">Status</div>
            <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-sm text-zinc-300">System Ready</span>
            </div>
         </CardContent>
      </Card>
    </div>
  );
}
