"use client";

import { useStore } from "@/store/useStore";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export function ControlPanel() {
  const {
      enableDetection,
      enableTracking,
      showAnnotations,
      streamActive,
      dualMode,
      setToggle,
      streamSource,
      setStreamSource,
      setThresholds,
      setDualMode
  } = useStore();
  
  // Local state for the configuration form
  const [sourceType, setSourceType] = useState(() => {
    if (streamSource === "0") return "webcam";
    if (streamSource.startsWith("rtsp://") || streamSource.startsWith("http://")) return "rtsp";
    return "file";
  });
  const [customUrl, setCustomUrl] = useState(streamSource !== "0" ? streamSource : "");
  const [conf, setConf] = useState(0.5);
  const [iou, setIou] = useState(0.45);

  const handleApplySource = () => {
      setStreamSource(sourceType === "webcam" ? "0" : customUrl);
  };

  return (
    <div className="flex flex-col gap-4 w-full xl:w-80 shrink-0 min-h-0 xl:h-full overflow-y-auto pr-2">
      <Card className="bg-zinc-950/50 border-zinc-800 shrink-0">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Source Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4">
            <div className="space-y-2">
                <Label className="text-xs text-zinc-400">Input Source Type</Label>
                <Select value={sourceType} onValueChange={(v) => { if (v) setSourceType(v); }}>
                <SelectTrigger className="w-full bg-zinc-900 border-zinc-800 text-sm">
                    <SelectValue placeholder="Select Source" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="webcam">Webcam (0)</SelectItem>
                    <SelectItem value="rtsp">RTSP / HTTP Stream</SelectItem>
                    <SelectItem value="file">Local File Path</SelectItem>
                </SelectContent>
                </Select>
            </div>
            
            {sourceType !== "webcam" && (
                <div className="space-y-2">
                    <Label className="text-xs text-zinc-400">Source URL / Path</Label>
                    <Input 
                        placeholder={sourceType === "rtsp" ? "rtsp://192.168.1.100/stream" : "/path/to/video.mp4"}
                        value={customUrl}
                        onChange={(e) => setCustomUrl(e.target.value)}
                        className="bg-zinc-900 border-zinc-800 text-sm"
                    />
                </div>
            )}
            
            <Button 
                onClick={handleApplySource} 
                className="w-full text-xs" 
                variant="secondary"
                disabled={sourceType !== "webcam" && customUrl.trim() === ""}
            >
                Apply Stream Source
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-950/50 border-zinc-800 shrink-0">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Model Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4 border-b border-zinc-800/50 pb-4">
              <div className="flex items-center justify-between">
                <Label className="text-sm">Enable Detection</Label>
                <Switch checked={enableDetection} onCheckedChange={(v) => setToggle("enableDetection", v)} />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-sm">Enable Tracking</Label>
                <Switch checked={enableTracking} onCheckedChange={(v) => setToggle("enableTracking", v)} />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-sm">Show Annotations</Label>
                <Switch checked={showAnnotations} onCheckedChange={(v) => setToggle("showAnnotations", v)} />
              </div>
          </div>

          <div className="flex items-center justify-between">
            <Label className="text-sm text-zinc-400">Dual-Detector Backoff</Label>
            <Switch id="dual-mode" checked={dualMode} onCheckedChange={(v) => setDualMode(v)} />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-400">YOLO Confidence</Label>
              <span className="text-xs font-medium text-zinc-300">{conf.toFixed(2)}</span>
            </div>
            <Slider value={[conf]} onValueChange={(v) => { const val = Array.isArray(v) ? v[0] : v; setConf(val); setThresholds(val, undefined); }} max={1} step={0.05} />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-400">IoU Threshold</Label>
              <span className="text-xs font-medium text-zinc-300">{iou.toFixed(2)}</span>
            </div>
            <Slider value={[iou]} onValueChange={(v) => { const val = Array.isArray(v) ? v[0] : v; setIou(val); setThresholds(undefined, val); }} max={1} step={0.05} />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-950/50 border-zinc-800 mt-auto shrink-0">
         <CardContent className="p-4 space-y-4">
            <div>
                <div className="text-xs text-zinc-500 mb-2 font-mono uppercase tracking-wider">Status</div>
                <div className="flex items-center gap-2">
                    <div className={`h-2 w-2 rounded-full ${streamActive ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
                    <span className="text-sm text-zinc-300">{streamActive ? 'System Ready' : 'Stream Stopped'}</span>
                </div>
            </div>
            
            <Button 
                onClick={() => setToggle('streamActive', !streamActive)} 
                className={`w-full text-xs font-medium ${streamActive ? 'bg-rose-500 hover:bg-rose-600 text-white' : 'bg-emerald-500 hover:bg-emerald-600 text-white'}`}
            >
                {streamActive ? '■ Stop Stream' : '▶ Start Stream'}
            </Button>
         </CardContent>
      </Card>
    </div>
  );
}
