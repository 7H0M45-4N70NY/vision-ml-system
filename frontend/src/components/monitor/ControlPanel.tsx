"use client";

import { useStore } from "@/store/useStore";
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Camera, Film, Radio, Play, Square, Power } from "lucide-react";

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
    turnOffStream,
    turnOnStream,
    availableVideos,
    fetchSources,
    setThresholds,
    setDualMode,
  } = useStore();

  useEffect(() => {
    fetchSources();
  }, [fetchSources]);

  // Determine active tab: derive from streamSource unless user clicked another tab
  const isWebcam = (streamSource === "0" || streamSource === "1");
  const isRtsp = streamSource.startsWith("rtsp://") || streamSource.startsWith("http://");
  const derivedTab = isWebcam ? "webcam" : isRtsp ? "rtsp" : "video";
  const [tabOverride, setTabOverride] = useState<string | null>(null);
  const activeTab = tabOverride ?? derivedTab;

  const [selectedCamera, setSelectedCamera] = useState("0");
  const [selectedVideo, setSelectedVideo] = useState("data/sample_videos/video_1.mp4");
  const [customVideoPath, setCustomVideoPath] = useState("");
  const [rtspUrl, setRtspUrl] = useState("rtsp://localhost:8554/live");
  const [conf, setConf] = useState(0.5);
  const [iou, setIou] = useState(0.45);

  const isWebcamActive = (streamSource === "0" || streamSource === "1");
  const isVideoActive = !isWebcamActive && !streamSource.startsWith("rtsp://") && !streamSource.startsWith("http://");
  const isRtspActive = streamSource.startsWith("rtsp://") || streamSource.startsWith("http://");

  const effectiveVideoPath = selectedVideo === "custom" ? customVideoPath.trim() : selectedVideo;

  const handleSelectTab = (tab: string) => {
    setTabOverride(tab);
  };

  const handleSwitchSource = (src: string) => {
    setTabOverride(null);
    setStreamSource(src);
  };

  const handleTurnOn = (src: string) => {
    setTabOverride(null);
    turnOnStream(src);
  };

  return (
    <div className="flex flex-col gap-4 w-full xl:w-84 shrink-0 min-h-0 xl:h-full overflow-y-auto pr-1">
      {/* ── Video Source & Stream Power ── */}
      <Card className="bg-zinc-950/50 border-zinc-800 shrink-0">
        <CardHeader className="pb-3 border-b border-zinc-800/60 flex flex-row items-center justify-between">
          <div>
            <CardTitle className="text-sm font-semibold tracking-tight text-zinc-100">Video Sources</CardTitle>
            <div className="text-[11px] text-zinc-400 mt-0.5">Switch & individually control feeds</div>
          </div>
          <div className="flex items-center gap-1.5 bg-zinc-900 border border-zinc-800 px-2.5 py-1 rounded-full">
            <div className={`h-2 w-2 rounded-full ${streamActive ? "bg-emerald-400 animate-pulse" : "bg-zinc-600"}`} />
            <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-300">
              {streamActive ? "Online" : "Off"}
            </span>
          </div>
        </CardHeader>
        <CardContent className="pt-4 space-y-4">
          <Tabs value={activeTab} onValueChange={handleSelectTab} className="w-full">
            <TabsList className="grid grid-cols-3 w-full bg-zinc-900 border border-zinc-800 p-1">
              <TabsTrigger value="webcam" className="text-xs gap-1.5 py-1.5">
                <Camera className="h-3.5 w-3.5" /> Webcam
              </TabsTrigger>
              <TabsTrigger value="video" className="text-xs gap-1.5 py-1.5">
                <Film className="h-3.5 w-3.5" /> Video
              </TabsTrigger>
              <TabsTrigger value="rtsp" className="text-xs gap-1.5 py-1.5">
                <Radio className="h-3.5 w-3.5" /> RTSP
              </TabsTrigger>
            </TabsList>

            {/* ── Webcam Panel ── */}
            <TabsContent value="webcam" className="space-y-3 pt-3">
              <div className="space-y-1.5">
                <Label className="text-xs text-zinc-400">Camera Device</Label>
                <Select value={selectedCamera} onValueChange={(v) => { if (v) setSelectedCamera(v); }}>
                  <SelectTrigger className="w-full bg-zinc-900 border-zinc-800 text-xs">
                    <SelectValue placeholder="Select Camera" />
                  </SelectTrigger>
                  <SelectContent className="bg-zinc-900 border-zinc-800 text-zinc-200">
                    <SelectItem value="0">Camera 0 (Default / Integrated)</SelectItem>
                    <SelectItem value="1">Camera 1 (Secondary USB)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="pt-1 flex gap-2">
                {isWebcamActive && streamSource === selectedCamera ? (
                  streamActive ? (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={turnOffStream}
                      className="w-full text-xs gap-1.5 bg-rose-600/90 hover:bg-rose-600"
                    >
                      <Square className="h-3.5 w-3.5 fill-current" /> Turn Off Webcam
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      onClick={() => handleTurnOn(selectedCamera)}
                      className="w-full text-xs gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white"
                    >
                      <Play className="h-3.5 w-3.5 fill-current" /> Turn On Webcam
                    </Button>
                  )
                ) : (
                  <Button
                    size="sm"
                    onClick={() => handleSwitchSource(selectedCamera)}
                    className="w-full text-xs gap-1.5 bg-blue-600 hover:bg-blue-500 text-white"
                  >
                    <Play className="h-3.5 w-3.5 fill-current" /> Switch to Webcam & On
                  </Button>
                )}
              </div>
            </TabsContent>

            {/* ── Video File Panel ── */}
            <TabsContent value="video" className="space-y-3 pt-3">
              <div className="space-y-1.5">
                <Label className="text-xs text-zinc-400">Sample Video / File Path</Label>
                <Select value={selectedVideo} onValueChange={(v) => { if (v) setSelectedVideo(v); }}>
                  <SelectTrigger className="w-full bg-zinc-900 border-zinc-800 text-xs truncate">
                    <SelectValue placeholder="Select Video File" />
                  </SelectTrigger>
                  <SelectContent className="bg-zinc-900 border-zinc-800 text-zinc-200">
                    {availableVideos.map((v) => (
                      <SelectItem key={v.path} value={v.path}>
                        {v.name}
                      </SelectItem>
                    ))}
                    <SelectItem value="custom">Custom File Path...</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {selectedVideo === "custom" && (
                <div className="space-y-1">
                  <Input
                    placeholder="e.g. data/sample_videos/video_1.mp4"
                    value={customVideoPath}
                    onChange={(e) => setCustomVideoPath(e.target.value)}
                    className="bg-zinc-900 border-zinc-800 text-xs"
                  />
                </div>
              )}

              <div className="pt-1 flex gap-2">
                {isVideoActive && streamSource === effectiveVideoPath ? (
                  streamActive ? (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={turnOffStream}
                      className="w-full text-xs gap-1.5 bg-rose-600/90 hover:bg-rose-600"
                    >
                      <Square className="h-3.5 w-3.5 fill-current" /> Turn Off Video
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      onClick={() => handleTurnOn(effectiveVideoPath)}
                      className="w-full text-xs gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white"
                    >
                      <Play className="h-3.5 w-3.5 fill-current" /> Turn On Video
                    </Button>
                  )
                ) : (
                  <Button
                    size="sm"
                    disabled={!effectiveVideoPath}
                    onClick={() => handleSwitchSource(effectiveVideoPath)}
                    className="w-full text-xs gap-1.5 bg-blue-600 hover:bg-blue-500 text-white"
                  >
                    <Play className="h-3.5 w-3.5 fill-current" /> Switch to Video & Play
                  </Button>
                )}
              </div>
            </TabsContent>

            {/* ── RTSP Stream Panel ── */}
            <TabsContent value="rtsp" className="space-y-3 pt-3">
              <div className="space-y-1.5">
                <Label className="text-xs text-zinc-400">RTSP / HTTP Stream URL</Label>
                <Input
                  placeholder="rtsp://192.168.1.100:8554/live"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  className="bg-zinc-900 border-zinc-800 text-xs font-mono"
                />
              </div>

              <div className="flex items-center gap-1.5">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => setRtspUrl("rtsp://localhost:8554/live")}
                  className="text-[11px] h-6 px-2 bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-zinc-200"
                >
                  Local Preset
                </Button>
              </div>

              <div className="pt-1 flex gap-2">
                {isRtspActive && streamSource === rtspUrl.trim() ? (
                  streamActive ? (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={turnOffStream}
                      className="w-full text-xs gap-1.5 bg-rose-600/90 hover:bg-rose-600"
                    >
                      <Square className="h-3.5 w-3.5 fill-current" /> Turn Off RTSP
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      onClick={() => handleTurnOn(rtspUrl.trim())}
                      className="w-full text-xs gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white"
                    >
                      <Play className="h-3.5 w-3.5 fill-current" /> Turn On RTSP
                    </Button>
                  )
                ) : (
                  <Button
                    size="sm"
                    disabled={!rtspUrl.trim()}
                    onClick={() => handleSwitchSource(rtspUrl.trim())}
                    className="w-full text-xs gap-1.5 bg-blue-600 hover:bg-blue-500 text-white"
                  >
                    <Play className="h-3.5 w-3.5 fill-current" /> Switch to RTSP & Connect
                  </Button>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* ── Model & Pipeline Configuration ── */}
      <Card className="bg-zinc-950/50 border-zinc-800 shrink-0">
        <CardHeader className="pb-3 border-b border-zinc-800/60">
          <CardTitle className="text-sm font-semibold tracking-tight text-zinc-100">Model Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-5 pt-4">
          <div className="space-y-3 border-b border-zinc-800/50 pb-4">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-200">Enable Detection</Label>
              <Switch checked={enableDetection} onCheckedChange={(v) => setToggle("enableDetection", v)} />
            </div>
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-200">Enable Tracking</Label>
              <Switch checked={enableTracking} onCheckedChange={(v) => setToggle("enableTracking", v)} />
            </div>
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-200">Show Annotations</Label>
              <Switch checked={showAnnotations} onCheckedChange={(v) => setToggle("showAnnotations", v)} />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <Label className="text-xs text-zinc-400">Dual-Detector Backoff</Label>
            <Switch id="dual-mode" checked={dualMode} onCheckedChange={(v) => setDualMode(v)} />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-400">YOLO Confidence</Label>
              <span className="text-xs font-mono text-zinc-300">{conf.toFixed(2)}</span>
            </div>
            <Slider
              value={[conf]}
              onValueChange={(v) => {
                const val = Array.isArray(v) ? v[0] : v;
                setConf(val);
                setThresholds(val, undefined);
              }}
              max={1}
              step={0.05}
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-400">IoU Threshold</Label>
              <span className="text-xs font-mono text-zinc-300">{iou.toFixed(2)}</span>
            </div>
            <Slider
              value={[iou]}
              onValueChange={(v) => {
                const val = Array.isArray(v) ? v[0] : v;
                setIou(val);
                setThresholds(undefined, val);
              }}
              max={1}
              step={0.05}
            />
          </div>
        </CardContent>
      </Card>

      {/* ── Master Stream Power ── */}
      <Card className="bg-zinc-950/50 border-zinc-800 mt-auto shrink-0">
        <CardContent className="p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[10px] text-zinc-500 font-mono uppercase tracking-wider">Stream Power</div>
              <div className="text-xs text-zinc-300 font-medium">
                {streamActive ? "Capturing & Inferring" : "Processing Suspended"}
              </div>
            </div>
            <Switch
              checked={streamActive}
              onCheckedChange={(v) => (v ? turnOnStream() : turnOffStream())}
            />
          </div>

          <Button
            onClick={() => (streamActive ? turnOffStream() : turnOnStream())}
            className={`w-full text-xs font-medium gap-2 ${
              streamActive
                ? "bg-rose-600 hover:bg-rose-500 text-white"
                : "bg-emerald-600 hover:bg-emerald-500 text-white"
            }`}
          >
            <Power className="h-3.5 w-3.5" />
            {streamActive ? "Stop Stream & Release Capture" : "Start Stream Processing"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
