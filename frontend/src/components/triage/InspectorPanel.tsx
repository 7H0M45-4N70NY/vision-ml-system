"use client";

import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Check, X, Tag, Code } from "lucide-react";
import { mockFrames } from "./GalleryGrid";

interface InspectorPanelProps {
  activeId: string | null;
}

export function InspectorPanel({ activeId }: InspectorPanelProps) {
  const frame = mockFrames.find(f => f.id === activeId);

  if (!frame) {
    return (
      <Card className="w-80 h-full bg-zinc-950/50 border-zinc-800 flex flex-col items-center justify-center text-zinc-500">
         <Code className="w-12 h-12 mb-4 text-zinc-800" />
         <p className="text-sm">Select a frame to inspect</p>
      </Card>
    );
  }

  const mockJsonMetadata = {
    id: frame.id,
    timestamp: frame.timestamp,
    reason: frame.reason,
    trigger_confidence: parseFloat(frame.confidence),
    detections: [
      { class: frame.class, conf: parseFloat(frame.confidence), bbox: [120, 45, 230, 400] }
    ],
    telemetry: {
      fps_at_capture: 28.5,
      camera_id: "cam_01",
      lux_sensor: 450
    }
  };

  return (
    <Card className="w-80 h-full flex flex-col bg-zinc-950/50 border-zinc-800 shrink-0">
      <CardHeader className="py-4 border-b border-zinc-800/50">
        <CardTitle className="text-sm font-medium">Inspector</CardTitle>
      </CardHeader>
      
      <ScrollArea className="flex-1 p-0">
        <div className="p-4 space-y-6">
          <div className="aspect-video bg-zinc-900 rounded-md border border-zinc-800 flex items-center justify-center relative overflow-hidden">
             {/* Thumbnail */}
             <div className="absolute inset-x-[15%] inset-y-[10%] border-2 border-primary border-dashed">
                 <div className="absolute -top-6 left-0 bg-primary text-primary-foreground text-[10px] px-1 font-mono">
                    {frame.class} {frame.confidence}
                 </div>
             </div>
          </div>

          <div className="space-y-3">
             <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Metadata</h4>
             <div className="bg-zinc-950 rounded-md p-3 border border-zinc-800 overflow-x-auto">
               <pre className="text-[10px] text-zinc-300 font-mono leading-relaxed">
                 {JSON.stringify(mockJsonMetadata, null, 2)}
               </pre>
             </div>
          </div>
        </div>
      </ScrollArea>

      <CardFooter className="flex-col gap-2 p-4 border-t border-zinc-800/50">
          <div className="flex gap-2 w-full">
            <Button variant="outline" className="flex-1 border-rose-900/50 bg-rose-950/20 text-rose-400 hover:bg-rose-950/40 hover:text-rose-300" size="sm">
               <X className="w-4 h-4 mr-1.5" /> Discard (X)
            </Button>
            <Button variant="outline" className="flex-1 border-emerald-900/50 bg-emerald-950/20 text-emerald-400 hover:bg-emerald-950/40 hover:text-emerald-300" size="sm">
               <Check className="w-4 h-4 mr-1.5" /> Accept (K)
            </Button>
          </div>
          <Button variant="outline" className="w-full bg-primary/10 border-primary/30 text-primary hover:bg-primary/20 hover:text-primary transition-colors" size="sm">
             <Tag className="w-4 h-4 mr-1.5" /> Send to Labeling (Enter)
          </Button>
      </CardFooter>
    </Card>
  );
}
