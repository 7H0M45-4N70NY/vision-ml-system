"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Image as ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export interface TriageFrame {
  id: string;
  timestamp: number;
  reason: string;
  confidence: string | number;
  class: string;
  imageUrl?: string;
}

interface GalleryGridProps {
  frames: TriageFrame[];
  selectedIds: string[];
  onToggleSelection: (id: string, multi: boolean) => void;
  activeId: string | null;
}

export function GalleryGrid({ frames, selectedIds, onToggleSelection, activeId }: GalleryGridProps) {
  return (
    <div className="flex-1 overflow-y-auto pr-2 rounded-lg">
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
        {frames.length === 0 && (
            <div className="col-span-full h-40 flex items-center justify-center text-zinc-500 italic">
                No frames captured yet...
            </div>
        )}
        {frames.map((frame) => {
          const isSelected = selectedIds.includes(frame.id);
          const isActive = activeId === frame.id;
          
          return (
            <div
              key={frame.id}
              role="button"
              tabIndex={0}
              onClick={(e) => onToggleSelection(frame.id, e.shiftKey || e.metaKey || e.ctrlKey)}
              onKeyDown={(e) => {
                 if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onToggleSelection(frame.id, e.shiftKey || e.metaKey || e.ctrlKey);
                 }
              }}
              className={cn(
                "group relative bg-zinc-900 border rounded-md overflow-hidden cursor-pointer select-none transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-950",
                isSelected ? "border-primary ring-1 ring-primary" : "border-zinc-800 hover:border-zinc-700",
                isActive && !isSelected ? "border-zinc-500" : ""
              )}
            >
              {/* Image Rendering */}
              <div className="aspect-video bg-zinc-950 flex flex-col items-center justify-center relative">
                 {frame.imageUrl ? (
                     <img 
                        src={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}${frame.imageUrl}`} 
                        alt={`Triage frame — ${frame.class}, confidence ${frame.confidence}`}
                        className="w-full h-full object-cover"
                     />
                 ) : (
                     <ImageIcon className="w-8 h-8 text-zinc-800 mb-2" />
                 )}
                 <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-t from-zinc-950/80 to-transparent pointer-events-none" />
                 <span className="absolute bottom-1 right-2 text-[10px] text-zinc-400 font-mono tracking-widest">{frame.id}</span>
                 
                 {isSelected && (
                   <div className="absolute top-2 right-2 text-primary bg-zinc-950 rounded-full border border-primary z-10">
                      <CheckCircle2 className="w-5 h-5" />
                   </div>
                 )}
              </div>
              
              <div className="p-3">
                <div className="flex justify-between items-start mb-2">
                  <Badge variant="outline" className={cn(
                    "text-[10px] uppercase tracking-wider py-0 leading-tight",
                    frame.reason.includes("Low Conf") ? "border-amber-500/30 text-amber-400 font-medium" : "border-rose-500/30 text-rose-400 font-medium"
                  )}>
                    {frame.reason}
                  </Badge>
                  <span className="text-xs font-mono text-zinc-500">{frame.confidence}</span>
                </div>
                <div className="text-xs text-zinc-400 flex justify-between">
                  <span>Class: <span className="text-zinc-200">{frame.class}</span></span>
                  <span>{new Date(frame.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
