"use client";

import { useState, useCallback, useEffect } from "react";
import { FilterBar } from "@/components/triage/FilterBar";
import { GalleryGrid, mockFrames } from "@/components/triage/GalleryGrid";
import { InspectorPanel } from "@/components/triage/InspectorPanel";
import { toast } from "sonner";

export default function TriagePage() {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [activeId, setActiveId] = useState<string | null>(mockFrames[0]?.id || null);

  const handleToggleSelection = useCallback((id: string, multi: boolean) => {
    setActiveId(id);
    if (!multi) {
      setSelectedIds([id]);
      return;
    }
    setSelectedIds(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!activeId && selectedIds.length === 0) return;
      
      const targetIds = selectedIds.length > 0 ? selectedIds : (activeId ? [activeId] : []);
      
      if (e.key === 'x' || e.key === 'X') {
        toast.error(`Discarded ${targetIds.length} frames`);
        // Remove from local state mock logic
      } else if (e.key === 'k' || e.key === 'K') {
        toast.success(`Accepted ${targetIds.length} frames`);
      } else if (e.key === 'Enter') {
        toast.info(`Sent ${targetIds.length} frames to labeling`);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeId, selectedIds]);

  return (
    <div className="flex flex-col h-full w-full p-4 overflow-hidden">
      <div className="mb-2">
        <h2 className="text-xl font-bold tracking-tight">Active Learning Triage</h2>
        <p className="text-sm text-zinc-400">Review low-confidence detections and anomalies before retraining.</p>
      </div>
      
      <FilterBar />

      <div className="flex flex-1 min-h-0 gap-4">
         {/* Main Gallery Area */}
         <GalleryGrid 
           selectedIds={selectedIds} 
           onToggleSelection={handleToggleSelection} 
           activeId={activeId}
         />

         {/* Sidebar Inspector */}
         <InspectorPanel activeId={activeId} />
      </div>
    </div>
  );
}
