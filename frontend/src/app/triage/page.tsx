"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import { FilterBar, type TriageFilters } from "@/components/triage/FilterBar";
import { GalleryGrid, type TriageFrame } from "@/components/triage/GalleryGrid";
import { InspectorPanel } from "@/components/triage/InspectorPanel";
import { toast } from "sonner";

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const POLL_INTERVAL = 10_000; // refresh every 10s

export default function TriagePage() {
  const [frames, setFrames] = useState<TriageFrame[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [filters, setFilters] = useState<TriageFilters>({
    search: "",
    reason: "all",
    classFilter: "all",
  });

  // ── Data fetching ──────────────────────────────────────────────
  const fetchFrames = useCallback(async (silent = false) => {
    try {
      const res = await fetch(`${API}/triage/frames`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setFrames(data.frames || []);
      setActiveId(prev => {
        // Keep the current selection if it still exists, else pick the first frame
        const ids = (data.frames || []).map((f: TriageFrame) => f.id);
        if (prev && ids.includes(prev)) return prev;
        return ids[0] ?? null;
      });
    } catch (_e) {
      if (!silent) toast.error("Failed to load triage frames — is the API running?");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFrames();
    const timer = setInterval(() => fetchFrames(true), POLL_INTERVAL);
    return () => clearInterval(timer);
  }, [fetchFrames]);

  // ── Filtering ─────────────────────────────────────────────────
  const filteredFrames = useMemo(() => {
    return frames.filter(f => {
      if (filters.classFilter !== "all" && f.class.toLowerCase() !== filters.classFilter) return false;
      if (filters.reason !== "all") {
        const reasonMap: Record<string, string[]> = {
          "low-conf":  ["low confidence", "low conf"],
          "ambiguous": ["high iou", "overlap"],
          "anomaly":   ["anomaly", "tracker"],
        };
        const keywords = reasonMap[filters.reason] ?? [];
        if (!keywords.some(k => f.reason.toLowerCase().includes(k))) return false;
      }
      if (filters.search) {
        const q = filters.search.toLowerCase();
        if (!f.id.toLowerCase().includes(q) &&
            !f.class.toLowerCase().includes(q) &&
            !f.reason.toLowerCase().includes(q)) return false;
      }
      return true;
    });
  }, [frames, filters]);

  // ── Actions ───────────────────────────────────────────────────
  const performAction = useCallback(async (ids: string[], action: "accept" | "reject" | "label") => {
    try {
      const res = await fetch(`${API}/triage/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame_ids: ids, action }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();

      // Remove actioned frames from local state
      setFrames(prev => prev.filter(f => !ids.includes(f.id)));
      setSelectedIds(prev => prev.filter(id => !ids.includes(id)));

      const messages: Record<string, string> = {
        reject: `Discarded ${result.processed} frame(s)`,
        accept: `Accepted ${result.processed} frame(s) → auto_labeled/`,
        label:  `Sent ${result.processed} frame(s) to labeling queue`,
      };
      if (action === "reject") toast.error(messages[action]);
      else toast.success(messages[action]);

      if (result.errors?.length > 0) {
        toast.warning(`${result.errors.length} frame(s) failed`);
      }
    } catch (_e) {
      toast.error("Action failed — check API connection");
    }
  }, []);

  // ── Selection ─────────────────────────────────────────────────
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

  // ── Keyboard shortcuts ────────────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't fire when typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (!activeId && selectedIds.length === 0) return;

      const targetIds = selectedIds.length > 0 ? selectedIds : (activeId ? [activeId] : []);

      if (e.key === 'x' || e.key === 'X') {
        performAction(targetIds, "reject");
      } else if (e.key === 'k' || e.key === 'K') {
        performAction(targetIds, "accept");
      } else if (e.key === 'Enter') {
        performAction(targetIds, "label");
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeId, selectedIds, performAction]);

  const activeFrame = filteredFrames.find(f => f.id === activeId) ||
                      frames.find(f => f.id === activeId) ||
                      null;

  return (
    <div className="flex flex-col h-full w-full p-4 overflow-hidden">
      <div className="mb-2">
        <h2 className="text-xl font-bold tracking-tight">Active Learning Triage</h2>
        <p className="text-sm text-zinc-400">
          Review low-confidence detections. Press <kbd className="px-1 py-0.5 text-xs bg-zinc-800 rounded">X</kbd> discard,{" "}
          <kbd className="px-1 py-0.5 text-xs bg-zinc-800 rounded">K</kbd> accept,{" "}
          <kbd className="px-1 py-0.5 text-xs bg-zinc-800 rounded">Enter</kbd> send to labeling.
        </p>
      </div>

      <FilterBar
        filters={filters}
        totalFrames={frames.length}
        filteredCount={filteredFrames.length}
        onFiltersChange={setFilters}
      />

      <div className="flex flex-1 min-h-0 gap-4">
        {/* Main Gallery Area */}
        {loading ? (
          <div className="flex-1 flex items-center justify-center text-zinc-500">
            Scanning frame storage...
          </div>
        ) : frames.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-zinc-500 gap-3">
            <p className="text-sm font-medium">No captured frames yet</p>
            <p className="text-xs text-zinc-600 text-center max-w-sm">
              Run inference with <code className="text-zinc-400">DualDetector</code> enabled to capture
              low-confidence frames here for review.
            </p>
            <button
              onClick={() => fetchFrames()}
              className="text-xs text-zinc-400 underline underline-offset-2 hover:text-zinc-200"
            >
              Refresh
            </button>
          </div>
        ) : (
          <GalleryGrid
            frames={filteredFrames}
            selectedIds={selectedIds}
            onToggleSelection={handleToggleSelection}
            activeId={activeId}
          />
        )}

        {/* Sidebar Inspector */}
        <InspectorPanel
          frame={activeFrame}
          onAction={performAction}
        />
      </div>
    </div>
  );
}
