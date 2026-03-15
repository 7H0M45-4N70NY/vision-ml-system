"use client";

import { Search, SlidersHorizontal } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export interface TriageFilters {
  search: string;
  reason: string;
  classFilter: string;
}

interface FilterBarProps {
  filters: TriageFilters;
  totalFrames: number;
  filteredCount: number;
  onFiltersChange: (filters: TriageFilters) => void;
}

export function FilterBar({ filters, totalFrames, filteredCount, onFiltersChange }: FilterBarProps) {
  const update = (patch: Partial<TriageFilters>) =>
    onFiltersChange({ ...filters, ...patch });

  return (
    <div className="flex items-center gap-3 w-full bg-zinc-950/50 border border-zinc-800 rounded-lg p-2 px-3 mb-4">
      <div className="relative flex-1 max-w-sm">
        <Search className="absolute left-2 top-2.5 h-4 w-4 text-zinc-500" />
        <Input
          placeholder="Filter by class, reason, ID..."
          className="pl-8 bg-zinc-900 border-zinc-800 h-9"
          value={filters.search}
          onChange={(e) => update({ search: e.target.value })}
        />
      </div>

      <div className="h-6 w-px bg-zinc-800 mx-2" />

      <Select value={filters.reason} onValueChange={(v) => update({ reason: v ?? "all" })}>
        <SelectTrigger className="w-[180px] h-9 bg-zinc-900 border-zinc-800">
          <SelectValue placeholder="Reason" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Reasons</SelectItem>
          <SelectItem value="low-conf">Confidence &lt; 0.5</SelectItem>
          <SelectItem value="ambiguous">High IoU Overlap</SelectItem>
          <SelectItem value="anomaly">Tracker Anomaly</SelectItem>
        </SelectContent>
      </Select>

      <Select value={filters.classFilter} onValueChange={(v) => update({ classFilter: v ?? "all" })}>
        <SelectTrigger className="w-[140px] h-9 bg-zinc-900 border-zinc-800">
          <SelectValue placeholder="Class" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">Any Class</SelectItem>
          <SelectItem value="person">Person</SelectItem>
          <SelectItem value="vehicle">Vehicle</SelectItem>
        </SelectContent>
      </Select>

      <div className="flex-1" />

      <span className="text-xs text-zinc-500 font-mono whitespace-nowrap">
        {filteredCount} / {totalFrames}
      </span>

      <Button
        variant="outline"
        size="sm"
        className="h-9 border-zinc-800 bg-zinc-900 text-zinc-300"
        onClick={() => onFiltersChange({ search: "", reason: "all", classFilter: "all" })}
        disabled={filters.search === "" && filters.reason === "all" && filters.classFilter === "all"}
      >
        <SlidersHorizontal className="h-4 w-4 mr-2" />
        Clear
      </Button>
    </div>
  );
}
