"use client";

import { useStore } from "@/store/useStore";
import { Search, Filter, SlidersHorizontal } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export function FilterBar() {
  return (
    <div className="flex items-center gap-3 w-full bg-zinc-950/50 border border-zinc-800 rounded-lg p-2 px-3 mb-4">
      <div className="relative flex-1 max-w-sm">
        <Search className="absolute left-2 top-2.5 h-4 w-4 text-zinc-500" />
        <Input 
          placeholder="Filter by tags, session ID..." 
          className="pl-8 bg-zinc-900 border-zinc-800 h-9"
        />
      </div>

      <div className="h-6 w-px bg-zinc-800 mx-2" />
      
      <Select defaultValue="low-conf">
        <SelectTrigger className="w-[180px] h-9 bg-zinc-900 border-zinc-800">
          <SelectValue placeholder="Reason" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="low-conf">Confidence &lt; 0.5</SelectItem>
          <SelectItem value="ambiguous">High IoU Overlap</SelectItem>
          <SelectItem value="anomaly">Tracker Anomaly</SelectItem>
          <SelectItem value="all">All Suspicious</SelectItem>
        </SelectContent>
      </Select>

      <Select defaultValue="all">
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

      <Button variant="outline" size="sm" className="h-9 border-zinc-800 bg-zinc-900 text-zinc-300">
        <SlidersHorizontal className="h-4 w-4 mr-2" />
        Advanced
      </Button>
    </div>
  );
}
