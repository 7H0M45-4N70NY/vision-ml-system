"use client";

import { useStore } from "@/store/useStore";
import { Activity, Cpu, Server, Wifi, WifiOff, Menu, Layers, Camera, BoxSelect, Settings } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger, SheetTitle } from "@/components/ui/sheet";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Live Monitor", href: "/", icon: Camera },
  { name: "Triage", href: "/triage", icon: BoxSelect },
  { name: "Analytics", href: "/analytics", icon: Activity },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function Topbar() {
  const { gpuUtilization, fps, isConnected, modelLoaded } = useStore();
  const pathname = usePathname();

  return (
    <header className="flex flex-wrap h-auto min-h-14 py-2 items-center gap-4 border-b border-border bg-zinc-950/50 px-4 md:px-6 backdrop-blur supports-[backdrop-filter]:bg-zinc-950/50 shrink-0">
      
      {/* Mobile Hamburger Layout */}
      <div className="flex items-center lg:hidden">
        <Sheet>
          <SheetTrigger className="mr-2 flex items-center justify-center rounded-md p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100">
              <Menu className="h-5 w-5" />
              <span className="sr-only">Toggle menu</span>
          </SheetTrigger>
          <SheetContent side="left" className="w-64 border-zinc-800 bg-zinc-950 p-0 text-zinc-50">
             <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
             <div className="flex h-full w-full flex-col px-3 py-4">
               <div className="mb-8 flex items-center px-3">
                 <div className="mr-3 flex items-center justify-center rounded-md bg-primary p-1.5">
                   <Layers className="h-5 w-5 text-primary-foreground" />
                 </div>
                 <span className="text-lg font-bold text-zinc-100">VisionFlow</span>
               </div>
               <nav className="flex-1 space-y-1">
                 {navigation.map((item) => {
                   const isActive = pathname === item.href;
                   return (
                     <Link
                       key={item.name}
                       href={item.href}
                       aria-current={isActive ? "page" : undefined}
                       className={cn(
                         "group flex items-center rounded-md px-3 py-2 text-sm font-medium transition-colors",
                         isActive ? "bg-zinc-800 text-zinc-100" : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-100"
                       )}
                     >
                       <item.icon className={cn("mr-3 h-5 w-5 flex-shrink-0", isActive ? "text-zinc-100" : "text-zinc-500 group-hover:text-zinc-300")} aria-hidden="true" />
                       {item.name}
                     </Link>
                   );
                 })}
               </nav>
             </div>
          </SheetContent>
        </Sheet>
        
        {/* Optional Title for Mobile Header */}
        <div className="flex items-center font-bold text-zinc-100">VisionFlow</div>
      </div>

      <div className="flex flex-1 items-center justify-end space-x-2 md:space-x-4 flex-wrap gap-y-2">
        
        {/* Connection Status */}
        <div className="flex items-center gap-2 text-xs font-medium">
          {isConnected ? (
            <Wifi className="h-4 w-4 text-emerald-500" />
          ) : (
            <WifiOff className="h-4 w-4 text-rose-500" />
          )}
          <span className={isConnected ? "text-emerald-500" : "text-rose-500"}>
            {isConnected ? "Connected" : "Disconnected"}
          </span>
        </div>

        <div className="h-4 w-px bg-zinc-800" />

        {/* Model Status */}
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <Server className="h-4 w-4" />
          <span>{modelLoaded ? "Model: Active" : "Model: Standby"}</span>
        </div>

        <div className="h-4 w-px bg-zinc-800" />

        {/* GPU Status */}
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <Cpu className="h-4 w-4" />
          <span>GPU: {gpuUtilization.toFixed(1)}%</span>
        </div>

        <div className="h-4 w-px bg-zinc-800" />

        {/* FPS */}
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <Activity className="h-4 w-4" />
          <span>{fps.toFixed(1)} FPS</span>
        </div>

      </div>
    </header>
  );
}
