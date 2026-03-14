"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Camera, Settings, Layers, BoxSelect } from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Live Monitor", href: "/", icon: Camera },
  { name: "Triage", href: "/triage", icon: BoxSelect },
  { name: "Analytics", href: "/analytics", icon: Activity },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="hidden lg:flex h-full w-64 flex-col border-r border-border bg-zinc-950 px-3 py-4 shrink-0">
      <div className="mb-8 flex items-center px-3">
        <div className="flex items-center justify-center rounded-md bg-primary p-1.5 mr-3">
          <Layers className="h-5 w-5 text-primary-foreground" />
        </div>
        <h1 className="text-lg font-bold text-zinc-100">VisionFlow</h1>
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
                isActive
                  ? "bg-zinc-800 text-zinc-100"
                  : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-100"
              )}
            >
              <item.icon
                className={cn(
                  "mr-3 h-5 w-5 flex-shrink-0",
                  isActive ? "text-zinc-100" : "text-zinc-500 group-hover:text-zinc-300"
                )}
                aria-hidden="true"
              />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* System Status Summary could go here */}
      <div className="mt-auto px-3 py-4 border-t border-zinc-800/50">
        <p className="text-xs text-zinc-500">Vision ML System v1.0.0</p>
      </div>
    </div>
  );
}
