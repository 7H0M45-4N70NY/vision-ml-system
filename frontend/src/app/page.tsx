import { ControlPanel } from "@/components/monitor/ControlPanel";
import { TelemetryPanel } from "@/components/monitor/TelemetryPanel";
import { VideoCanvas } from "@/components/monitor/VideoCanvas";

export default function MonitorPage() {
  return (
    <div className="flex flex-col xl:flex-row h-full w-full gap-4 p-4 overflow-y-auto xl:overflow-hidden">
      {/* Left Panel */}
      <ControlPanel />

      {/* Center Stage (Video Stream) */}
      <div className="flex-1 flex flex-col min-h-[400px] xl:h-full lg:min-w-0">
         <VideoCanvas />
      </div>

      {/* Right Panel */}
      <TelemetryPanel />
    </div>
  );
}
