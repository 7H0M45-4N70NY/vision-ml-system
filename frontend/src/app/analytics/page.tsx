import { AnalyticsSummaryCards } from "@/components/analytics/AnalyticsSummaryCards";
import { DriftChart } from "@/components/analytics/DriftChart";
import { VisitorHeatmap } from "@/components/analytics/VisitorHeatmap";
import { ClassDistribution } from "@/components/analytics/ClassDistribution";
import { DataTable } from "@/components/analytics/DataTable";

export default function AnalyticsPage() {
  return (
    <div className="flex flex-col h-full w-full p-6 overflow-y-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold tracking-tight">Analytics Dashboard</h2>
        <p className="text-sm text-zinc-400">Visitor dwell time intelligence, flow patterns, and model health.</p>
      </div>

      <AnalyticsSummaryCards />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <VisitorHeatmap />
        <ClassDistribution />
        <DriftChart />
        <DataTable />
      </div>
    </div>
  );
}
