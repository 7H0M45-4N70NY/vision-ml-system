import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function SettingsPage() {
  return (
    <div className="flex flex-col h-full w-full p-6 max-w-4xl mx-auto overflow-y-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold tracking-tight">System Settings</h2>
        <p className="text-sm text-zinc-400">Manage backend connections and pipeline defaults.</p>
      </div>

      <div className="space-y-6">
        <Card className="bg-zinc-950/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-base font-medium">Connection Settings</CardTitle>
            <CardDescription>Configure the connection to the Vision ML FastAPI Backend.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-2">
              <Label>Backend WS URL</Label>
              <Input defaultValue="ws://localhost:8000/ws/live-stream" className="bg-zinc-900 border-zinc-800" />
            </div>
            <div className="grid gap-2">
              <Label>Backend REST URL</Label>
              <Input defaultValue="http://localhost:8000" className="bg-zinc-900 border-zinc-800" />
            </div>
            <Button className="mt-2 bg-primary text-primary-foreground hover:bg-primary/90">Save Connection</Button>
          </CardContent>
        </Card>

        <Card className="bg-zinc-950/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-base font-medium">Active Learning Policies</CardTitle>
            <CardDescription>Rules for when frames should be automatically collected.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Collect Low Confidence Frames</Label>
                <div className="text-xs text-zinc-400">Save detections &lt; Threshold</div>
              </div>
              <Switch defaultChecked />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Dual-Detector Overlap Fallback</Label>
                <div className="text-xs text-zinc-400">Trigger RF-DETR when IoU is high</div>
              </div>
              <Switch defaultChecked />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
