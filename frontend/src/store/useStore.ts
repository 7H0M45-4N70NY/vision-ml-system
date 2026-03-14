import { create } from 'zustand';

interface SystemState {
  gpuUtilization: number;
  fps: number;
  latency: number;
  modelLoaded: boolean;
  objectCount: number;
  isConnected: boolean;
  
  // Pipeline Toggles
  enableDetection: boolean;
  enableTracking: boolean;
  showAnnotations: boolean;
  streamActive: boolean;
  
  // Stream Source
  streamSource: string;

  setTelemetry: (data: Partial<SystemState>) => void;
  setToggle: (key: keyof Pick<SystemState, 'enableDetection' | 'enableTracking' | 'showAnnotations' | 'streamActive'>, value: boolean) => void;
  setStreamSource: (source: string) => void;
}

export const useStore = create<SystemState>((set) => ({
  gpuUtilization: 0,
  fps: 0,
  latency: 0,
  modelLoaded: false,
  objectCount: 0,
  isConnected: false,
  
  enableDetection: true,
  enableTracking: true,
  showAnnotations: true,
  streamActive: true,
  
  streamSource: "0",

  setTelemetry: (data) => set((state) => ({ ...state, ...data })),
  setToggle: (key, value) => {
    set((state) => ({ ...state, [key]: value }));
    // In a real app, this should make an async API call to update the backend
    try {
        fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/config/toggles`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                enable_detection: key === 'enableDetection' ? value : undefined,
                enable_tracking: key === 'enableTracking' ? value : undefined,
                show_annotations: key === 'showAnnotations' ? value : undefined,
                stream_active: key === 'streamActive' ? value : undefined,
            })
        });
    } catch(e) { console.error("Toggle sync failed", e) }
  },
  setStreamSource: (source) => {
    set({ streamSource: source });
    try {
        fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/stream/switch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source })
        });
    } catch(e) { console.error("Stream switch failed", e) }
  }
}));
