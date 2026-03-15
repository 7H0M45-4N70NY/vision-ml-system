import { create } from 'zustand';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface SystemState {
  gpuUtilization: number;
  fps: number;
  latency: number;
  modelLoaded: boolean;
  objectCount: number;
  isConnected: boolean;
  confidences: number[];

  // Pipeline Toggles
  enableDetection: boolean;
  enableTracking: boolean;
  showAnnotations: boolean;
  streamActive: boolean;
  dualMode: boolean;

  // Stream Source
  streamSource: string;

  setTelemetry: (data: TelemetryPayload) => void;
  setToggle: (key: keyof Pick<SystemState, 'enableDetection' | 'enableTracking' | 'showAnnotations' | 'streamActive'>, value: boolean) => void;
  setStreamSource: (source: string) => void;
  setThresholds: (conf?: number, iou?: number) => void;
  setDualMode: (enabled: boolean) => void;
}

// Separate payload type so avgConfidence can be sent from the WebSocket
// without polluting SystemState (it is accumulated into `confidences`).
interface TelemetryPayload extends Partial<Omit<SystemState, 'confidences'>> {
  avgConfidence?: number;
}

export const useStore = create<SystemState>((set) => ({
  gpuUtilization: 0,
  fps: 0,
  latency: 0,
  modelLoaded: false,
  objectCount: 0,
  isConnected: false,
  confidences: [],

  enableDetection: true,
  enableTracking: true,
  showAnnotations: true,
  streamActive: true,
  dualMode: false,

  streamSource: "0",

  setTelemetry: ({ avgConfidence, ...rest }) => set((state) => {
    const update: Partial<SystemState> = { ...rest };
    if (avgConfidence !== undefined) {
      update.confidences = [...state.confidences, avgConfidence].slice(-60);
    }
    return update;
  }),

  setToggle: (key, value) => {
    set({ [key]: value });
    fetch(`${API_BASE}/config/toggles`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        enable_detection:  key === 'enableDetection'  ? value : undefined,
        enable_tracking:   key === 'enableTracking'   ? value : undefined,
        show_annotations:  key === 'showAnnotations'  ? value : undefined,
        stream_active:     key === 'streamActive'     ? value : undefined,
      }),
    }).catch(e => console.error("Toggle sync failed", e));
  },

  setStreamSource: (source) => {
    set({ streamSource: source });
    fetch(`${API_BASE}/stream/switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    }).catch(e => console.error("Stream switch failed", e));
  },

  setThresholds: (conf, iou) => {
    fetch(`${API_BASE}/config`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confidence_threshold: conf, iou_threshold: iou }),
    }).catch(e => console.error("Threshold sync failed", e));
  },

  setDualMode: (enabled) => {
    set({ dualMode: enabled });
    fetch(`${API_BASE}/config`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dual_mode: enabled }),
    }).catch(e => console.error("Dual mode sync failed", e));
  },
}));
