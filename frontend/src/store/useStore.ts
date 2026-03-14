import { create } from 'zustand';

interface SystemState {
  gpuUtilization: number;
  fps: number;
  latency: number;
  modelLoaded: boolean;
  objectCount: number;
  isConnected: boolean;
  setTelemetry: (data: Partial<SystemState>) => void;
}

export const useStore = create<SystemState>((set) => ({
  gpuUtilization: 0,
  fps: 0,
  latency: 0,
  modelLoaded: false,
  objectCount: 0,
  isConnected: false,
  setTelemetry: (data) => set((state) => ({ ...state, ...data })),
}));
