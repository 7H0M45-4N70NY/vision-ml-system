# Product Specification: VisionFlow Control Center (React/Next.js)

**Version:** 1.0.0
**Status:** Draft
**Target Audience:** ML Engineers, Computer Vision Developers

---

## 1. Executive Summary

This specification outlines the architecture and design for **VisionFlow Control Center**, a professional-grade, local-first web interface for the Vision ML System. 

**Objective:** Transform the current MVP (Streamlit) into a responsive, low-latency "Mission Control" console using modern web technologies, enabling real-time monitoring, debugging, and active learning for edge computer vision models.

**Core Philosophy:** "The IDE for your Computer Vision Pipeline."

---

## 2. Technical Architecture

The system uses a **Hybrid Architecture**:
- **Backend (Existing):** Python (FastAPI) handles the heavy lifting—inference, hardware access, database (SQLite), and file I/O.
- **Frontend (New):** Next.js (React) handles the presentation, state management, and visualization.
- **Bridge:** WebSockets for real-time streams; REST API for CRUD operations.

### 2.1 Tech Stack

| Layer | Technology | Justification |
| :--- | :--- | :--- |
| **Framework** | **Next.js 14 (App Router)** | Robust routing, server-side rendering for initial load, easy deployment. |
| **Styling** | **Tailwind CSS** | Rapid UI development, consistency, easy dark mode. |
| **Components** | **ShadCN UI** | accessible, customizable, copy-paste components (based on Radix UI). |
| **State** | **Zustand** | Minimalist global state (simpler than Redux) for "Model Status". |
| **Data Viz** | **Recharts** | Composable React charts for telemetry and drift monitoring. |
| **Streaming** | **Socket.io / WebSocket** | Low-latency transmission of MJPEG frames and JSON telemetry. |
| **Backend** | **FastAPI (Python)** | High-performance Python API, native async support for streams. |

### 2.2 System Diagram

```mermaid
graph TD
    User[ML Engineer] -->|HTTP/WS| NextJS[Next.js Frontend]
    NextJS -->|REST API| FastAPI[Python Backend]
    NextJS -->|WebSocket| FastAPI
    
    subgraph "Python Backend (src/vision_ml)"
        FastAPI --> Inference[Inference Engine (YOLO/RT-DETR)]
        FastAPI --> DB[(SQLite Analytics.db)]
        FastAPI --> FS[File System (Images/Config)]
    end
```

---

## 3. API Requirements (Backend Expansion)

To support the new UI, the existing `src/vision_ml/api/main.py` must be upgraded.

### 3.1 New WebSocket Endpoints
*   `WS /ws/live-stream`: Bi-directional stream.
    *   **Server -> Client:** Sends JPEG binary chunks (video) + JSON Metadata (bounding boxes, inference time, confidence).
    *   **Client -> Server:** Sends control commands (e.g., `{"action": "set_threshold", "value": 0.6}`).

### 3.2 Enhanced REST Endpoints
*   **Configuration:**
    *   `GET /config`: Retrieve current YAML config.
    *   `PATCH /config`: Update runtime parameters (hot-reload).
*   **Triage (Active Learning):**
    *   `GET /triage/frames`: List low-confidence images with metadata.
    *   `POST /triage/action`: Batch accept/reject/label frames.
*   **Analytics:**
    *   `GET /analytics/stats`: Aggregate stats for the "Overview" dashboard.
    *   `GET /analytics/timeseries`: Data for the drift/latency charts.

---

## 4. UI/UX Specification

The interface is divided into three primary **Workspaces** accessible via a global sidebar.

### 4.1 Global Elements
*   **Theme:** "Developer Dark Mode" (Zinc-950 background, Zinc-800 borders).
*   **Sidebar:** Collapsible, containing:
    *   **Status Indicators:** GPU Utilization, FPS, Model Loaded.
    *   **Navigation:** Monitor, Triage, Analytics, Settings.
*   **Toast Notifications:** Non-blocking alerts for "Drift Detected" or "Export Complete".

### 4.2 Workspace A: Monitor (The "Live" View)
**Goal:** Real-time observability of the inference pipeline.

*   **Layout:** 3-Column Grid.
*   **Left Panel (Controls):**
    *   **Source Selector:** Webcam / RTSP URL / Upload File.
    *   **Model Config:** Sliders for `Confidence Threshold`, `IOU Threshold`.
    *   **Mode Switch:** "Hot Path" vs "Inline Dual-Detector".
*   **Center Panel (Stage):**
    *   **Canvas:** Renders the raw video stream.
    *   **Overlay Layer:** SVG overlay drawing bounding boxes client-side (smoother than server-side burned-in boxes).
    *   **Tooltip:** Hover over a box to see `Class: Person (0.92) | Tracker ID: 45`.
*   **Right Panel (Telemetry):**
    *   **Live Metrics:** Big numbers for FPS, Latency (ms), Object Count.
    *   **Confidence Graph:** Scrolling line chart showing average confidence over the last 60 seconds.
    *   **Log Stream:** A terminal-like window showing structured logs (`[INFO] Drift check passed: 0.12`).

### 4.3 Workspace B: Triage (Active Learning)
**Goal:** Rapidly process "interesting" frames captured by the system.

*   **Layout:** Gallery Grid + Action Bar.
*   **Filter Bar:** `Filter by: Low Confidence (< 0.5) | Specific Class (Person)`.
*   **Grid View:**
    *   Cards showing the captured frame.
    *   Badges for "Reason" (e.g., "Low Conf", "Ambiguous").
*   **Selection Mode:**
    *   Click to select multiple frames.
    *   **Keyboard Shortcuts:** `X` to Discard, `K` to Keep, `Enter` to Label.
*   **Inspector (Side Panel):**
    *   When a frame is selected, show detailed JSON metadata (sensor values, timestamp).

### 4.4 Workspace C: Analytics (Long-term Health)
**Goal:** Historical analysis and drift detection.

*   **Charts:**
    *   **Drift Trends:** Line chart of daily drift scores.
    *   **Visitor Heatmap:** Time-of-day distribution of detections.
    *   **Class Distribution:** Bar chart of detected object classes.
*   **Data Table:**
    *   Sortable table of all "Inference Runs" with export to CSV/JSON.

---

## 5. Implementation Plan

This is a **Parallel Effort**. The Streamlit app remains the "Stable" branch while we build the "Next-Gen" frontend.

### Phase 1: Foundation (Weeks 1-2)
1.  **Repo Setup:** Initialize `frontend/` directory with Next.js + Tailwind.
2.  **API Bridge:** Update `src/vision_ml/api` to support CORS and basic WebSocket echoing.
3.  **"Hello World":** Create the Next.js app that connects to the Python backend and displays the version number.

### Phase 2: The Live Monitor (Weeks 3-4)
1.  **Video Streaming:** Implement the MJPEG/WebSocket stream in FastAPI.
2.  **Canvas Renderer:** Build the React component to draw bounding boxes on top of the video stream.
3.  **Telemetry:** Connect the real-time stats (FPS/Latency) to the UI.

### Phase 3: Triage & Config (Weeks 5-6)
1.  **Gallery UI:** Build the grid view for `data/low_confidence_frames`.
2.  **Config API:** specific endpoints to read/write `config/inference/base.yaml`.
3.  **Theme Polish:** Apply the "Linear-like" styling tokens.

### Phase 4: Beta Launch
1.  **Docker Compose:** Update `docker-compose.yml` to spin up both the API (backend) and Next.js (frontend).
2.  **User Testing:** Verify latency and responsiveness compared to Streamlit.

---

## 6. Development Guidelines

*   **Component Library:** Use `shadcn/ui` for all interactive elements (Select, Sliders, Dialogs). Do not build from scratch.
*   **Icons:** `lucide-react` for consistent iconography.
*   **Type Safety:** Generate TypeScript interfaces from the FastAPI Pydantic models (using `openapi-typescript`).
*   **State:** Use `React Query` (TanStack Query) for server state (fetching lists) and `Zustand` for client state (UI toggles).

---
