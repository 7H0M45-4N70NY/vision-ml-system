# Inference Module — `src/vision_ml/inference/`

## What This Module Does

This is the **orchestrator** — the module that wires detection, tracking, annotation, and analytics together into a working end-to-end pipeline. It supports two modes:

- **Offline mode**: Process a video file, save annotated output + analytics JSON
- **Online mode**: Process a live stream (webcam/RTSP), display real-time overlay, accumulate analytics

## Files

### `pipeline.py` — `InferencePipeline` Class

**Architecture:**

```
InferencePipeline
  ├── YOLODetector      (detection/)
  ├── ByteTrackTracker  (tracking/)
  ├── FrameAnnotator    (annotation/)
  └── VisitorAnalytics  (analytics/)
```

This is the **Facade Pattern** — a single entry point (`pipeline.run()`) that coordinates four independent modules. External code never needs to know about the internal wiring.

**The Core Loop — `process_frame()`:**

```python
def process_frame(self, frame, frame_idx):
    detections = self.detector.detect(frame)        # Step 1: YOLO inference
    detections = self.tracker.update(detections)     # Step 2: Assign track IDs
    self.analytics.update(tracker_ids, frame_idx)    # Step 3: Record analytics
    annotated = self.annotator.annotate(frame, ...)  # Step 4: Draw overlays
    return detections, annotated
```

Each step is ~one line because all the complexity lives inside each module. This is the payoff of modular design.

**Offline Mode — `run_offline(video_path)`:**

```
Open video file → Read frames sequentially → process_frame() each
                → Write annotated frames to output video
                → Save analytics JSON at the end
```

Key details:
- FPS is read from the video file and propagated to analytics (for dwell time)
- `cv2.VideoWriter` uses `mp4v` codec (widely compatible)
- Analytics JSON is saved to `runs/inference/analytics.json`

**Online Mode — `run_online(source)`:**

```
Open camera/stream → Read frames in real-time → process_frame() each
                   → Display annotated frame with cv2.imshow()
                   → Overlay visitor count + in-frame count
                   → Press 'q' to quit
                   → Save analytics on exit
```

Key details:
- `cv2.waitKey(1)` keeps the display responsive (1ms wait per frame)
- `try/finally` ensures camera is released even on crash
- Visitor count overlay uses `cv2.putText` (simple, no dependency)

**Dispatcher — `run(source)`:**

```python
def run(self, source=None):
    if self.mode == 'online':
        return self.run_online(source)
    else:
        return self.run_offline(source)
```

The mode comes from config (`mode.type: online/offline`) but can be overridden via the CLI script's `--mode` flag. This lets you test online mode without editing YAML.

## How the Modules Communicate

```
         sv.Detections (no tracker_id)
              |
YOLODetector ─┘
              |
              v
         sv.Detections (with tracker_id)
              |
ByteTrackTracker ─┘
              |
              ├──→ VisitorAnalytics.update(tracker_ids, frame_idx)
              |
              └──→ FrameAnnotator.annotate(frame, detections, labels)
                        |
                        v
                   np.ndarray (annotated frame)
```

The key insight: `sv.Detections` is the **lingua franca**. Every module speaks it. The tracker enriches it (adds `tracker_id`), the annotator reads it, the analytics reads from it. No format conversions, no adapters needed.

## System Design: Online vs Offline at Scale

```
Offline Mode (Batch Processing):
  ┌─────────────────────────────────────────────┐
  │  Video Storage (S3 / GCS)                    │
  │       |                                      │
  │  Job Scheduler (Airflow / Prefect)           │
  │       |                                      │
  │  Worker Pool (K8s Jobs)                      │
  │    ├── Worker 1: video_001.mp4               │
  │    ├── Worker 2: video_002.mp4               │
  │    └── Worker 3: video_003.mp4               │
  │       |                                      │
  │  Results → Analytics DB + Annotated Videos    │
  └─────────────────────────────────────────────┘

  Scaling: Embarrassingly parallel per-video.
  Add more workers = process more videos simultaneously.

Online Mode (Real-Time Streaming):
  ┌─────────────────────────────────────────────┐
  │  Camera RTSP Streams                         │
  │       |                                      │
  │  Stream Ingestion (GStreamer / FFmpeg)        │
  │       |                                      │
  │  Frame Buffer (shared memory / ZMQ)          │
  │       |                                      │
  │  Inference Service (GPU, one per camera)     │
  │       |                                      │
  │  Event Stream → Kafka → Analytics Service    │
  │       |                                      │
  │  Annotated Stream → WebRTC → Dashboard       │
  └─────────────────────────────────────────────┘

  Scaling: One inference service per camera stream.
  Bottleneck is GPU memory (how many streams per GPU).
  YOLO11n at 640px ≈ 200MB VRAM → ~20 streams per 4GB GPU.
```

## What I Learned Building This

1. **`cv2.VideoCapture` returns `(False, None)` at end of video** — The `while cap.isOpened()` + `if not success: break` pattern is the standard way to handle this. Without the success check, you get a crash on `None` frame.

2. **FPS from `cap.get(cv2.CAP_PROP_FPS)` can be 0** — Some webcams and RTSP streams report 0 fps. The `or 30` fallback prevents division-by-zero in dwell time calculation. Not perfect (assumes 30fps) but safe.

3. **`cv2.VideoWriter` codec compatibility** — `mp4v` works everywhere but produces larger files. `avc1` (H.264) is smaller but requires system codecs. For a project, `mp4v` is the safe choice.

4. **Online mode needs `cv2.destroyAllWindows()`** — Without this, OpenCV windows persist after the script exits, hanging the process. The `try/finally` block ensures cleanup.

5. **`reset()` matters for the pipeline** — It resets both tracker and analytics. Without this, running the pipeline twice on different videos in the same Python process produces wrong results. The scripts don't hit this (they exit after one run), but library users might.
