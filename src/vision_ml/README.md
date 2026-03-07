# Vision ML System — `src/vision_ml/`

## The Big Picture

This is a **Visitor Analytics System** built with a FAANG-level modular architecture. It detects people in video feeds, tracks them across frames, computes business metrics (unique visitors, dwell time), and supports continuous model improvement through drift detection and automated retraining.

```
                        ┌─────────────────────────────────┐
                        │         VIDEO SOURCE             │
                        │  (file / webcam / RTSP stream)   │
                        └───────────────┬─────────────────┘
                                        │
                                        v
┌───────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                             │
│                                                                   │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │ Detection│───→│ Tracking │───→│Analytics │    │Annotation│  │
│   │ (YOLO11n)│    │(ByteTrack│    │(Visitor  │    │(Supervision│ │
│   │          │    │ via SV)  │    │ Metrics) │    │ Annotators)│ │
│   └──────────┘    └──────────┘    └────┬─────┘    └─────┬─────┘  │
│                                        │                │         │
│                                   Analytics JSON   Annotated Video│
└───────────────────────────────────────────────────────────────────┘
          │
          │ confidence scores feed into...
          v
┌───────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                              │
│                                                                   │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │  Drift   │───→│ Trainer  │───→│  MLflow  │───→│  Model   │  │
│   │ Detector │    │(YOLO11n  │    │ Callback │    │ Registry │  │
│   │          │    │ finetune)│    │ (DagsHub)│    │(Versioned)│  │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                   │
│   ┌──────────┐                                                    │
│   │  Auto    │───→ High-confidence detections become new labels   │
│   │ Labeler  │    (Roboflow integration ready)                    │
│   └──────────┘                                                    │
└───────────────────────────────────────────────────────────────────┘
```

## Module Map

```
src/vision_ml/
├── detection/        ← WHAT is in the frame (YOLO11n → sv.Detections)
├── tracking/         ← WHO is who across frames (ByteTrack → tracker IDs)
├── annotation/       ← VISUALIZE results (Supervision annotators)
├── analytics/        ← MEASURE business metrics (visitors, dwell time)
├── inference/        ← ORCHESTRATE the above into Online/Offline pipelines
├── training/         ← IMPROVE the model (train, drift detect, MLflow)
├── labeling/         ← AUTOMATE data labeling (pseudo-labels, Roboflow)
└── utils/            ← FOUNDATION (config loading, merging, validation)
```

Each module has its own README.md with detailed explanations of:
- What it does and why
- How the code works line by line
- System design thinking for scaling to FAANG level
- What I learned building it

## Data Flow: The `sv.Detections` Contract

The entire system communicates through **one data format**: `supervision.Detections`.

```
sv.Detections object:
  .xyxy          → np.ndarray [N, 4]  — Bounding boxes (x1, y1, x2, y2)
  .confidence    → np.ndarray [N]     — Detection confidence scores
  .class_id      → np.ndarray [N]     — Class IDs (0 = person in COCO)
  .tracker_id    → np.ndarray [N]     — Track IDs (None before tracking, int after)
```

This is the **lingua franca** of the system. Every module reads and/or writes this format:

| Module | Reads | Writes |
|---|---|---|
| Detection | Image (np.ndarray) | sv.Detections (no tracker_id) |
| Tracking | sv.Detections (no tracker_id) | sv.Detections (with tracker_id) |
| Annotation | sv.Detections | Annotated frame (np.ndarray) |
| Analytics | sv.Detections.tracker_id | Summary dict |
| Auto Labeler | sv.Detections | Label JSON |

**Why this matters for scalability**: Any module can be replaced without affecting others. Swap YOLO for RF-DETR? Just implement `BaseDetector.detect() → sv.Detections`. Swap ByteTrack for BoT-SORT? Just implement `BaseTracker.update(sv.Detections) → sv.Detections`. The pipeline doesn't care.

## Online vs Offline Modes

| Aspect | Online (Real-Time) | Offline (Batch) |
|---|---|---|
| **Source** | Webcam, RTSP stream | Video file on disk |
| **Latency** | Critical (~33ms per frame at 30fps) | Not critical |
| **Output** | Live display + event stream | Saved video + analytics JSON |
| **Use case** | Store monitoring dashboard | End-of-day analysis |
| **Scaling** | One pipeline per camera stream | Parallel workers per video |

Both modes use the exact same `process_frame()` method. The only difference is how frames are read (VideoCapture source) and where results go (display vs file).

## The Training Lifecycle

```
Day 0: Deploy YOLO11n (pretrained on COCO)
         ↓
         Inference runs, analytics collected
         ↓
Day 7: Collect auto-labels from high-confidence detections
         ↓
         Human review (via Roboflow or manual)
         ↓
Day 14: Fine-tune on store-specific data
         ↓
         Model v2 registered in MLflow
         ↓
         Compare v1 vs v2 metrics in DagsHub
         ↓
         Promote v2 to production
         ↓
Ongoing: Drift detector monitors confidence
         ↓
         If confidence drops → trigger retraining
         ↓
         Cycle repeats
```

## SOLID Principles in This Codebase

| Principle | How It's Applied |
|---|---|
| **S — Single Responsibility** | Each module does ONE thing. Detection detects. Tracking tracks. Analytics analyzes. |
| **O — Open/Closed** | Add new detectors by subclassing `BaseDetector`. No changes to existing code. |
| **L — Liskov Substitution** | Any `BaseDetector` subclass works in the pipeline. Any `BaseTracker` subclass works. |
| **I — Interface Segregation** | `BaseDetector` has 2 methods. `BaseTracker` has 2 methods. Minimal interfaces. |
| **D — Dependency Inversion** | Pipeline depends on `BaseDetector` (abstraction), not `YOLODetector` (concrete). |

## Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Detection | Ultralytics YOLO11n | State-of-the-art speed/accuracy, great API |
| Tracking | Supervision ByteTrack | Fast, no Re-ID needed, clean API |
| Annotation | Supervision Annotators | Composable, config-driven, beautiful output |
| Experiment Tracking | MLflow + DagsHub | Free hosted, Git-integrated, model registry |
| Auto-Labeling | Roboflow (optional) | Annotation UI, dataset versioning, augmentation |
| Config | PyYAML | Simple, human-readable, standard |
| Video I/O | OpenCV | Industry standard, hardware acceleration |

## System Design: Current vs FAANG Scale

```
┌─────────────────────────────────────────────────────────────────────┐
│  CURRENT ARCHITECTURE (Project Level — Single Machine)              │
│                                                                     │
│  python scripts/inference.py --source video.mp4                     │
│    → Loads YOLO11n into RAM                                         │
│    → Reads frames from disk                                         │
│    → Processes sequentially (1 frame at a time)                     │
│    → Writes output to disk                                          │
│                                                                     │
│  Throughput: ~30 fps (GPU) / ~5 fps (CPU)                          │
│  Latency: ~30ms (GPU) / ~200ms (CPU)                               │
│  Cameras: 1                                                         │
│  Storage: Local disk                                                │
└─────────────────────────────────────────────────────────────────────┘

                            ↓ Scale to ↓

┌─────────────────────────────────────────────────────────────────────┐
│  FAANG ARCHITECTURE (Multi-Store, Multi-Camera, Real-Time)          │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │  Store A     │  │  Store B     │  │  Store C     │               │
│  │  8 cameras   │  │  12 cameras  │  │  6 cameras   │               │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          │                                          │
│                    ┌─────┴─────┐                                    │
│                    │   Edge    │  (Per-store GPU box)                │
│                    │  Gateway  │  YOLO11n TensorRT                  │
│                    │  + Triton │  Dynamic batching                  │
│                    └─────┬─────┘                                    │
│                          │                                          │
│                    Event Stream (Kafka)                              │
│                          │                                          │
│              ┌───────────┼───────────┐                              │
│              │           │           │                               │
│        ┌─────┴────┐ ┌───┴───┐ ┌────┴─────┐                        │
│        │ Analytics│ │ Drift │ │  Auto    │                          │
│        │ Service  │ │Monitor│ │ Labeler  │                          │
│        │(Flink)   │ │       │ │(Roboflow)│                          │
│        └─────┬────┘ └───┬───┘ └────┬─────┘                        │
│              │           │          │                                │
│        ┌─────┴────┐ ┌───┴────┐ ┌───┴─────┐                        │
│        │TimescaleDB│ │Retrain │ │ Dataset │                         │
│        │(metrics)  │ │Trigger │ │ Version │                         │
│        └─────┬────┘ └───┬────┘ └─────────┘                        │
│              │           │                                          │
│        ┌─────┴──────────┴─────┐                                    │
│        │    Dashboard          │                                    │
│        │  (Grafana / React)    │                                    │
│        └──────────────────────┘                                    │
│                                                                     │
│  Throughput: 1000+ fps across all cameras                           │
│  Latency: <10ms per frame (TensorRT)                                │
│  Cameras: 100+                                                      │
│  Storage: S3 + TimescaleDB + MLflow on DagsHub                     │
└─────────────────────────────────────────────────────────────────────┘
```

**The path from here to there:**

1. **Export YOLO11n to TensorRT** → 3-5x inference speedup
2. **Add NVIDIA Triton** → Dynamic batching, multi-stream GPU sharing
3. **Replace OpenCV I/O with GStreamer** → Hardware video decode (NVDEC)
4. **Add Kafka event stream** → Decouple detection from analytics
5. **Add Flink/Spark** → Real-time aggregation across cameras
6. **Add Grafana dashboard** → Live monitoring + historical trends
7. **Add Kubeflow** → Automated retraining orchestration
8. **Add canary deployment** → Safe model updates in production

Each step is independent and can be done incrementally. The modular architecture we have today is the foundation that makes all of this possible.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference on a video (offline mode)
python scripts/inference.py --source path/to/video.mp4

# Run live detection (online mode)
python scripts/inference.py --mode online --source 0

# Train a model (logs to DagsHub MLflow)
python scripts/train.py --config config/training/base.yaml

# Drift-triggered retraining
python scripts/train.py --trigger drift
```

## What I Learned Building This System

1. **Start with the data contract, not the code** — Deciding that `sv.Detections` would be the universal format BEFORE writing modules made everything click together naturally.

2. **Config-driven > hardcoded** — Every threshold, toggle, and path comes from YAML. This means you can run hundreds of experiments without touching Python code.

3. **Thin wrappers over good libraries** — `ByteTrackTracker` is 20 lines wrapping `sv.ByteTrack`. `YOLODetector` is 50 lines wrapping `ultralytics.YOLO`. The value isn't in reimplementing — it's in the interface contracts and config integration.

4. **Online and Offline share the same core** — `process_frame()` is identical in both modes. Only the I/O differs. This means a bug fix in detection logic automatically fixes both modes.

5. **Drift detection is surprisingly simple** — A sliding window average of confidence scores catches most real-world degradation. You don't need complex statistical tests for a project-level system.

6. **The flywheel matters more than the model** — YOLO11n vs YOLO11x is a 2% mAP difference. But having auto-labeling + drift detection + continuous retraining is a 20% improvement over time. The system is more important than the model.
