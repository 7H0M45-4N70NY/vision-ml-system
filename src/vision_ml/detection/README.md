# Detection Module — `src/vision_ml/detection/`

## What This Module Does

This module is responsible for **one thing only**: taking an image (numpy array) and returning structured detections (bounding boxes, confidence scores, class IDs). It follows the **Single Responsibility Principle** — it does NOT track, annotate, or save anything.

## Files

### `base.py` — The Abstract Interface

```python
class BaseDetector(ABC):
    def load_model(self) -> None: ...
    def detect(self, image: np.ndarray) -> sv.Detections: ...
```

**Why an abstract base class?**

This is the **Dependency Inversion Principle** (the D in SOLID) in action. The rest of the system depends on `BaseDetector`, NOT on `YOLODetector` directly. This means we can swap YOLO for RF-DETR, DETR, or any future model by implementing a new subclass — zero changes to the inference pipeline.

**Why does `detect()` return `sv.Detections`?**

`sv.Detections` from the Supervision library is our **canonical detection format**. It's a structured numpy-backed object with fields: `xyxy`, `confidence`, `class_id`, `tracker_id`. By standardizing on this format at the detection boundary, every downstream module (tracker, annotator, analytics) speaks the same language.

### `yolo_detector.py` — YOLO11n Implementation

The concrete implementation wrapping Ultralytics YOLO11n.

**Key design decisions:**

| Decision | Why |
|---|---|
| Config-driven thresholds | Confidence/IOU thresholds come from YAML, not hardcoded. Every experiment can tune them. |
| `classes=[0]` default | COCO class 0 = person. For visitor analytics we only care about people. Filtering at detection time saves tracker computation. |
| `sv.Detections.from_ultralytics()` | The Supervision library provides a bridge that converts Ultralytics result objects into `sv.Detections`. This is the cleanest integration point. |
| `detect_raw()` method | Sometimes you need the raw Ultralytics result object (e.g., for plotting with their built-in visualizer, or accessing segmentation masks). This escape hatch keeps the module flexible. |

**How YOLO11n works (the mental model):**

```
Input Image (640x640x3)
    |
    v
Backbone (feature extraction) -----> Multi-scale feature maps
    |
    v
Neck (feature fusion - FPN/PANet) -> Fused features at 3 scales
    |
    v
Head (detection) ------------------> Raw predictions (8400 boxes for 640px)
    |
    v
NMS (Non-Maximum Suppression) -----> Filtered detections (typically 5-50)
    |
    v
sv.Detections.from_ultralytics() --> Structured output
```

## System Design: Scaling Detection

```
Current (Project Level):
  Single process, single model, batch_size=1
  Latency: ~20-50ms/frame on GPU, ~200ms on CPU

FAANG Production Scale:
  ┌─────────────────────────────────────────────────┐
  │  Load Balancer (NGINX / Envoy)                  │
  │       |                                          │
  │  ┌────┴────┐  ┌────────┐  ┌────────┐           │
  │  │ Triton  │  │ Triton │  │ Triton │  (N pods) │
  │  │ Server  │  │ Server │  │ Server │           │
  │  │ YOLO11n │  │ YOLO11n│  │ YOLO11n│           │
  │  │ TensorRT│  │ TensorRT│ │ TensorRT│          │
  │  └─────────┘  └────────┘  └────────┘           │
  │       |            |            |                │
  │       Dynamic Batching (batch=8-16)              │
  │       |                                          │
  │  GPU Inference: 2-5ms per batch                  │
  └─────────────────────────────────────────────────┘

Key scaling levers:
1. TensorRT export: 3-5x faster than PyTorch eager mode
2. Dynamic batching: Triton queues frames, sends batch to GPU
3. Horizontal scaling: Add more Triton pods behind load balancer
4. Resolution: 640 -> 320 = 4x fewer FLOPs (quadratic scaling)
5. INT8 quantization: 2x faster on Tensor Cores
```

## What I Learned Building This

1. **Model loading happens once in `__init__`** — YOLO model loading downloads weights (~6MB for nano) and initializes the computation graph. This takes 1-3 seconds. You NEVER want this in the hot path (per-frame). Load once, infer many.

2. **`device` parameter matters** — Passing `device='cpu'` to `YOLO()` vs `model()` call behaves differently. We pass it in the `model()` call (inference time) so the model can be loaded on CPU and optionally moved. For production, you'd load directly to GPU.

3. **`classes=[0]` filtering at YOLO level** — We could filter AFTER detection, but that wastes NMS computation on classes we don't care about. Filtering IN the YOLO call means NMS only processes person detections. At 8400 raw predictions, this matters.

4. **`verbose=False`** — Ultralytics prints a LOT by default (speed, boxes, etc). For a pipeline that processes thousands of frames, this floods stdout and slows down I/O. Always suppress in production.
