# Annotation Module — `src/vision_ml/annotation/`

## What This Module Does

This module is the **visualization layer**. It takes a raw video frame and detection results, and draws bounding boxes, labels, and movement traces on top. It does NOT modify the detections or affect the analytics — it's purely for human consumption (debugging, demos, monitoring dashboards).

## Files

### `annotator.py` — `FrameAnnotator` Class

This class composes multiple Supervision annotators into a single `annotate()` call.

**The three annotators we use:**

| Annotator | What It Draws | When Useful |
|---|---|---|
| `sv.BoundingBoxAnnotator` | Colored rectangles around each detection | Always — shows WHERE detections are |
| `sv.LabelAnnotator` | Text labels (track ID + confidence) | Always — shows WHO and HOW SURE |
| `sv.TraceAnnotator` | Path lines showing movement history | Visitor analytics — shows WHERE people walked |

**How they compose:**

```python
annotated = frame.copy()                          # Never mutate original
annotated = box_annotator.annotate(annotated, detections)
annotated = label_annotator.annotate(annotated, detections, labels)
annotated = trace_annotator.annotate(annotated, detections)
```

Each annotator takes the scene in and returns a new scene with its drawings added. They stack like layers in Photoshop.

**Why `frame.copy()` on line 1?**

OpenCV images are numpy arrays (mutable). If we annotate the original frame, we corrupt it for downstream consumers (analytics, video writer, next frame's tracker). The copy ensures the original stays pristine. This costs ~3MB per 1080p frame, which is acceptable.

### `build_labels()` — Static Helper

```python
labels = FrameAnnotator.build_labels(detections)
# → ["#1 0.87", "#2 0.92", "#? 0.45"]
```

Formats labels as `#<track_id> <confidence>`. The `?` fallback handles the case where tracking is disabled (no `tracker_id` assigned). This is a static method because it doesn't depend on annotator state.

## Config-Driven Annotators

Each annotator is individually toggleable via YAML:

```yaml
annotation:
  bounding_box:
    enabled: true        # Set false to hide boxes
    thickness: 2         # Line thickness in pixels
  label:
    enabled: true
    text_scale: 0.5
    text_thickness: 1
  trace:
    enabled: true        # Movement traces (requires tracking)
    trace_length: 60     # How many past positions to show
```

**Why config-driven?**

- **Debugging**: Enable all annotators to see everything
- **Production monitoring**: Maybe just boxes, no labels (cleaner dashboard)
- **Performance**: Annotation adds ~2-5ms per frame. Disabling all annotators when you only need analytics data saves this overhead
- **Demo mode**: Enable traces to impress stakeholders with movement visualization

## System Design: Scaling Annotation

```
Current (Project Level):
  Annotate every frame synchronously in the inference pipeline

FAANG Scale:
  ┌────────────────────────────────────────────────┐
  │  Detection + Tracking (GPU, real-time)          │
  │        |                                        │
  │        ├──→ Analytics Engine (CPU, real-time)    │
  │        |                                        │
  │        └──→ Annotation Queue (async, optional)  │
  │                    |                             │
  │             Annotation Workers (CPU)             │
  │                    |                             │
  │             Video Encoder (H264/H265)            │
  │                    |                             │
  │             CDN / Dashboard Stream               │
  └────────────────────────────────────────────────┘

  Key insight: Annotation is a PRESENTATION concern, not a COMPUTATION
  concern. In production, you often skip annotation entirely and only
  generate annotated video on-demand (e.g., when an operator clicks
  "view camera feed" on the dashboard).

  Scaling levers:
  1. Decouple annotation from inference (async queue)
  2. Only annotate when someone is watching (lazy rendering)
  3. Use hardware video encoding (NVENC) for output streams
  4. Downsample annotated output (720p is fine for dashboards)
```

## What I Learned Building This

1. **Supervision annotators are composable** — They all follow the same `annotate(scene, detections)` pattern. You can add `sv.HeatMapAnnotator`, `sv.MaskAnnotator`, etc. without changing the architecture.

2. **`TraceAnnotator` needs `tracker_id`** — If tracking is disabled, the trace annotator still works but draws nothing meaningful (no persistent IDs to trace). This is handled gracefully — no crash, just no traces.

3. **Label building is separate from annotation** — The annotators don't know what text to show. We build labels externally and pass them in. This is good separation — the label format can change without touching the annotator.

4. **Color palette is automatic** — When `color=null` in config, Supervision assigns colors from a built-in palette based on class_id or tracker_id. This means each tracked person gets a consistent color across frames.
