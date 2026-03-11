# Labeling Module — `src/vision_ml/labeling/`

## What This Module Does

This module handles **active learning** — collecting frames where the model struggles (low-confidence detections) and using them to improve future model iterations:

```
Inference → Low-confidence detections (failures) → Save frames
    → Label with RF-DETR (secondary detector)
        → Training dataset for retraining
            → Model v2 handles previous failure cases better
```

This is **active learning**: focus on hard examples (failures) rather than easy wins. It's more data-efficient than collecting everything, and it's how FAANG companies improve model robustness without massive labeling budgets.

## Files

### `auto_labeler.py` — The `AutoLabeler` Class

**Two modes of operation:**

| Method | What Happens |
|---|---|
| **`collect()`** | No-op. (Disabled — high-confidence collection removed) |
| **`load_dual_detector_frames()`** | Loads low-confidence frames from `data/low_confidence_frames/` (saved by DualDetector, labeled by RF-DETR) |

**Active Learning Approach:**

The module focuses on **hard examples** — frames where the primary detector (YOLO) had low confidence. These are labeled by the secondary detector (RF-DETR) and used for retraining.

Why low-confidence frames?
- YOLO struggles on these frames (model uncertainty = opportunity to improve)
- RF-DETR provides alternative labels for comparison
- Training on failures teaches the model to handle edge cases
- More data-efficient than collecting everything

**The `load_dual_detector_frames()` method:**

```python
auto_labeler.load_dual_detector_frames(frame_dir='data/low_confidence_frames')
```

Loads all low-confidence frames saved by DualDetector. Each frame has:
- **Image**: `frame_000000.jpg`
- **Labels**: `frame_000000.json` (boxes, confidences, class_ids from RF-DETR)

**The `flush()` method — called at the end:**

```python
auto_labeler.flush(output_dir='data/auto_labeled')
```

This writes all loaded labels to `auto_labels.json`. Batch I/O is always faster than per-frame I/O.

**Label Format (auto_labels.json):**

```json
[
  {
    "image_id": "frame_000000",
    "image_path": "data/low_confidence_frames/frame_000000.jpg",
    "boxes": [[24.67, 116.70, 641.06, 479.45]],
    "confidences": [0.625],
    "class_ids": [0]
  }
]
```

These are low-confidence frames from the primary detector (YOLO < 0.5 confidence) labeled by the secondary detector (RF-DETR). The labels are converted to YOLO format for training:
```
# YOLO format: class_id center_x center_y width height (normalized)
0 0.3125 0.4688 0.3125 0.3125
```

### Roboflow Integration

When `provider: roboflow`, the module connects to the Roboflow platform:

```python
from roboflow import Roboflow
rf = Roboflow(api_key=self.roboflow_api_key)
project = rf.workspace(workspace).project(project_name)
```

Roboflow provides:
- **Annotation UI** — Human reviewers correct auto-labels
- **Dataset versioning** — Track which labels were used for which training run
- **Augmentation pipeline** — Automatically augment your dataset
- **Export to YOLO format** — One-click export compatible with Ultralytics

The API key comes from either the config or the `ROBOFLOW_API_KEY` environment variable (security best practice — never hardcode secrets).

## The Active Learning Flywheel

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐  │
│   │ Model v1 │───→│ Inference    │───→│ Identify │  │
│   │          │    │ (detect)     │    │ Failures │  │
│   └──────────┘    └──────────────┘    └────┬─────┘  │
│        ^                                    │        │
│        │                             low-confidence  │
│        │                                    v        │
│   ┌────┴─────┐    ┌──────────────┐    ┌──────────┐  │
│   │ Train    │←───│ RF-DETR      │←───│ Save     │  │
│   │ Model v2 │    │ Labels       │    │ Frames   │  │
│   └──────────┘    └──────────────┘    └──────────┘  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

The focus is on hard examples (low-confidence detections) rather than easy wins. This teaches the model to handle edge cases and improves robustness faster than general data collection.

## System Design: Scaling Active Learning

```
Current (Project Level):
  DualDetector finds failures → RF-DETR labels them → Retrain

FAANG Scale (Uncertainty Sampling):
  ┌────────────────────────────────────────────────────────┐
  │  Inference Service (real-time)                          │
  │       |                                                 │
  │  Uncertainty Detector (disagreement between models)      │
  │    ├── YOLO_conf > 0.8 AND RF-DETR agrees               │
  │    │   → Skip (high confidence, model agrees)           │
  │    ├── YOLO_conf < 0.5 OR models disagree              │
  │    │   → HIGH UNCERTAINTY (save for labeling)           │
  │    └── YOLO_conf < 0.2                                  │
  │        → DISCARD (too noisy, likely false positive)     │
  │       |                                                 │
  │  Prioritized Review Queue                               │
  │    ├── High-uncertainty samples reviewed first           │
  │    ├── Corrections fed back to training                 │
  │    └── Trigger retrain when uncertainty pool filled      │
  │       |                                                 │
  │  Dataset Versioning (DVC / Roboflow)                    │
  │    ├── Snapshot dataset after each labeling batch        │
  │    ├── Track label provenance (auto vs human)            │
  │    └── Measure label quality improvements over time      │
  └────────────────────────────────────────────────────────┘

  Active Learning with disagreement detection finds the hardest
  examples with minimal labeling cost (5-10x reduction vs random).
```

## Config Parameters

```yaml
labeling:
  enabled: false                  # Turn on when ready for active learning
  provider: roboflow              # 'local' or 'roboflow'
  roboflow_api_key: null          # Set via ROBOFLOW_API_KEY env var
  roboflow_workspace: null        # Your Roboflow workspace name
  roboflow_project: null          # Your Roboflow project name
```

## What I Learned Building This

1. **Never hardcode API keys** — The `os.environ.get('ROBOFLOW_API_KEY')` fallback is a security pattern. Config files get committed to Git. Environment variables don't.

2. **Lazy import for optional dependencies** — `from roboflow import Roboflow` is inside the method, not at the top of the file. This means the `roboflow` package is only required if you actually use the Roboflow provider. Users who only want local export don't need to install it.

3. **Active learning is more efficient than random sampling** — Focusing on low-confidence frames (failures) teaches the model what it doesn't know. Random sampling wastes effort on easy examples the model already handles. Active learning improves models 3-5x faster.

4. **Dual detection enables active learning** — DualDetector (YOLO primary + RF-DETR secondary) captures cases where models disagree. These disagreements are high-uncertainty examples — perfect for active learning.

5. **Hard examples compound** — As the model improves on failure cases, it becomes more robust. This is why active learning creates a virtuous cycle: harder examples → better training → fewer failures → focus on remaining hard examples.
