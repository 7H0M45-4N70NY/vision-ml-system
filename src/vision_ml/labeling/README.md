# Labeling Module — `src/vision_ml/labeling/`

## What This Module Does

This module handles **automated data labeling** — using the model's own predictions to generate training labels for future model iterations. This is the "flywheel" that makes the system self-improving:

```
Model v1 detects people → High-confidence detections become labels
    → Labels feed into training data → Train Model v2
        → Model v2 is better → Even better labels → Model v3 → ...
```

This is sometimes called **pseudo-labeling** or **self-training**, and it's how FAANG companies scale their annotation pipelines without hiring thousands of human labelers.

## Files

### `auto_labeler.py` — The `AutoLabeler` Class

**Two modes of operation:**

| Mode | Provider | What Happens |
|---|---|---|
| **Local** | `provider: local` | Saves labels as JSON to `data/auto_labeled/auto_labels.json` |
| **Roboflow** | `provider: roboflow` | Uploads to Roboflow for review, versioning, and augmentation |

**The Confidence Gate:**

```python
high_conf_mask = detections.confidence >= self.min_confidence  # default 0.7
filtered = detections[high_conf_mask]
```

This is the critical quality control step. We only auto-label detections where the model is **very confident** (>70%). Why?

- At 70%+ confidence, YOLO11n has extremely low false positive rate for person detection
- These high-confidence labels are nearly as good as human labels
- Low-confidence detections (30-70%) are ambiguous — they need human review
- Below 30% is likely noise — discard entirely

**The `collect()` method — called during inference:**

```python
auto_labeler.collect(image=frame, detections=detections, image_id="frame_42")
```

This doesn't save anything to disk immediately. It buffers label entries in memory. This is intentional — disk I/O during real-time inference would create latency spikes.

**The `flush()` method — called at the end:**

```python
auto_labeler.flush(output_dir='data/auto_labeled')
```

This writes all buffered labels at once. Batch I/O is always faster than per-frame I/O.

**Label Format (local export):**

```json
[
  {
    "image_id": "frame_42",
    "boxes": [[100, 200, 300, 400], [500, 100, 600, 350]],
    "confidences": [0.92, 0.85],
    "class_ids": [0, 0]
  }
]
```

This is a simple intermediate format. To use these labels for YOLO training, you'd convert them to YOLO format:
```
# YOLO format: class_id center_x center_y width height (normalized)
0 0.3125 0.4688 0.3125 0.3125
0 0.8594 0.3516 0.1563 0.3906
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

## The Auto-Labeling Flywheel

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐  │
│   │ Model v1 │───→│ Inference    │───→│ Collect  │  │
│   │          │    │ (detect)     │    │ Labels   │  │
│   └──────────┘    └──────────────┘    └────┬─────┘  │
│        ^                                    │        │
│        │                                    v        │
│   ┌────┴─────┐    ┌──────────────┐    ┌──────────┐  │
│   │ Train    │←───│ Review       │←───│ Roboflow │  │
│   │ Model v2 │    │ (human QA)   │    │ Upload   │  │
│   └──────────┘    └──────────────┘    └──────────┘  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

The human review step is critical — it catches the ~5% of auto-labels that are wrong. Over time, as the model improves, the human review burden decreases (fewer corrections needed).

## System Design: Scaling Labeling

```
Current (Project Level):
  Buffer labels in memory → JSON dump → Manual review

FAANG Scale (Active Learning Pipeline):
  ┌────────────────────────────────────────────────────────┐
  │  Inference Service (real-time)                          │
  │       |                                                 │
  │  Label Confidence Router                                │
  │    ├── conf > 0.9 → Auto-accept (no human needed)      │
  │    ├── 0.5 < conf < 0.9 → Send to human review queue   │
  │    └── conf < 0.5 → Discard (too noisy)                 │
  │       |                                                 │
  │  Human Review Platform (Label Studio / Scale AI)        │
  │    ├── Prioritize low-confidence, high-uncertainty       │
  │    ├── Review corrections feed back to training         │
  │    └── Annotator agreement metrics for quality control   │
  │       |                                                 │
  │  Dataset Versioning (DVC / Roboflow)                    │
  │    ├── Snapshot dataset after each labeling batch        │
  │    ├── Track label provenance (auto vs human)            │
  │    └── Trigger retrain when N new labels accumulated     │
  └────────────────────────────────────────────────────────┘

  This is called "Active Learning" — the model identifies which
  samples would benefit MOST from human labeling, instead of
  randomly labeling everything. It reduces labeling cost by 5-10x.
```

## Config Parameters

```yaml
labeling:
  enabled: false                  # Turn on when ready for auto-labeling
  provider: roboflow              # 'local' or 'roboflow'
  roboflow_api_key: null          # Set via ROBOFLOW_API_KEY env var
  roboflow_workspace: null        # Your Roboflow workspace name
  roboflow_project: null          # Your Roboflow project name
  auto_label_confidence: 0.7     # Minimum confidence for auto-labels
```

## What I Learned Building This

1. **Never hardcode API keys** — The `os.environ.get('ROBOFLOW_API_KEY')` fallback is a security pattern. Config files get committed to Git. Environment variables don't.

2. **Lazy import for optional dependencies** — `from roboflow import Roboflow` is inside the method, not at the top of the file. This means the `roboflow` package is only required if you actually use the Roboflow provider. Users who only want local export don't need to install it.

3. **The confidence threshold is a hyperparameter** — 0.7 is conservative. For well-performing models in controlled environments (fixed cameras, good lighting), you could lower it to 0.5. For noisy environments, raise it to 0.85. Tune based on your false positive rate.

4. **Buffering prevents I/O latency** — Writing to disk on every frame would add ~5-10ms of latency (file open, write, close). Buffering 1000 frames and writing once adds <1ms amortized. This matters for online mode.

5. **The flywheel effect is real** — At FAANG companies, models that auto-generate training data improve 2-3x faster than models that rely solely on human annotation. The key is quality control (the confidence gate + human review).
