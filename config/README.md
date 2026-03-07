# Config — `config/`

## What This Directory Does

This is the **single source of truth** for all hyperparameters, thresholds, and system settings. No magic numbers in code — everything is driven by these YAML files.

The fundamental rule: **if you change behavior, change config, not code.**

## Directory Structure

```
config/
├── README.md
├── inference/
│   └── base.yaml      ← Detection, tracking, annotation, analytics settings
└── training/
    └── base.yaml      ← Training, drift, MLflow, labeling settings
```

## Files

### `inference/base.yaml`

Controls the entire inference pipeline. Broken into logical sections:

| Section | What It Controls |
|---|---|
| `model` | Which YOLO model to load (`yolo11n`) |
| `inference` | Confidence/IOU thresholds, device (cpu/cuda), class filter |
| `mode` | Online vs offline, source path, output settings |
| `tracking` | ByteTrack parameters (thresholds, buffer, frame rate) |
| `annotation` | Which annotators to enable, visual settings |
| `analytics` | Visitor counting, dwell time computation |

**Key design decision**: Inference and training configs are **separate files**. This is because:
- Inference runs in production (needs stability)
- Training runs in experiments (needs flexibility)
- Different people may own each (ML engineer vs. DevOps)
- They have different lifecycle (inference config changes rarely, training config changes per experiment)

### `training/base.yaml`

Controls the training pipeline. Sections:

| Section | What It Controls |
|---|---|
| `model` | Model architecture and pretrained weights |
| `training` | Epochs, batch size, LR, optimizer, early stopping |
| `data` | Dataset paths (YOLO format dataset.yaml) |
| `schedule` | When to trigger retraining (manual/daily/weekly/on_drift) |
| `drift` | Drift detection method and thresholds |
| `mlflow` | DagsHub tracking URI, experiment name, model registration |
| `labeling` | Auto-labeling settings (Roboflow integration) |

## How To Create Experiment Configs

Never edit `base.yaml` for experiments. Instead, create an override:

```yaml
# config/training/experiment_lr_sweep.yaml
training:
  learning_rate: 0.001
  epochs: 20

mlflow:
  run_name: lr_sweep_001
```

Then in code (or a future improvement):
```python
from vision_ml.utils.config import load_config, merge_configs

base = load_config("config/training/base.yaml")
experiment = load_config("config/training/experiment_lr_sweep.yaml")
final = merge_configs(base, experiment)
# final has all base settings + overridden lr and epochs
```

This pattern is used by every major ML framework (Hydra, OmegaConf, Lightning CLI).

## Config Parameter Reference

### Model Parameters
```yaml
model:
  name: yolo11n        # Model variant. Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
                        # n=nano (fastest, least accurate), x=xlarge (slowest, most accurate)
  pretrained: true      # Start from COCO pretrained weights (transfer learning)
  num_classes: 1        # Number of classes to detect (1 = person only)
  task: detect          # 'detect', 'segment', 'classify', 'pose'
```

### Inference Thresholds
```yaml
inference:
  confidence_threshold: 0.35   # Detections below this are discarded
                                # Lower = more detections (more false positives)
                                # Higher = fewer detections (more false negatives)
                                # Sweet spot for people: 0.25-0.5
  iou_threshold: 0.45          # NMS: if two boxes overlap more than this, keep only the best
                                # Lower = more aggressive suppression (fewer overlapping boxes)
                                # Higher = less suppression (allows more overlapping detections)
  classes: [0]                  # COCO class IDs to detect. 0=person. [0,1,2]=person+bicycle+car
```

### Tracking Tuning Guide
```yaml
tracking:
  track_thresh: 0.25    # Detection must be > this to create a NEW track
                         # Lower = tracks appear faster but more false tracks
                         # Higher = tracks appear slower but more reliable
  track_buffer: 30       # Frames to keep a lost track alive
                         # Higher = survives longer occlusions but may ID-swap
                         # Lower = faster cleanup but loses tracks on brief occlusions
  match_thresh: 0.8      # IoU needed to match a detection to an existing track
                         # Higher = stricter matching (fewer ID swaps, more track breaks)
                         # Lower = looser matching (more ID swaps, fewer track breaks)
```

## System Design: Config Management at Scale

```
Current (Project Level):
  YAML files in the repo, loaded at startup

FAANG Scale:
  ┌─────────────────────────────────────────────────┐
  │  Config Sources (priority order):                │
  │    1. CLI arguments (highest priority)            │
  │    2. Environment variables                       │
  │    3. Experiment override YAML                    │
  │    4. Base YAML (lowest priority)                 │
  │                                                   │
  │  Config Service (for production):                 │
  │    ├── AWS Parameter Store / etcd / Consul        │
  │    ├── Feature flags (LaunchDarkly)               │
  │    ├── A/B test configs (experiment platform)     │
  │    └── Secrets (AWS Secrets Manager / Vault)      │
  │                                                   │
  │  Config Versioning:                               │
  │    ├── Every training run snapshots its config    │
  │    ├── MLflow logs config as params               │
  │    ├── Config changes are code-reviewed (PR)      │
  │    └── Config drift detection (prod != expected)  │
  └─────────────────────────────────────────────────┘

  Key principle: Config is code. Treat it with the same
  rigor as source code (version control, review, testing).
```

## What I Learned Building This

1. **Separate inference from training config** — Early on I had one config file for everything. It became confusing which parameters affected which pipeline. Splitting into `inference/` and `training/` directories made ownership clear.

2. **`null` in YAML = `None` in Python** — Used for optional fields like `source: null` and `dataset_yaml: null`. The code checks `or 'fallback'` to handle this: `data_cfg.get('dataset_yaml') or 'coco8.yaml'`.

3. **Lists in YAML** — `classes: [0]` is a YAML list that becomes a Python list `[0]`. Be careful with the flattener in MLflow logging — it converts lists to strings: `"[0]"`. This is fine for logging but would break if you tried to use the flattened value as a list.

4. **Comments are documentation** — YAML supports `#` comments. I use them extensively because the config file IS the documentation for operators. When someone changes `confidence_threshold`, the comment right next to it explains what it does.

5. **Don't nest too deep** — Our configs go 2 levels deep (`annotation.bounding_box.thickness`). Going deeper makes the flatten/merge logic more complex and configs harder to read. Two levels is the sweet spot.
