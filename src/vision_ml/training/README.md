# Training Module — `src/vision_ml/training/`

## What This Module Does

This module handles **everything related to making the model better over time**: training new models, tracking experiments, detecting when the model degrades (drift), and triggering retraining. It's the "offline" counterpart to the inference pipeline.

The key insight: **training is not a one-time event**. In production, models degrade as the world changes (lighting conditions shift, store layouts change, new camera angles). This module is designed for **continuous improvement** — train once, monitor always, retrain when needed.

## Files

### `trainer.py` — The `Trainer` Class

This orchestrates the full training lifecycle:

```
Trainer.__init__(config)
    │
    ├── Load YOLO11n model (pretrained weights)
    ├── Initialize MLflowCallback (DagsHub connection)
    │
    ├── train(run_name)           ← Manual trigger
    │     ├── Start MLflow run
    │     ├── Tag as "manual" trigger
    │     ├── Call model.train() with config params
    │     ├── Log metrics + model artifact
    │     ├── Register model in MLflow registry
    │     └── End MLflow run
    │
    └── train_on_drift(run_name)  ← Drift trigger
          ├── Start MLflow run
          ├── Tag as "drift_detected" trigger
          ├── Same training loop
          └── End MLflow run
```

**Why two training methods?**

The training logic is identical, but the **metadata is different**. Tagging runs as `trigger=manual` vs `trigger=drift_detected` in MLflow lets you:
- Filter experiments by trigger type in the MLflow UI
- Track how often drift retraining happens
- Compare manual vs. drift-triggered model quality
- Build dashboards showing model lifecycle events

**The `_log_results()` helper:**

After training, Ultralytics saves weights to `runs/train/<name>/weights/best.pt`. This method:
1. Logs training metrics (mAP, precision, recall, loss) to MLflow
2. Uploads `best.pt` as an MLflow artifact
3. Registers it in the MLflow Model Registry (if `register_model: true`)

The `run_name` parameter was a bug I caught during self-review — the original code always looked up the config's `run_name`, which broke drift retraining (it saves to `drift_retrain/` but looked for `yolo11n_baseline/`). Now it takes the actual name used during training.

**Why Ultralytics' built-in training?**

We could write a custom PyTorch training loop (as shown in the docs/TRAINING_PIPELINE.md), but for a project-level system, Ultralytics handles:
- Data loading with augmentations (mosaic, mixup, hsv shifts)
- Learning rate scheduling (cosine annealing with warmup)
- Early stopping (patience-based)
- Automatic mixed precision
- Checkpoint saving
- mAP computation

Writing all of this from scratch would be 500+ lines with potential bugs. The Ultralytics wrapper is ~30 lines and battle-tested. We add value on top with MLflow tracking and drift detection, not by reimplementing the training loop.

### `callbacks.py` — The `MLflowCallback` Class

This handles all communication with MLflow (and DagsHub).

**DagsHub Integration:**

```python
if 'dagshub.com' in tracking_uri:
    dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
```

DagsHub provides a **free hosted MLflow server** tied to your Git repo. The `dagshub.init()` call configures MLflow to use DagsHub as its backend. This means:
- Experiments are stored remotely (not just `./mlruns/` locally)
- Team members can see each other's experiments
- Model registry is shared
- No need to run your own MLflow server

**Parameter Safety:**

```python
safe_params = {}
for k, v in flat.items():
    str_val = str(v)
    if len(str_val) <= 500:  # MLflow has a 500-char limit on param values
        safe_params[k] = str_val
```

MLflow has a 500-character limit on parameter values. Without this guard, logging a config with long lists or nested dicts crashes the run. The safety filter silently drops oversized params.

**Metric Safety:**

```python
for k, v in metrics.items():
    try:
        safe[k] = float(v)
    except (TypeError, ValueError):
        continue
```

Ultralytics' `results_dict` sometimes contains non-numeric values (strings, None). MLflow only accepts float metrics. This filter skips non-numeric entries instead of crashing.

**Model Registration:**

```python
mlflow.register_model(artifact_uri, model_name)
```

This adds the trained model to the MLflow Model Registry with a version number. The registry supports lifecycle stages: `None → Staging → Production → Archived`. This is how you promote models safely:

```
Train v1 → Register → Staging → Validate → Production
Train v2 → Register → Staging → Compare with v1 → If better → Production
```

### `drift_detector.py` — The `DriftDetector` Class

This monitors inference-time confidence scores to detect model degradation.

**How it works:**

```python
# During inference (online mode):
drift_detector.record(detection_confidences)  # Buffer last N confidences

# Periodically:
if drift_detector.check():
    trainer.train_on_drift()  # Trigger retraining
```

**The confidence drop method:**

```
Healthy model: avg confidence = 0.75 (people detected clearly)
Degraded model: avg confidence = 0.25 (lighting changed, camera moved)

If avg_confidence < threshold (0.3):
    → DRIFT DETECTED → Trigger retraining
```

**Why a sliding window (deque)?**

```python
self.confidence_buffer = deque(maxlen=self.window_size)
```

A deque with `maxlen` automatically discards old values. This means:
- Memory is bounded (always ≤ window_size entries)
- The detector only looks at RECENT performance (not historical)
- No need for manual cleanup

**Why check only every N inferences?**

```python
if self.inference_count % self.check_interval != 0:
    return False
```

Computing the average every single frame is wasteful. Checking every 100 inferences (~3 seconds at 30fps) is frequent enough to catch drift quickly while being negligible in compute cost.

## Config Parameters Explained

```yaml
training:
  epochs: 10          # Full passes over the dataset
  batch_size: 16      # Images per gradient update
  imgsz: 640          # Input resolution (640x640)
  learning_rate: 0.01 # Initial LR (cosine-annealed by Ultralytics)
  optimizer: auto      # Ultralytics picks the best optimizer
  patience: 5          # Stop if no improvement for 5 epochs

drift:
  enabled: false                # Turn on for production monitoring
  method: confidence_drop       # Simple but effective
  confidence_threshold: 0.3     # Below this = drift
  window_size: 500              # Look at last 500 detections
  check_interval: 100           # Check every 100 inferences

schedule:
  mode: manual      # 'manual' | 'daily' | 'weekly' | 'monthly' | 'on_drift'
```

## System Design: Scaling Training

```
Current (Project Level):
  Single GPU, single experiment, manual trigger

FAANG Scale (Continuous Training Platform):
  ┌───────────────────────────────────────────────────────┐
  │  Trigger Sources                                       │
  │    ├── Scheduled (Airflow cron: weekly retrain)        │
  │    ├── Drift Alert (monitoring service)                 │
  │    ├── New Data (DVC push triggers pipeline)            │
  │    └── Manual (engineer clicks "retrain" in dashboard)  │
  │         |                                               │
  │         v                                               │
  │  Training Orchestrator (Kubeflow Pipelines / Vertex AI) │
  │    ├── Data Validation Step (schema + distribution)     │
  │    ├── Training Step (GPU cluster, DDP if needed)       │
  │    ├── Evaluation Step (holdout set + A/B metrics)      │
  │    ├── Registration Step (MLflow Model Registry)        │
  │    └── Promotion Step (Staging → Canary → Production)   │
  │         |                                               │
  │  Model Serving                                          │
  │    ├── Shadow Mode (new model runs alongside old)       │
  │    ├── Canary Deploy (5% traffic → new model)           │
  │    └── Full Rollout (if canary metrics pass)            │
  └───────────────────────────────────────────────────────┘

  Key differences from our project:
  1. Training runs on dedicated GPU nodes (not your laptop)
  2. Data validation prevents training on corrupted datasets
  3. Evaluation gate prevents bad models from reaching production
  4. Canary deployment catches real-world regressions
  5. Rollback is automatic if metrics degrade post-deploy
```

## What I Learned Building This

1. **`actual_name` bug** — The original code used `config.mlflow.run_name` to find the best.pt path, but drift retraining saved to a different directory. This mismatch meant drift-retrained models were never logged. The fix: pass the actual run name through to `_log_results()`.

2. **Ultralytics + MLflow double-logging** — If you `pip install mlflow`, Ultralytics AUTOMATICALLY logs to MLflow. Our custom `MLflowCallback` adds DagsHub integration and model registration on top. The two don't conflict because our callback uses `mlflow.log_artifact` (explicit) while Ultralytics uses its own integration (separate run).

3. **`exist_ok=True` in `model.train()`** — Without this, Ultralytics creates `exp`, `exp2`, `exp3` directories. With it, it overwrites the same directory. For reproducibility, you want `exist_ok=True` so the same config always writes to the same path.

4. **DagsHub auth** — `dagshub.init()` uses your DagsHub token (set via `dagshub login` CLI or `DAGSHUB_TOKEN` env var). If not authenticated, it falls back to anonymous mode (read-only). Always authenticate before training.

5. **Drift detection is a proxy** — Confidence drop doesn't directly measure model accuracy (you'd need ground truth for that). It measures model CERTAINTY. A confident but wrong model won't trigger drift. For production, combine with human-reviewed samples.
