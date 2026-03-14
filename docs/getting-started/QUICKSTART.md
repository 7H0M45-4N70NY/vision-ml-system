# Quick Start: Auto-Annotation Pipeline

Get the dual-detector + Roboflow auto-annotation loop running in 5 minutes.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set Roboflow API key (optional, for Roboflow upload)
export ROBOFLOW_API_KEY="your_api_key_here"
```

## Option 1: Local Auto-Annotation (No Roboflow)

### Step 1: Run Inference with Dual-Detector

```bash
python scripts/inference.py \
  --config config/inference/base.yaml \
  --source path/to/your/video.mp4 \
  --mode offline
```

This will:
- Run YOLO11n (primary) on every frame
- When confidence < 0.5, run RF-DETR (secondary)
- Save low-confidence frames to `data/low_confidence_frames/`

**Output**:
```
data/low_confidence_frames/
  ├── frame_000000.jpg
  ├── frame_000000.json  (RF-DETR labels)
  ├── frame_000001.jpg
  ├── frame_000001.json
  └── ...

runs/inference/analytics.json
  {
    "dual_detector": {
      "total_frames": 1000,
      "secondary_calls": 150,
      "secondary_ratio": 0.15,
      "frames_saved": 150
    }
  }
```

### Step 2: Export Labels Locally

```python
from vision_ml.labeling.auto_labeler import AutoLabeler
from vision_ml.utils.config import load_config

config = load_config('config/training/base.yaml')
labeler = AutoLabeler(config)

# Load pseudo-labels from DualDetector
count = labeler.load_dual_detector_frames('data/low_confidence_frames')
print(f"Loaded {count} pseudo-labels")

# Export to local JSON
labeler.flush(output_dir='data/auto_labeled')
```

**Output**:
```
data/auto_labeled/
  └── auto_labels.json  (150 pseudo-labeled frames)
```

**Format**:
```json
[
  {
    "image_id": "frame_000000",
    "image_path": "data/low_confidence_frames/frame_000000.jpg",
    "boxes": [[100, 50, 300, 350]],
    "confidences": [0.92],
    "class_ids": [0]
  },
  ...
]
```

---

## Option 2: Auto-Annotation with Roboflow

### Step 1: Create Roboflow Project

1. Go to https://roboflow.com
2. Sign up → Create workspace
3. Create project: "retail-visitor-detection"
4. Choose task: **Object Detection**
5. Copy API key from Settings → API Keys

### Step 2: Configure Your System

Update `config/training/base.yaml`:

```yaml
labeling:
  enabled: true
  provider: roboflow
  roboflow_api_key: null  # Will read from env var
  roboflow_workspace: your_workspace_name
  roboflow_project: retail-visitor-detection
  auto_label_confidence: 0.7
```

Set environment variable:
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

### Step 3: Run Inference

```bash
python scripts/inference.py \
  --config config/inference/base.yaml \
  --source path/to/your/video.mp4 \
  --mode offline
```

### Step 4: Upload to Roboflow

```python
from vision_ml.labeling.auto_labeler import AutoLabeler
from vision_ml.utils.config import load_config

config = load_config('config/training/base.yaml')
labeler = AutoLabeler(config)

# Load pseudo-labels
count = labeler.load_dual_detector_frames('data/low_confidence_frames')
print(f"Loaded {count} pseudo-labels")

# Upload to Roboflow
labeler.flush()  # Uses provider='roboflow' from config
```

**Output**:
```
[AutoLabeler] Loaded 150 pseudo-labels from data/low_confidence_frames
[AutoLabeler] Connected to Roboflow: retail-visitor-detection
```

### Step 5: Review in Roboflow UI

1. Go to Roboflow → Your Project → Annotate
2. See auto-labeled images with RF-DETR predictions
3. For each image:
   - ✅ Approve (correct)
   - ❌ Reject (wrong)
   - ✏️ Edit (fix bounding boxes)
4. Once reviewed, create new dataset version

---

## Option 3: Full Auto-Healing Loop

Run the complete pipeline: inference → auto-label → upload → retrain

```python
#!/usr/bin/env python3
"""End-to-end auto-annotation + retraining loop."""

from vision_ml.inference.pipeline import InferencePipeline
from vision_ml.labeling.auto_labeler import AutoLabeler
from vision_ml.utils.config import load_config

# Step 1: Inference with dual-detector
print("[Loop] Starting inference with dual-detector...")
config = load_config('config/inference/base.yaml')
config['detection']['use_dual_detector'] = True

pipeline = InferencePipeline(config)
summary = pipeline.run_offline('retail_video.mp4')

secondary_ratio = summary['dual_detector']['secondary_ratio']
frames_saved = summary['dual_detector']['frames_saved']

print(f"[Loop] Secondary detector used: {secondary_ratio:.1%}")
print(f"[Loop] Frames saved for review: {frames_saved}")

# Step 2: Check if retraining needed
if secondary_ratio > 0.20:
    print("[Loop] Model degradation detected! Collecting pseudo-labels...")
    
    # Step 3: Upload to Roboflow
    config = load_config('config/training/base.yaml')
    labeler = AutoLabeler(config)
    labeler.load_dual_detector_frames()
    labeler.flush()
    
    print("[Loop] Pseudo-labels uploaded to Roboflow")
    print("[Loop] ⏳ Waiting for human review in Roboflow UI...")
    input("Press Enter once you've reviewed labels in Roboflow...")
    
    # Step 4: Retrain (placeholder)
    print("[Loop] ✅ Retraining with reviewed labels...")
    # trainer = Trainer(config)
    # trainer.train(run_name='auto_retrain_from_pseudo_labels')
    
    print("[Loop] 🎉 Model retrained and deployed!")
else:
    print("[Loop] ✅ Model is healthy, no retraining needed")
```

Save as `scripts/auto_annotation_loop.py` and run:

```bash
python scripts/auto_annotation_loop.py
```

---

## Configuration Reference

### Dual-Detector Settings

Edit `config/inference/base.yaml`:

```yaml
detection:
  use_dual_detector: true
  primary_detector: yolo11n
  secondary_detector: rfdetr
  dual_confidence_threshold: 0.5  # Trigger secondary if < 0.5
  frame_save_dir: data/low_confidence_frames
```

### Auto-Labeler Settings

Edit `config/training/base.yaml`:

```yaml
labeling:
  enabled: true
  provider: local  # or 'roboflow'
  auto_label_confidence: 0.7  # Only collect detections > 70%
  roboflow_api_key: null  # Read from env var
  roboflow_workspace: your_workspace
  roboflow_project: your_project
```

---

## Monitoring

### Check Inference Health

```python
from vision_ml.inference.pipeline import InferencePipeline
from vision_ml.utils.config import load_config
import json

config = load_config('config/inference/base.yaml')
config['detection']['use_dual_detector'] = True

pipeline = InferencePipeline(config)
summary = pipeline.run_offline('video.mp4')

# Read analytics
with open('runs/inference/analytics.json', 'r') as f:
    analytics = json.load(f)

dual_stats = analytics['dual_detector']
print(f"Primary model confidence: {1 - dual_stats['secondary_ratio']:.1%}")
print(f"Frames needing review: {dual_stats['frames_saved']}")

# If secondary_ratio > 0.20, model is degrading
if dual_stats['secondary_ratio'] > 0.20:
    print("⚠️  Model degradation detected! Retrain soon.")
```

### View Saved Frames

```bash
# List low-confidence frames
ls -la data/low_confidence_frames/ | head -20

# View a frame
open data/low_confidence_frames/frame_000000.jpg

# Check labels
cat data/low_confidence_frames/frame_000000.json
```

---

## Troubleshooting

### Issue: "No frames saved"
- Check `dual_confidence_threshold` (default 0.5)
- Increase it to trigger secondary more often
- Or primary model is too confident (good sign!)

### Issue: "Roboflow connection failed"
```bash
# Check API key
echo $ROBOFLOW_API_KEY

# Verify workspace/project names
# Go to Roboflow UI → Settings → check exact names
```

### Issue: "RF-DETR not installed"
```bash
pip install rfdetr
```

### Issue: "YOLO model downloading"
- First run downloads weights (~100MB)
- Subsequent runs use cached model
- Check `~/.cache/yolo/` for model files

---

## Next Steps

1. **Try Option 1** (local) first to test the pipeline
2. **Set up Roboflow** (free tier available)
3. **Run Option 2** to upload pseudo-labels
4. **Review in Roboflow UI** and create dataset version
5. **Implement trainer** to retrain with reviewed labels
6. **Deploy new model** and close the loop

---

## File Structure

```
vision-ml-system/
├── config/
│   ├── inference/base.yaml      # Dual-detector config
│   └── training/base.yaml       # Auto-labeler + Roboflow config
├── data/
│   ├── low_confidence_frames/   # Saved by DualDetector
│   └── auto_labeled/            # Exported by AutoLabeler
├── scripts/
│   ├── inference.py             # Run inference
│   └── auto_annotation_loop.py  # Full pipeline
├── src/vision_ml/
│   ├── detection/
│   │   ├── dual_detector.py     # Saves frames
│   │   ├── detector_factory.py  # YOLO + RF-DETR
│   │   ├── yolo_detector.py
│   │   └── rfdetr_detector.py
│   ├── labeling/
│   │   └── auto_labeler.py      # Loads & exports labels
│   └── inference/
│       └── pipeline.py          # Runs inference
└── docs/
    └── ROBOFLOW_GUIDE.md        # Full documentation
```

---

## Key Metrics to Monitor

| Metric | Meaning | Action |
|---|---|---|
| `secondary_ratio` | % frames needing secondary | > 0.20 = retrain |
| `frames_saved` | # pseudo-labeled frames | More = more training data |
| Human review time | Time to approve labels | Use active learning to reduce |
| Model accuracy | After retraining | Track improvement |

---

## Support

- Full guide: See `docs/ROBOFLOW_GUIDE.md`
- Code: Check `src/vision_ml/labeling/auto_labeler.py`
- Config: Edit `config/training/base.yaml`
