# Roboflow Integration Guide for Computer Vision Engineers

## Overview

Roboflow is a **data-centric ML platform** for computer vision. It handles:
- Dataset versioning & management
- Auto-annotation (using models to label data)
- Active learning (prioritize uncertain predictions for human review)
- Model training & deployment
- API for inference

This guide covers **auto-annotation + training loop** integration with your Vision ML system.

---

## 1. Roboflow Concepts

### Dataset vs Project
- **Project**: Container for all versions of a dataset (e.g., "retail-visitor-detection")
- **Dataset Version**: Immutable snapshot (e.g., v1, v2, v3) with train/val/test splits
- **Workspace**: Organization containing multiple projects

### Annotation Methods
1. **Manual**: Human annotators label images
2. **Auto-annotation**: Model predicts labels (fast, needs review)
3. **Active Learning**: Model identifies uncertain predictions, human reviews only those

### Label Format
Roboflow supports multiple formats:
- **COCO JSON**: `{image_id, annotations: [{bbox, category_id, ...}]}`
- **Pascal VOC XML**: Bounding box format
- **YOLO txt**: Per-image label files with normalized coordinates

---

## 2. Your Auto-Annotation Pipeline

### Flow

```
Video Inference (pipeline.py)
    │
    ├─ Primary Detector (YOLO11n) → High confidence → Keep
    │
    └─ Low confidence → Secondary Detector (RF-DETR)
         │
         ├─ Save frame.jpg + labels.json to data/low_confidence_frames/
         │  (These are pseudo-labels from RF-DETR)
         │
         └─ DualDetector.stats tracks:
            - secondary_ratio: % of frames needing secondary
            - frames_saved: # of low-confidence frames
            
Training (when ready)
    │
    ├─ AutoLabeler.load_dual_detector_frames()
    │  → Reads data/low_confidence_frames/*.json as pseudo-labels
    │
    ├─ AutoLabeler.flush(provider='roboflow')
    │  → Upload to Roboflow for human review
    │
    └─ Human review in Roboflow UI
       → Approve/correct labels
       → Create new dataset version
       → Train new model
```

### Why This Matters

**Traditional pipeline** (manual annotation):
```
Collect video → Manual annotation (expensive, slow) → Train → Deploy
```

**Auto-annotation pipeline** (your system):
```
Collect video → Auto-label with secondary detector → Human review (fast) → Train → Deploy
```

**Benefit**: 5-10x faster annotation by having the model do the heavy lifting.

---

## 3. Setting Up Roboflow

### Step 1: Create Account & Project

```bash
# Go to https://roboflow.com
# Sign up → Create workspace
# Create project: "retail-visitor-detection"
# Choose task: Object Detection
# Choose annotation format: COCO JSON (or YOLO)
```

### Step 2: Get API Key

```bash
# In Roboflow UI: Settings → API Keys
# Copy your API key
export ROBOFLOW_API_KEY="your_api_key_here"
```

### Step 3: Configure Your System

Update `config/training/base.yaml`:

```yaml
labeling:
  enabled: true
  provider: roboflow
  roboflow_api_key: null  # Will read from env var ROBOFLOW_API_KEY
  roboflow_workspace: your_workspace_name
  roboflow_project: retail-visitor-detection
  auto_label_confidence: 0.7  # Only collect detections > 70% confidence
```

---

## 4. Auto-Annotation Workflow

### Phase 1: Collect Pseudo-Labels

```python
from vision_ml.inference.pipeline import InferencePipeline
from vision_ml.utils.config import load_config

config = load_config('config/inference/base.yaml')
config['detection']['use_dual_detector'] = True  # Enable dual-detector mode

pipeline = InferencePipeline(config)
summary = pipeline.run_offline('retail_video.mp4')

# Output:
# data/low_confidence_frames/
#   ├── frame_000000.jpg
#   ├── frame_000000.json  (RF-DETR labels)
#   ├── frame_000001.jpg
#   ├── frame_000001.json
#   └── ...
#
# analytics.json includes:
# {
#   "dual_detector": {
#     "total_frames": 1000,
#     "secondary_calls": 150,
#     "secondary_ratio": 0.15,  # 15% needed secondary
#     "frames_saved": 150
#   }
# }
```

**Interpretation**:
- `secondary_ratio: 0.15` → Primary model is 85% confident
- `frames_saved: 150` → 150 frames need human review
- If ratio > 0.20 → Primary model is degrading, retrain soon

### Phase 2: Upload to Roboflow

```python
from vision_ml.labeling.auto_labeler import AutoLabeler
from vision_ml.utils.config import load_config

config = load_config('config/training/base.yaml')
labeler = AutoLabeler(config)

# Load pseudo-labels from DualDetector
count = labeler.load_dual_detector_frames('data/low_confidence_frames')
print(f"Loaded {count} pseudo-labels")

# Upload to Roboflow
labeler.flush(output_dir='data/auto_labeled')
# → Uploads to Roboflow workspace/project
```

### Phase 3: Human Review in Roboflow UI

1. Go to Roboflow → Your Project → Annotate
2. See auto-labeled images with RF-DETR predictions
3. For each image:
   - ✅ Approve (correct labels)
   - ❌ Reject (wrong labels, delete)
   - ✏️ Edit (fix bounding boxes)
4. Once reviewed, create new dataset version

### Phase 4: Train New Model

```python
from vision_ml.training.trainer import Trainer
from vision_ml.utils.config import load_config

config = load_config('config/training/base.yaml')
trainer = Trainer(config)

# Download latest dataset version from Roboflow
# (Trainer can auto-download via Roboflow API)
trainer.train(run_name='retrain_with_pseudo_labels')

# New model deployed → Primary detector improved → Fewer secondary calls
```

---

## 5. Advanced: Active Learning

Instead of uploading ALL pseudo-labels, upload only the **uncertain ones** for human review.

```python
from vision_ml.labeling.auto_labeler import AutoLabeler

labeler = AutoLabeler(config)
labeler.load_dual_detector_frames()

# Filter to only uncertain predictions (confidence 0.5-0.7)
uncertain = [
    label for label in labeler.pending_labels
    if 0.5 <= min(label['confidences']) < 0.7
]

# Upload only uncertain for review (faster)
labeler.pending_labels = uncertain
labeler.flush()
```

**Benefit**: Reduce human review time by 50-70% by focusing on edge cases.

---

## 6. Label Format Details

### COCO JSON (Roboflow Default)

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "frame_000000.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 50, 200, 300],  // [x, y, width, height]
      "area": 60000,
      "iscrowd": 0,
      "confidence": 0.95  // Optional: your confidence score
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    }
  ]
}
```

### Your AutoLabeler Output

```json
[
  {
    "image_id": "frame_000000",
    "image_path": "data/low_confidence_frames/frame_000000.jpg",
    "boxes": [[100, 50, 300, 350]],  // [x1, y1, x2, y2]
    "confidences": [0.92],
    "class_ids": [0]
  }
]
```

**Conversion to COCO** (if needed):

```python
def to_coco_format(auto_labels, image_dir):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "person"}]
    }
    
    ann_id = 1
    for img_id, label in enumerate(auto_labels):
        coco["images"].append({
            "id": img_id,
            "file_name": label["image_path"],
            "width": 1920,
            "height": 1080
        })
        
        for box, conf, cls_id in zip(label["boxes"], label["confidences"], label["class_ids"]):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "confidence": conf
            })
            ann_id += 1
    
    return coco
```

---

## 7. Roboflow API Usage

### Download Dataset

```python
from roboflow import Roboflow

rf = Roboflow(api_key="your_api_key")
project = rf.workspace("your_workspace").project("retail-visitor-detection")

# Download latest version
dataset = project.version(1).download("coco")
# → Downloads to ./retail-visitor-detection-1/
```

### Upload Annotations

```python
# Roboflow SDK doesn't directly upload annotations
# Instead: Use Roboflow UI or API endpoint

import requests

url = "https://api.roboflow.com/dataset/your_project/upload"
files = {
    "file": open("frame_000000.jpg", "rb"),
    "annotation": open("frame_000000.json", "rb")
}
params = {
    "api_key": "your_api_key",
    "name": "frame_000000"
}
response = requests.post(url, files=files, params=params)
```

### Query Model Predictions

```python
from inference import get_model

model = get_model("retail-visitor-detection/1")
predictions = model.infer("image.jpg", confidence=0.5)
# → Returns detections in Roboflow format
```

---

## 8. Best Practices

### 1. Version Your Datasets
```
retail-visitor-detection-v1: 500 images, manual labels
retail-visitor-detection-v2: v1 + 300 auto-labeled images
retail-visitor-detection-v3: v2 + 200 active-learning images
```

### 2. Track Label Provenance
```python
label = {
    "image_id": "frame_000000",
    "boxes": [...],
    "confidences": [...],
    "source": "rf-detr",  # Track which detector made the label
    "reviewed": False,    # Track if human reviewed
    "reviewer": None
}
```

### 3. Monitor Annotation Quality
```python
# After human review, compute agreement between auto-labels and human labels
from sklearn.metrics import jaccard_score

auto_boxes = [...]
human_boxes = [...]
iou = jaccard_score(auto_boxes, human_boxes)
print(f"Auto-annotation agreement: {iou:.2%}")
# If < 80%, retrain secondary detector
```

### 4. Automate Retraining
```python
# If secondary_ratio > 0.20, trigger retraining
if dual_detector.secondary_ratio > 0.20:
    print("Model degradation detected!")
    labeler.load_dual_detector_frames()
    labeler.flush()  # Upload to Roboflow
    # → Human reviews → New dataset version
    # → Trainer downloads & retrains
```

### 5. Cost Optimization
- **Free tier**: 5000 images/month
- **Pro tier**: Unlimited images
- **Tip**: Use active learning to reduce annotation volume by 5-10x

---

## 9. Troubleshooting

### Issue: "Roboflow API key not set"
```bash
export ROBOFLOW_API_KEY="your_key"
# Or set in config/training/base.yaml
```

### Issue: "Connection refused"
```python
# Check internet connection
# Verify API key is valid
# Check workspace/project names match
```

### Issue: "Labels not appearing in Roboflow UI"
- Ensure image files exist at specified paths
- Check label format matches COCO JSON spec
- Verify bounding boxes are within image bounds

### Issue: "Model accuracy low after retraining"
- Increase `auto_label_confidence` threshold (be more selective)
- Manually review auto-labels before training
- Use active learning to focus on edge cases

---

## 10. Integration with Your System

### Full Auto-Annotation Loop

```python
#!/usr/bin/env python3
"""End-to-end auto-annotation + retraining loop."""

from vision_ml.inference.pipeline import InferencePipeline
from vision_ml.labeling.auto_labeler import AutoLabeler
from vision_ml.training.trainer import Trainer
from vision_ml.utils.config import load_config

# Step 1: Inference with dual-detector
config = load_config('config/inference/base.yaml')
config['detection']['use_dual_detector'] = True

pipeline = InferencePipeline(config)
summary = pipeline.run_offline('retail_video.mp4')

print(f"[Loop] Secondary detector used: {summary['dual_detector']['secondary_ratio']:.1%}")

# Step 2: Check if retraining needed
if summary['dual_detector']['secondary_ratio'] > 0.20:
    print("[Loop] Model degradation detected! Collecting pseudo-labels...")
    
    # Step 3: Upload pseudo-labels to Roboflow
    config = load_config('config/training/base.yaml')
    labeler = AutoLabeler(config)
    labeler.load_dual_detector_frames()
    labeler.flush()
    
    print("[Loop] Pseudo-labels uploaded to Roboflow for review")
    print("[Loop] Waiting for human review in Roboflow UI...")
    input("Press Enter once you've reviewed labels in Roboflow...")
    
    # Step 4: Retrain with reviewed labels
    trainer = Trainer(config)
    trainer.train(run_name='auto_retrain_from_pseudo_labels')
    
    print("[Loop] Model retrained! Deploying...")
    # Deploy new model...
else:
    print("[Loop] Model is healthy, no retraining needed")
```

---

## Summary

**Roboflow** is the glue between your inference pipeline and training loop:

1. **DualDetector** saves low-confidence frames
2. **AutoLabeler** collects pseudo-labels from RF-DETR
3. **Roboflow** hosts the dataset and provides human review UI
4. **Trainer** downloads reviewed labels and retrains
5. **Loop repeats** → Self-improving system

**Key metrics to monitor**:
- `secondary_ratio` → Model health
- `frames_saved` → Annotation volume
- Human review time → Cost
- Model accuracy → Quality

This is the **auto-healing** system you wanted — no manual annotation, just human review of uncertain predictions.
