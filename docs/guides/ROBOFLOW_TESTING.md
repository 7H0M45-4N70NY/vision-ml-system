# Roboflow Upload Testing Guide

## Setup

1. **Get Roboflow API Key**
   - Go to https://roboflow.com → Account Settings → API Key
   - Copy your API key

2. **Set Environment Variable**
   ```bash
   export ROBOFLOW_API_KEY="your_api_key_here"
   ```

3. **Update Config** (optional, can use env var)
   Edit `config/inference/base.yaml`:
   ```yaml
   labeling:
     enabled: true
     provider: roboflow
     roboflow_api_key: null  # Falls back to env var
     roboflow_workspace: your_workspace_name
     roboflow_project: your_project_name
   ```

## Testing Roboflow Upload

### Option 1: Via Dashboard (Recommended)

1. **Run dashboard:**
   ```bash
   streamlit run home.py
   ```

2. **Go to Auto-Labeling page** (🏷️ Auto-Labeling)

3. **Load Frames:**
   - Click "📂 Load Frames" → Should load frames from `data/low_confidence_frames/`

4. **Upload to Roboflow:**
   - Click "☁️ Upload to Roboflow"
   - Check for success message
   - Go to Roboflow UI to verify frames uploaded

### Option 2: Via CLI Script

Create `test_roboflow.py`:
```python
import os
from src.vision_ml.utils.config import load_config
from src.vision_ml.labeling.auto_labeler import AutoLabeler

# Load config
config = load_config('config/inference/base.yaml')
config['labeling']['provider'] = 'roboflow'

# Create labeler
labeler = AutoLabeler(config)

# Load low-confidence frames
count = labeler.load_dual_detector_frames('data/low_confidence_frames')
print(f"Loaded {count} low-confidence frames")

# Upload to Roboflow
labeler.flush(output_dir='data/auto_labeled')
print(f"Flushed {len(labeler.pending_labels)} labels to Roboflow")
```

Run:
```bash
python test_roboflow.py
```

## Debugging

### Issue: "Roboflow API key not set"
**Solution:** Make sure environment variable is set:
```bash
echo $ROBOFLOW_API_KEY
```
If empty, set it:
```bash
export ROBOFLOW_API_KEY="your_key"
```

### Issue: "Workspace or project not configured"
**Solution:** Update config:
```yaml
roboflow_workspace: your_workspace_name  # From roboflow.com account
roboflow_project: your_project_name      # From project URL
```

### Issue: "Uploaded 0/N labels (image files not found)"
**Solution:** Make sure low-confidence frames are saved:
```bash
ls data/low_confidence_frames/
# Should show: frame_*.jpg and frame_*.json files
```

### Issue: Network timeout or 403 error
**Solution:** Check Roboflow API key validity:
1. Visit https://roboflow.com/api
2. Verify the key is active
3. Try uploading one image manually to test

## What Gets Uploaded

Each low-confidence frame upload includes:
- **Image**: `frame_000000.jpg` (the actual frame)
- **Annotation**: YOLO format `.txt` file with bounding boxes

Roboflow will:
1. Ingest the images
2. Store annotations
3. Provide review UI for corrections
4. Allow dataset versioning
5. Export in any format (YOLO, Pascal VOC, etc.)

## After Upload

1. **Review in Roboflow UI:**
   - Go to project → Images
   - Check that frames and bboxes loaded correctly
   - Correct any wrong annotations

2. **Create Dataset Version:**
   - Click "Generate" → Dataset version
   - Select "Auto-Orient: ✓", "Augmentation: ✓" (optional)
   - Export as YOLO v8 format

3. **Download & Retrain:**
   - Copy the download link
   - Update `config/training/base.yaml` with dataset YAML
   - Run training: `python scripts/train.py --config config/training/base.yaml`

## Success Checklist

- [x] Low-confidence frames saved in `data/low_confidence_frames/`
- [x] Roboflow API key set
- [x] Dashboard auto-labeling page loads frames
- [x] Upload button completes without error
- [x] Frames appear in Roboflow UI
- [x] Annotations are correct (bounding boxes visible)
- [x] Can create dataset version in Roboflow
- [x] Can download and retrain with new dataset
