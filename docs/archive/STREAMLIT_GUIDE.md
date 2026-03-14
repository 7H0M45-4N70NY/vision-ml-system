# Streamlit Frontend Guide

## Overview

The Streamlit app provides a web UI for:
1. **Inference Mode**: Run inference on videos with dual-detector
2. **Auto-Labeling Mode**: Load pseudo-labels and export/upload to Roboflow

No complex setup needed — just run and use.

---

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the App

```bash
# Option 1: Via main.py
python main.py

# Option 2: Direct Streamlit
streamlit run app.py

# Option 3: With custom port
streamlit run app.py --server.port 8501
```

The app will open at `http://localhost:8501`

---

## Features

### 🎯 Inference Mode

**Input**: Video file (MP4, AVI, MOV)

**Process**:
1. Upload video
2. Configure dual-detector settings
3. Click "Run Inference"
4. View results:
   - Annotated video
   - Analytics JSON
   - Dual-detector stats

**Dual-Detector Stats**:
- **Total Frames**: # of frames processed
- **Secondary Calls**: # of times RF-DETR was used
- **Secondary Ratio**: % of frames needing secondary (health metric)
- **Frames Saved**: # of pseudo-labeled frames for training

**Health Indicator**:
- ✅ Secondary ratio < 20% → Model is healthy
- ⚠️ Secondary ratio > 20% → Model degrading, consider retraining

**Output Files**:
```
runs/inference/
  ├── output.mp4           (annotated video)
  ├── analytics.json       (stats + dual-detector info)
  └── ...

data/low_confidence_frames/
  ├── frame_000000.jpg
  ├── frame_000000.json    (RF-DETR labels)
  └── ...
```

### 🏷️ Auto-Labeling Mode

**Step 1: Load Pseudo-Labels**
- Specify frame directory (default: `data/low_confidence_frames`)
- Click "Load Frames"
- Shows count of loaded pseudo-labels

**Step 2: Preview Labels**
- Optional: View first 3 labels in JSON format
- Verify RF-DETR predictions look correct

**Step 3: Export or Upload**
- **Export Local**: Save to `data/auto_labeled/auto_labels.json`
- **Upload to Roboflow**: Send to Roboflow for human review

**Workflow**:
```
Load Frames → Preview → Export/Upload → Human Review → Retrain
```

---

## Configuration

### Inference Settings

Edit `config/inference/base.yaml`:

```yaml
detection:
  use_dual_detector: true
  primary_detector: yolo11n
  secondary_detector: rfdetr
  dual_confidence_threshold: 0.5  # Trigger secondary if < 0.5
  frame_save_dir: data/low_confidence_frames
```

### Auto-Labeling Settings

Edit `config/training/base.yaml`:

```yaml
labeling:
  enabled: true
  provider: local  # or 'roboflow'
  auto_label_confidence: 0.7
  roboflow_api_key: null  # Read from env var
  roboflow_workspace: your_workspace
  roboflow_project: your_project
```

### Roboflow Setup

```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

---

## Workflow Examples

### Example 1: Local Inference Only

```
1. Upload video.mp4
2. Use Dual-Detector: ON
3. Run Inference
4. View stats
5. Download output.mp4
```

**No Roboflow needed** — just local processing.

### Example 2: Full Auto-Annotation Loop

```
1. Run Inference (saves low-confidence frames)
2. Switch to Auto-Labeling Mode
3. Load Frames from data/low_confidence_frames
4. Upload to Roboflow
5. Review in Roboflow UI
6. Create dataset version
7. Retrain model
```

### Example 3: Active Learning (Uncertain Predictions Only)

```python
# In auto-labeling mode, filter to only uncertain predictions
# (confidence 0.5-0.7) before uploading
# → Reduces human review time by 50-70%
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

```bash
pip install streamlit>=1.28.0
```

### Issue: "Port 8501 already in use"

```bash
streamlit run app.py --server.port 8502
```

### Issue: "No frames saved during inference"

- Check `dual_confidence_threshold` (default 0.5)
- Increase it to trigger secondary more often
- Or primary model is too confident (good sign!)

### Issue: "Roboflow upload fails"

```bash
# Check API key
echo $ROBOFLOW_API_KEY

# Verify workspace/project names in config
# Go to Roboflow UI → Settings → copy exact names
```

### Issue: "Video processing is slow"

- Reduce video resolution before upload
- Use smaller model (yolo11n is already minimal)
- GPU acceleration: ensure CUDA is installed

---

## Future Enhancements

### 1. Data Drift Detection (Evidently AI)

Monitor model performance over time:

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Compare current inference stats vs baseline
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=baseline, current_data=current)

# Alert if drift detected
if report.show()['data_drift']['data_drift']:
    st.warning("⚠️ Data drift detected! Retrain recommended.")
```

**When to use**:
- Monitor `secondary_ratio` over time
- Alert if it increases > 5% from baseline
- Trigger retraining automatically

### 2. MLflow Experiment Tracking

Log inference runs and model versions:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("dual_detector", True)
    mlflow.log_metric("secondary_ratio", 0.15)
    mlflow.log_metric("frames_saved", 150)
    mlflow.log_artifact("runs/inference/analytics.json")
```

**Benefits**:
- Track all inference runs
- Compare model versions
- Reproduce results

### 3. Roboflow Model Registry

Deploy trained models via Roboflow:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="your_key")
project = rf.workspace("your_workspace").project("your_project")

# Deploy model version
project.version(2).deploy(model_type="yolov8", model_format="onnx")

# Use in inference
model = project.version(2).model
predictions = model.infer("image.jpg")
```

**Benefits**:
- Version control for models
- Easy rollback
- A/B testing support

### 4. Multi-Task Detection

Extend to multiple detection tasks (e.g., person + vehicle):

```yaml
detection:
  tasks:
    - name: person_detection
      primary: yolo11n
      secondary: rfdetr
      classes: [0]  # Person class
    
    - name: vehicle_detection
      primary: yolo11n
      secondary: rfdetr
      classes: [2, 5, 7]  # Car, Bus, Truck
```

**Note**: Roboflow registry works with multi-task if each task has separate project.

### 5. Training Pipeline UI

Add training controls to Streamlit:

```python
st.markdown("### 🔄 Retrain Model")
if st.button("Train with Reviewed Labels"):
    with st.spinner("Training..."):
        trainer = Trainer(config)
        trainer.train(run_name='streamlit_retrain')
        st.success("✅ Model trained!")
```

### 6. Real-Time Webcam Inference

Stream from webcam instead of video file:

```python
import streamlit_webrtc as webrtc

webrtc_ctx = webrtc.webrtc_streamer(
    key="inference",
    video_processor_factory=YOLOVideoProcessor,
)
```

---

## Architecture

```
Streamlit UI (app.py)
    │
    ├─ Inference Mode
    │   └─ InferencePipeline
    │       ├─ DualDetector (YOLO + RF-DETR)
    │       ├─ Tracker (ByteTrack)
    │       └─ Analytics (visitor counting)
    │
    └─ Auto-Labeling Mode
        └─ AutoLabeler
            ├─ load_dual_detector_frames()
            ├─ flush() → Local or Roboflow
            └─ Integration with Roboflow API
```

---

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is installed for faster inference
2. **Model Caching**: Detectors are cached (singleton) — no reload overhead
3. **Video Resolution**: Smaller videos process faster
4. **Batch Processing**: Use offline mode for large videos

---

## File Structure

```
vision-ml-system/
├── app.py                       # Streamlit app
├── main.py                      # Launcher
├── STREAMLIT_GUIDE.md          # This file
├── QUICKSTART.md               # Quick start
├── docs/ROBOFLOW_GUIDE.md      # Roboflow reference
├── config/
│   ├── inference/base.yaml
│   └── training/base.yaml
├── data/
│   ├── low_confidence_frames/  # Saved by DualDetector
│   └── auto_labeled/           # Exported by AutoLabeler
└── src/vision_ml/
    ├── inference/pipeline.py
    ├── detection/
    │   ├── dual_detector.py
    │   ├── detector_factory.py
    │   ├── yolo_detector.py
    │   └── rfdetr_detector.py
    └── labeling/auto_labeler.py
```

---

## Summary

**Streamlit app** is your portfolio project's UI:
- ✅ Minimal, clean interface
- ✅ Inference + auto-labeling in one place
- ✅ Real-time stats and health monitoring
- ✅ One-click export/upload to Roboflow
- ✅ Ready for future enhancements (drift detection, MLflow, etc.)

**Next steps**:
1. Run `python main.py` or `streamlit run app.py`
2. Upload a video
3. View dual-detector stats
4. Export pseudo-labels
5. Review in Roboflow UI
6. Retrain model

That's it! The auto-healing loop is now accessible via a web UI.
