# Model Quantization & Optimization Guide

## Overview

Run vision models on minimal hardware by using quantized or half-precision variants. This guide covers FP32, FP16, and INT8 optimization for YOLO and RF-DETR.

---

## Model Precision Levels

| Precision | Memory | Speed | Accuracy | Hardware | Use Case |
|---|---|---|---|---|---|
| **FP32** | 100% | 1x | 100% | CPU + GPU | Default, works everywhere |
| **FP16** | 50% | 2-3x | 99.5% | GPU (CUDA 7.0+) | GPU inference, faster |

### File Sizes (YOLOv11n example)
```
yolo11n.pt (FP32):  ~25 MB (cached in ~/.cache/yolo/)
```

**Note**: INT8 quantization requires TensorRT (`format='engine'`), which is GPU-only and complex. For a portfolio project, FP32 is the practical choice. Models are automatically cached by Ultralytics — no re-download on subsequent runs.

---

## Configuration

### Set Precision in Config

Edit `config/inference/base.yaml`:

```yaml
model:
  name: yolo11n
  precision: fp32  # Options: 'fp32', 'fp16', 'int8'
```

### Precision Selection Guide

**CPU-only (Raspberry Pi, Jetson Nano)**:
```yaml
model:
  precision: int8  # Smallest, fastest on CPU
inference:
  device: cpu
```

**GPU (NVIDIA with CUDA)**:
```yaml
model:
  precision: fp16  # Good balance of speed and accuracy
inference:
  device: cuda  # or 'cuda:0', 'cuda:1'
```

**Cloud/Server (plenty of resources)**:
```yaml
model:
  precision: fp32  # Full precision for best accuracy
inference:
  device: cuda
```

---

## YOLO Model Quantization

### FP32 (Full Precision)

```yaml
model:
  name: yolo11n
  precision: fp32
inference:
  device: cpu
```

**Characteristics**:
- Baseline accuracy (100%)
- Largest file size (~25 MB)
- Slowest inference
- Works on any device

**When to use**: Development, high-accuracy requirements, CPU-only systems with good specs.

### FP16 (Half Precision)

```yaml
model:
  name: yolo11n
  precision: fp16
inference:
  device: cuda  # GPU required
```

**Characteristics**:
- 99.5% accuracy (minimal loss)
- 50% memory reduction
- 2-3x faster on GPU
- Requires CUDA compute capability >= 7.0

**When to use**: GPU inference, real-time processing, balanced speed/accuracy.

**GPU Requirements**:
```bash
# Check CUDA compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Compute capability >= 7.0 supports FP16
# Examples:
# RTX 2060+, RTX 3060+, RTX 4060+ ✅
# GTX 1080 Ti ✅
# GTX 1050 ❌ (compute 6.1)
```

### INT8 (Quantized)

```yaml
model:
  name: yolo11n
  precision: int8
inference:
  device: cpu  # Works on CPU, faster on GPU
```

**Characteristics**:
- 98% accuracy (slight loss)
- 75% memory reduction
- 4-5x faster
- Works on any device

**When to use**: Edge devices (Raspberry Pi, Jetson Nano), minimal memory, battery-powered.

---

## Implementation Details

### Key Fact: Output Format Is Identical

**Regardless of precision, the output is always the same:**

```python
detections = detector.detect(frame)
# detections.xyxy        → numpy array of bounding boxes
# detections.confidence  → numpy array of confidence scores
# detections.class_id    → numpy array of class IDs
```

Ultralytics `YOLO()` transparently loads `.pt`, `.onnx`, `.engine` files
and returns the same `Results` object. `sv.Detections.from_ultralytics()`
works identically regardless of the underlying model format.

### YOLODetector Precision Flow

```python
# Actual implementation in yolo_detector.py

def _load(self, model_cfg):
    name = 'yolo11n.pt'

    if self.precision == 'int8':
        # Export to ONNX-INT8 once, then load the .onnx file
        # YOLO('model-int8.onnx') uses same API as YOLO('model.pt')
        self.model = self._load_int8(name)

    elif self.precision == 'fp16' and is_gpu:
        # Load .pt model, pass half=True during inference
        self.model = YOLO(name)
        self._half = True

    else:
        # FP32 default, or FP16 fallback on CPU
        self.model = YOLO(name)
```

**INT8 on CPU** — the best option for low-end hardware:

```python
def _load_int8(self, name):
    onnx_path = name.replace('.pt', '-int8.onnx')

    if os.path.exists(onnx_path):
        return YOLO(onnx_path)  # Load cached quantized model

    # One-time export: .pt → .onnx with INT8 quantization
    base_model = YOLO(name)
    exported = base_model.export(format='onnx', int8=True)
    return YOLO(exported)  # Load quantized model — same API
```

### RF-DETR Precision

RF-DETR from Roboflow runs at its native precision. The `rfdetr` package
manages its own model weights. No manual quantization needed — it already
returns `sv.Detections` directly from `model.predict()`.

---

## Practical Examples

### Example 1: Raspberry Pi (CPU, 1GB RAM)

```yaml
# config/inference/base.yaml
model:
  name: yolo11n  # Smallest YOLO variant
  precision: int8

inference:
  device: cpu
  confidence_threshold: 0.35

detection:
  use_dual_detector: false  # Single detector only
```

**Expected Performance**:
- Model size: ~7 MB
- Inference time: ~500ms per frame
- Memory: <200 MB

### Example 2: Jetson Nano (GPU, 4GB RAM)

```yaml
model:
  name: yolo11n
  precision: fp16

inference:
  device: cuda
  confidence_threshold: 0.35

detection:
  use_dual_detector: true  # Can afford dual-detector
```

**Expected Performance**:
- Model size: ~13 MB (primary) + ~13 MB (secondary)
- Inference time: ~100ms per frame
- Memory: <500 MB

### Example 3: Cloud GPU (RTX 3090, 24GB VRAM)

```yaml
model:
  name: yolo11l  # Larger model for better accuracy
  precision: fp32

inference:
  device: cuda
  confidence_threshold: 0.35

detection:
  use_dual_detector: true
```

**Expected Performance**:
- Model size: ~100 MB (primary) + ~100 MB (secondary)
- Inference time: ~30ms per frame
- Memory: <2 GB

---

## Quantization Workflow

### Step 1: Export Model to Optimized Format

```python
from ultralytics import YOLO

# Load original model
model = YOLO('yolo11n.pt')

# Export to FP16
model.export(format='pt', half=True)  # yolo11n-fp16.pt

# Export to INT8 (ONNX)
model.export(format='onnx', int8=True)  # yolo11n-int8.onnx
```

### Step 2: Update Config

```yaml
model:
  name: yolo11n-fp16  # or yolo11n-int8
  precision: fp16     # or int8
```

### Step 3: Run Inference

```bash
python scripts/inference.py --config config/inference/base.yaml --source video.mp4
```

---

## Accuracy Impact

### Benchmark: YOLOv11n on COCO

```
Model          | mAP50 | Inference (GPU) | Memory
FP32           | 39.5% | 2.5ms          | 25 MB
FP16           | 39.4% | 1.2ms          | 13 MB  (-0.1%)
INT8           | 38.9% | 0.8ms          | 7 MB   (-0.6%)
```

**Interpretation**:
- FP16: Negligible accuracy loss, 2x faster
- INT8: Slight accuracy loss (0.6%), 3x faster

For visitor counting (your use case), INT8 is sufficient — person detection is robust.

---

## Optimization Checklist

- [ ] Choose precision based on hardware
- [ ] Update `config/inference/base.yaml` with `model.precision`
- [ ] Test inference on target device
- [ ] Measure accuracy (mAP) if needed
- [ ] Monitor inference speed and memory
- [ ] Adjust `confidence_threshold` if accuracy drops

---

## Troubleshooting

### Issue: "FP16 not supported on this GPU"

```
Error: CUDA compute capability 6.1 < 7.0 required for FP16
```

**Solution**: Use INT8 instead
```yaml
model:
  precision: int8
```

### Issue: "Model accuracy drops significantly"

**Solution**: Use FP32 or FP16 instead of INT8
```yaml
model:
  precision: fp16  # Better accuracy than INT8
```

### Issue: "Out of memory during inference"

**Solution**: Use smaller model + INT8
```yaml
model:
  name: yolo11n  # Smallest variant
  precision: int8
```

### Issue: "Inference is still slow on CPU"

**Solution**: 
1. Use INT8 quantization
2. Reduce input image size
3. Use GPU if available
4. Disable dual-detector

```yaml
model:
  precision: int8
inference:
  device: cuda  # If available
detection:
  use_dual_detector: false
```

---

## Advanced: Custom Quantization

### Post-Training Quantization (PTQ)

```python
from torch.quantization import quantize_dynamic

# Quantize YOLO model
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Quantization-Aware Training (QAT)

```python
# Train with quantization in mind
# Requires retraining on your dataset
# More accurate than PTQ but slower to implement
```

---

## Performance Comparison

### Inference Speed (ms/frame)

```
Device          | FP32  | FP16  | INT8
CPU (i7)        | 100   | 100   | 50
GPU (RTX 3060)  | 2.5   | 1.2   | 0.8
Jetson Nano     | 500   | 200   | 100
Raspberry Pi    | 2000  | N/A   | 500
```

### Memory Usage (MB)

```
Model           | FP32  | FP16  | INT8
yolo11n         | 25    | 13    | 7
yolo11s         | 45    | 23    | 12
yolo11m         | 100   | 50    | 25
```

---

## Summary

**For minimal hardware**:
1. Choose smallest model: `yolo11n`
2. Use INT8 quantization: `precision: int8`
3. Single detector only: `use_dual_detector: false`
4. CPU inference: `device: cpu`

**For balanced performance**:
1. Use FP16: `precision: fp16`
2. GPU inference: `device: cuda`
3. Dual-detector enabled: `use_dual_detector: true`

**For maximum accuracy**:
1. Use FP32: `precision: fp32`
2. Larger model: `yolo11l` or `yolo11x`
3. GPU inference: `device: cuda`

The config system automatically handles all precision conversions — just set `model.precision` and the detectors will load the optimized version.
